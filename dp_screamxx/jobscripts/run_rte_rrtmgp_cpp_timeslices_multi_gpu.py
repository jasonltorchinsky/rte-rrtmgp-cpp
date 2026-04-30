#!/usr/bin/env python3
"""
Parallel time-slice driver for RTE-RRTMGP-CPP across multiple GPUs.

Slurm-friendly logging:
  - Prints timestamped progress lines (flush=True) so output appears promptly in slurm-%j.out
  - Captures the executable stdout/stderr and writes it into the same stream, prefixed by GPU/worker
  - Also writes per-task logs into the per-GPU work directory for post-mortem

Concurrency model:
  - One worker process per GPU
  - Shared task queue of (input_file, time_index)
  - Each worker uses a private working directory containing required symlinks and input/output filenames
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Event, JoinableQueue, Process, current_process
from pathlib import Path
from typing import List, Tuple

import xarray as xr


LR_RE = re.compile(r"(?:^|\.)(lr_(\d+))(?:\.|$)")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
CASE_RE = re.compile(r"^(.*?)(?:\.scream\.|\.Scream\.|\.SCREAM\.|\.scream_|\.SCREAM_|\.Scream_)")


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, *, stream=sys.stdout) -> None:
    print(f"[{ts()}] {msg}", file=stream, flush=True)


def parse_lr_tag(name: str) -> Tuple[str, int]:
    m = LR_RE.search(name)
    if not m:
        return ("lr_00", 0)
    return (m.group(1), int(m.group(2)))


def parse_date(name: str) -> str:
    m = DATE_RE.search(name)
    return m.group(1) if m else "unknown-date"


def parse_case(name: str) -> str:
    base = Path(name).name
    m = CASE_RE.match(base)
    if m:
        return m.group(1)
    return base.split(".")[0]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_gpus(s: str) -> List[str]:
    s = s.strip()
    if s.startswith("["):
        vals = json.loads(s)
        return [str(v) for v in vals]
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def safe_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)


def setup_run_directory(workdir: Path, rrtmgp_data_dir: Path, rte_data_dir: Path) -> None:
    ensure_dir(workdir)

    safe_symlink(rrtmgp_data_dir / "rrtmgp-clouds-sw.nc", workdir / "cloud_coefficients_sw.nc")
    safe_symlink(rrtmgp_data_dir / "rrtmgp-clouds-lw.nc", workdir / "cloud_coefficients_lw.nc")
    safe_symlink(rrtmgp_data_dir / "rrtmgp-gas-sw-g224.nc", workdir / "coefficients_sw.nc")
    safe_symlink(rrtmgp_data_dir / "rrtmgp-gas-lw-g256.nc", workdir / "coefficients_lw.nc")
    safe_symlink(rte_data_dir / "aerosol_optics.nc", workdir / "aerosol_optics.nc")


def write_single_timestep(src_path: Path, t_index: int, dst_path: Path) -> None:
    ds = xr.open_dataset(src_path, decode_times=False)
    try:
        if "time" not in ds.dims:
            raise RuntimeError(f"{src_path} has no 'time' dimension")
        nt = int(ds.sizes["time"])
        if t_index < 0 or t_index >= nt:
            raise IndexError(f"t_index {t_index} out of bounds for {src_path} (time={nt})")

        ds_t = ds.isel(time=t_index, drop=True)   # key change

        if dst_path.exists():
            dst_path.unlink()
        ds_t.to_netcdf(dst_path)
    finally:
        ds.close()


def run_executable_capture(
    exe_dir: Path,
    gpu: str,
    raytracing: int,
    extra_args: list[str],
    cwd: Path,
    prefix: str,
    per_task_log: Path,
) -> int:
    exe = exe_dir / "test_rte_rrtmgp_rt_gpu"
    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")

    cmd = [str(exe), "--cloud-optics", "--raytracing", str(raytracing), *extra_args]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Merge stderr into stdout so we preserve ordering in the slurm output.
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    ensure_dir(per_task_log.parent)
    with per_task_log.open("w") as f_log:
        f_log.write(f"[{ts()}] CMD: {' '.join(cmd)}\n")
        f_log.write(f"[{ts()}] CWD: {cwd}\n")
        f_log.write(f"[{ts()}] CUDA_VISIBLE_DEVICES={gpu}\n")

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            # Slurm output
            log(f"{prefix} | {line}")
            # Per-task file
            f_log.write(line + "\n")

    return proc.wait()


def build_output_path(output_root: Path, input_file: Path, t_index: int) -> Path:
    base = input_file.name
    lr_tag, _ = parse_lr_tag(base)

    # strip .nc
    stem = base[:-3] if base.endswith(".nc") else base
    # strip trailing ".in"
    if stem.endswith(".in"):
        stem = stem[:-3]
    # strip the FIRST lr tag occurrence (e.g., ".lr_32") from the stem
    stem = stem.replace(f".{lr_tag}", "", 1)

    out_name = f"{stem}.t_{t_index:03d}.{lr_tag}.out.nc"
    ensure_dir(output_root)
    return output_root / out_name


@dataclass(frozen=True)
class Task:
    input_file: str
    t_index: int
    nt: int


def worker_loop(
    gpu: str,
    tasks: JoinableQueue,
    stop_evt: Event,
    exe_dir: Path,
    out_dir: Path,
    work_root: Path,
    rrtmgp_data_dir: Path,
    rte_data_dir: Path,
    raytracing: int,
    extra_args: list[str],
    dry_run: bool,
) -> None:
    proc_name = current_process().name
    workdir = work_root / f"gpu_{gpu}"
    input_path = workdir / "rte_rrtmgp_input.nc"
    out_path = workdir / "rte_rrtmgp_output.nc"
    logs_dir = workdir / "logs"

    prefix_base = f"{proc_name} GPU={gpu}"

    try:
        if not dry_run:
            setup_run_directory(workdir, rrtmgp_data_dir, rte_data_dir)
        log(f"{prefix_base} | initialized workdir={workdir}")
    except Exception as e:
        stop_evt.set()
        log(f"{prefix_base} | ERROR during setup: {e}", stream=sys.stderr)
        return

    while not stop_evt.is_set():
        try:
            task = tasks.get(timeout=0.5)
        except Exception:
            continue

        if task is None:
            tasks.task_done()
            log(f"{prefix_base} | received sentinel; exiting")
            break

        f = Path(task.input_file)
        t = task.t_index
        nt = task.nt
        prefix = f"{prefix_base} | {f.name} t={t:03d}/{nt-1:03d}"

        t0 = time.time()
        try:
            log(f"{prefix} | START")

            if dry_run:
                final_out = build_output_path(out_dir, f, t)
                log(f"{prefix} | DRY RUN: would write {input_path}")
                log(f"{prefix} | DRY RUN: would run executable in {workdir}")
                log(f"{prefix} | DRY RUN: would move {out_path} -> {final_out}")
                tasks.task_done()
                continue

            # Write timestep input
            log(f"{prefix} | writing input slice -> {input_path}")
            write_single_timestep(f, t, input_path)

            # Run executable (capture output to slurm + per-task log file)
            if out_path.exists():
                out_path.unlink()

            per_task_log = logs_dir / f"{f.name}.t_{t:03d}.log.txt"
            log(f"{prefix} | running executable (raytracing={raytracing})")
            rc = run_executable_capture(
                exe_dir=exe_dir,
                gpu=gpu,
                raytracing=raytracing,
                extra_args=extra_args,
                cwd=workdir,
                prefix=prefix,
                per_task_log=per_task_log,
            )

            if rc != 0:
                raise RuntimeError(f"Executable returned nonzero exit code {rc}")

            if not out_path.exists():
                raise FileNotFoundError(f"Expected output not found: {out_path}")

            final_out = build_output_path(out_dir, f, t)
            log(f"{prefix} | moving output -> {final_out}")
            if final_out.exists():
                final_out.unlink()
            shutil.move(str(out_path), str(final_out))

            dt = time.time() - t0
            log(f"{prefix} | DONE in {dt:.2f}s")

        except Exception as e:
            stop_evt.set()
            log(f"{prefix} | ERROR: {e}", stream=sys.stderr)
        finally:
            tasks.task_done()


def iter_inputs_sorted(input_dir: Path, glob_pat: str) -> List[Path]:
    inputs = sorted(input_dir.glob(glob_pat))
    if not inputs:
        raise FileNotFoundError(f"No files matched {glob_pat} in {input_dir}")

    def sort_key(p: Path):
        _, lr_num = parse_lr_tag(p.name)
        return (-lr_num, p.name)

    inputs.sort(key=sort_key)
    return inputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RTE-RRTMGP-CPP per-time-slice across multiple GPUs with Slurm-friendly logging.")
    ap.add_argument("--rte-rrtmgp-cpp-input-dir", required=True, type=Path)
    ap.add_argument("--rte-rrtmgp-cpp-executable-dir", required=True, type=Path)
    ap.add_argument("--rte-rrtmgp-cpp-output-dir", required=True, type=Path)
    ap.add_argument("--lr", type=str, default=None,
        help="Comma-separated list of lr tags to plot, e.g. '01,04,16'. "
             "If omitted, plot all available lr_XX pairs.",
    )

    ap.add_argument("--rrtmgp-data-dir", required=True, type=Path,
                    help="Directory containing rrtmgp-clouds-*.nc and rrtmgp-gas-*.nc.")
    ap.add_argument("--rte-data-dir", required=True, type=Path,
                    help="Directory containing aerosol_optics.nc.")

    ap.add_argument("--gpus", required=True, help='GPU list, e.g. "0,1,2" or "[0,1,2]".')
    ap.add_argument("--raytracing", type=int, default=128)
    ap.add_argument("--extra-exe-args", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed through to the executable (after '--').")
    ap.add_argument("--glob", default="*.nc")
    ap.add_argument("--work-dir", default=".rte_rrtmgp_work", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_dir = args.rte_rrtmgp_cpp_input_dir
    exe_dir = args.rte_rrtmgp_cpp_executable_dir
    out_dir = args.rte_rrtmgp_cpp_output_dir
    rrtmgp_data_dir = args.rrtmgp_data_dir
    rte_data_dir = args.rte_data_dir
    work_root = args.work_dir
    gpus = parse_gpus(args.gpus)

    log(f"Driver START pid={os.getpid()} gpus={gpus}")
    log(f"Input dir: {in_dir}")
    log(f"Executable dir: {exe_dir}")
    log(f"Output dir: {out_dir}")
    log(f"Work dir: {work_root}")
    log(f"RRTMGP data dir: {rrtmgp_data_dir}")
    log(f"RTE data dir: {rte_data_dir}")

    for p in [in_dir, exe_dir, rrtmgp_data_dir, rte_data_dir]:
        if not p.is_dir():
            raise NotADirectoryError(f"Directory not found: {p}")

    ensure_dir(out_dir)
    ensure_dir(work_root)

    inputs = iter_inputs_sorted(in_dir, args.glob)
    log(f"Discovered {len(inputs)} input files (sorted by lr desc).")

    # Optional filtering by requested lr list
    if args.lr is not None:
        requested = [s.strip() for s in args.lr.split(",") if s.strip()]
        requested_tags = []
        for s in requested:
            # accept '01' or 'lr_01'
            if s.startswith("lr_"):
                requested_tags.append(s)
            else:
                requested_tags.append(f"lr_{s}")

        requested_tags = set(requested_tags)

        filtered_inputs = []
        for f in inputs:
            lr_tag, _ = parse_lr_tag(f.name)
            if lr_tag in requested_tags:
                filtered_inputs.append(f)

        log(f"Filtering by --lr: requested={sorted(requested_tags)}")
        log(f"Kept {len(filtered_inputs)} of {len(inputs)} input files after lr filtering.")

        if not filtered_inputs:
            raise FileNotFoundError(
                f"No input files matched requested lr tags {sorted(requested_tags)} in {in_dir}"
            )

        inputs = filtered_inputs
    q: JoinableQueue = JoinableQueue()
    stop_evt = Event()

    total_tasks = 0
    for f in inputs:
        ds = xr.open_dataset(f, decode_times=False)
        if "time" not in ds.dims:
            ds.close()
            raise RuntimeError(f"{f} has no 'time' dimension")
        nt = int(ds.dims["time"])
        ds.close()

        for t in range(nt):
            q.put(Task(str(f), t, nt))
            total_tasks += 1

        log(f"Queued {nt} tasks from {f.name}")

    log(f"Total tasks queued: {total_tasks}")

    procs: List[Process] = []
    for gpu in gpus:
        p = Process(
            target=worker_loop,
            args=(
                gpu, q, stop_evt,
                exe_dir, out_dir, work_root,
                rrtmgp_data_dir, rte_data_dir,
                args.raytracing, args.extra_exe_args, args.dry_run
            ),
            name=f"worker-gpu{gpu}",
            daemon=False,
        )
        p.start()
        procs.append(p)
        log(f"Started worker pid={p.pid} for GPU={gpu}")

    # Sentinels
    for _ in procs:
        q.put(None)

    q.join()
    log("All queued tasks marked done; joining workers...")

    for p in procs:
        p.join()

    # Fail job if any worker hit an error
    if stop_evt.is_set():
        log("Driver EXIT with errors (stop flag set).", stream=sys.stderr)
        raise SystemExit(2)

    log("Driver DONE successfully.")


if __name__ == "__main__":
    main()