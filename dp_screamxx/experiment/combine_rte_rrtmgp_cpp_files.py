#!/usr/bin/env python3
"""
MPI-parallel combine of RTE NetCDF files into time-stacked outputs, using DP-SCREAM
time coordinates mapped from filename stem + timestep index.

Features
--------
- Supports multiple input directories via repeated --input-dir.
- Supports multiple DP-SCREAM files via repeated --dpscream-file.
- Handles missing timestep indices naturally.
- Final output contains only times for files actually present.
- Duplicate timestep within an output group: warn, keep first.
- Parallelizes within a group by assigning files/time slices across MPI ranks.
- Writes to an intermediate Zarr store by region, then exports final NetCDF.
- Reads DP-SCREAM metadata only on rank 0, then broadcasts to all ranks.

If you need help using SandiaAI Chat: https://wp.sandia.gov/sandia-ai-chat/how-to-use/
"""

import argparse
import os
import re
import shutil
import sys
import time
import uuid
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

GRID_SCALARS = ("ngrid_x", "ngrid_y", "ngrid_z")

T_RE = re.compile(r"\.t_(\d+)\.")
LR_RE = re.compile(r"\.lr_(\d+)\.")
KIND_RE = re.compile(r"\.(in|out)\.nc$")
BASE_RE = re.compile(r"^(.*)\.t_\d+\.lr_\d+\.(?:in|out)\.nc$")


def get_mpi():
    try:
        from mpi4py import MPI  # type: ignore
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


COMM, RANK, NRANKS = get_mpi()


def log(msg: str, *, every_rank: bool = False):
    if (not every_rank) and RANK != 0:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    host = os.uname().nodename
    print(f"[{ts}] [rank {RANK}/{NRANKS} @ {host}] {msg}", flush=True)


def barrier():
    if COMM is not None:
        COMM.Barrier()


def t_from_name(p: Path) -> int:
    m = T_RE.search(p.name)
    if not m:
        raise ValueError(f"Could not parse timestep from filename: {p}")
    return int(m.group(1))


def lr_from_name(p: Path) -> str:
    m = LR_RE.search(p.name)
    if not m:
        raise ValueError(f"Could not parse lr from filename: {p}")
    return m.group(1)


def kind_from_name(p: Path) -> str:
    m = KIND_RE.search(p.name)
    if not m:
        raise ValueError(f"Could not parse kind from filename: {p}")
    return m.group(1)


def base_from_rte_name(p: Path) -> str:
    m = BASE_RE.match(p.name)
    if not m:
        raise ValueError(f"Could not parse base stem from filename: {p}")
    return m.group(1)


def combined_name_from_example(p: Path) -> str:
    return re.sub(r"\.t_\d+", "", p.name)


def dps_base_from_name(p: Path) -> str:
    if not p.name.endswith(".nc"):
        raise ValueError(f"DP-SCREAM file does not end with .nc: {p}")
    return p.name[:-3]


def _safe_rmtree(path: Path):
    try:
        shutil.rmtree(path, ignore_errors=False)
    except FileNotFoundError:
        return
    except NotADirectoryError:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _unique_tmp_for(path: Path) -> Path:
    suffix = f".rank{RANK}.pid{os.getpid()}.{uuid.uuid4().hex}.tmp"
    return path.with_name(path.name + suffix)


def _atomic_commit_dir(src: Path, dst: Path, *, replace_existing: bool):
    if dst.exists():
        if not replace_existing:
            raise RuntimeError(
                f"Destination already exists: {dst}\n"
                "Refusing to overwrite. Use --replace-existing to overwrite explicitly."
            )
        _safe_rmtree(dst)
    os.rename(src, dst)


@contextmanager
def _mkdir_lock(lock_dir: Path, timeout_s: float = 3600.0, poll_s: float = 0.2, quiet: bool = False):
    t0 = time.time()
    while True:
        try:
            lock_dir.mkdir(parents=False, exist_ok=False)
            try:
                (lock_dir / "owner.txt").write_text(
                    f"rank={RANK} pid={os.getpid()} host={os.uname().nodename} time={time.time()}\n"
                )
            except Exception:
                pass
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            if not quiet:
                log(f"Waiting for lock: {lock_dir}", every_rank=True)
            time.sleep(poll_s)
    try:
        yield
    finally:
        _safe_rmtree(lock_dir)


def open_ds(path: Path, engine: Optional[str]) -> xr.Dataset:
    return xr.open_dataset(path, engine=engine, decode_timedelta=False, chunks=None)


def add_vertical_coords(ds: xr.Dataset) -> xr.Dataset:
    if "z" in ds.coords and "lay" in ds.dims and ds.sizes["lay"] == ds.sizes.get("z", -1):
        z_lay = ds["z"].values
        if len(z_lay) >= 2 and "lev" in ds.dims:
            dz = float(z_lay[1] - z_lay[0])
            z_lev = (float(z_lay[0]) - 0.5 * dz) + dz * np.arange(ds.sizes["lev"])
            ds = ds.assign_coords(
                lay=("lay", z_lay),
                lev=("lev", z_lev),
            )
    return ds


def out_path_for_group(example: Path, output_dir: Path, input_roots: List[Path], flat_output: bool) -> Path:
    out_name = combined_name_from_example(example)
    if flat_output:
        return output_dir / out_name

    for root in input_roots:
        try:
            rel_parent = example.parent.relative_to(root)
            return output_dir / rel_parent / out_name
        except Exception:
            continue

    return output_dir / out_name


def choose_chunks_for_var(da: xr.DataArray, time_chunk: int, xy_chunk: Optional[int]) -> Tuple[int, ...]:
    chunks = []
    for d in da.dims:
        n = da.sizes[d]
        if d == "time":
            chunks.append(min(time_chunk, n))
        elif d in ("x", "y"):
            chunks.append(n if xy_chunk is None else min(xy_chunk, n))
        else:
            chunks.append(n)
    return tuple(chunks)


def build_zarr_encoding(ds: xr.Dataset, time_chunk: int, xy_chunk: Optional[int]) -> Dict[str, dict]:
    enc = {}
    for name, da in ds.variables.items():
        if name in ds.dims:
            continue
        if da.ndim == 0:
            continue
        enc[name] = {"chunks": choose_chunks_for_var(da, time_chunk=time_chunk, xy_chunk=xy_chunk)}
    return enc


def list_rte_files(input_dirs: List[Path], pattern: str) -> List[Path]:
    out = []
    for d in input_dirs:
        if not d.exists():
            raise SystemExit(f"Input dir does not exist: {d}")
        out.extend([p for p in d.rglob(pattern) if p.is_file()])
    return out


def load_dpscream_time_map(dps_files: List[Path]) -> Dict[str, np.ndarray]:
    time_map = {}
    for p in dps_files:
        base = dps_base_from_name(p)
        try:
            with xr.open_dataset(p, decode_times=False, engine="netcdf4") as ds:
                if "time" not in ds.variables:
                    raise ValueError(f"DP-SCREAM file missing 'time': {p}")
                t = np.asarray(ds["time"].values, dtype=np.float64)
                units = ds["time"].attrs.get("units", "")
                if "day" not in str(units).lower():
                    log(f"Warning: DP-SCREAM time units for {p} are '{units}', expected day-based units.")
                time_map[base] = 24.0 * t
        except Exception as e:
            raise RuntimeError(f"Failed to read DP-SCREAM file with engine='netcdf4': {p}\n{e}") from e
    return time_map


def build_groups(
    input_dirs: List[Path],
    pattern: str,
    kind_filter: str,
    lr_filter: Optional[str],
    dps_time_map: Dict[str, np.ndarray],
    quiet: bool,
):
    all_files = list_rte_files(input_dirs, pattern)
    files = [p for p in all_files if T_RE.search(p.name) and LR_RE.search(p.name) and KIND_RE.search(p.name)]
    if not files:
        raise SystemExit("No RTE files found matching expected naming pattern.")

    if kind_filter != "both":
        files = [p for p in files if kind_from_name(p) == kind_filter]

    groups = defaultdict(list)
    warnings_list = []
    seen = {}  # (kind, lr, t_index) -> first_path

    for p in sorted(files):
        kind = kind_from_name(p)
        lr = lr_from_name(p)
        if lr_filter is not None and lr not in lr_filter:
            continue

        base = base_from_rte_name(p)
        if base not in dps_time_map:
            warnings_list.append(f"Warning: no matching DP-SCREAM file found for base stem '{base}', skipping {p}")
            continue

        t_index = t_from_name(p)
        times = dps_time_map[base]
        if t_index < 0 or t_index >= len(times):
            warnings_list.append(
                f"Warning: timestep index {t_index} out of bounds for DP-SCREAM stem '{base}' "
                f"(len={len(times)}), skipping {p}"
            )
            continue

        dup_key = (kind, lr, t_index)
        if dup_key in seen:
            warnings_list.append(
                f"Warning: duplicate timestep for kind={kind} lr={lr} t={t_index}; "
                f"keeping first={seen[dup_key]} and skipping duplicate={p}"
            )
            continue

        seen[dup_key] = p
        groups[(kind, lr)].append(p)

    if not groups:
        raise SystemExit("No valid groups found after matching against DP-SCREAM files.")

    if RANK == 0 and (not quiet):
        for w in warnings_list:
            log(w)

    return groups


def build_group_entries(files: List[Path], dps_time_map: Dict[str, np.ndarray]) -> List[dict]:
    entries = []
    for p in files:
        base = base_from_rte_name(p)
        t_index = t_from_name(p)
        time_hours = float(dps_time_map[base][t_index])
        entries.append(
            {
                "path": p,
                "t_index": t_index,
                "time_hours": time_hours,
                "base": base,
            }
        )

    entries.sort(key=lambda e: (e["time_hours"], e["t_index"], e["path"].name))
    for i, e in enumerate(entries):
        e["time_pos"] = i
    return entries


def preprocess_one_file(
    path: Path,
    *,
    engine: Optional[str],
    time_hours: float,
    t_index: int,
    keep_t_index: bool,
) -> xr.Dataset:
    ds = open_ds(path, engine=engine)

    ds = ds.drop_vars([v for v in GRID_SCALARS if v in ds.variables and ds[v].ndim == 0], errors="ignore")

    if "time" in ds.variables and "time" not in ds.dims:
        ds = ds.drop_vars("time")

    ds = ds.expand_dims(time=np.array([time_hours], dtype=np.float64))
    ds = ds.assign_coords(time=("time", np.array([time_hours], dtype=np.float64)))
    ds["time"].attrs["units"] = "hours since simulation start"
    ds["time"].attrs["long_name"] = "time"

    if keep_t_index:
        ds["t_index"] = xr.DataArray(np.array([t_index], dtype=np.int64), dims=("time",))
        ds["t_index"].attrs["long_name"] = "RTE timestep index corresponding to DP-SCREAM time index"

    ds = add_vertical_coords(ds)
    return ds


def build_template_from_first(
    first_path: Path,
    *,
    engine: Optional[str],
    entries: List[dict],
    keep_t_index: bool,
):
    with open_ds(first_path, engine=engine) as ds0:
        scalar_vars = {}
        for v in GRID_SCALARS:
            if v in ds0.variables and ds0[v].ndim == 0:
                scalar_vars[v] = ds0[v].load()

        if "time" in ds0.variables and "time" not in ds0.dims:
            ds0 = ds0.drop_vars("time")

        ds0 = ds0.drop_vars([v for v in GRID_SCALARS if v in ds0.variables and ds0[v].ndim == 0], errors="ignore")
        ds0 = ds0.expand_dims(time=np.array([entries[0]["time_hours"]], dtype=np.float64))
        ds0 = add_vertical_coords(ds0)
        ds0 = ds0.isel(time=slice(0, 1)).copy()

        time_vals = np.array([e["time_hours"] for e in entries], dtype=np.float64)
        ds0 = ds0.reindex(time=time_vals)
        ds0 = ds0.assign_coords(
            time=xr.DataArray(
                time_vals,
                dims=("time",),
                attrs={"units": "hours since simulation start", "long_name": "time"},
            )
        )

        if keep_t_index:
            ds0["t_index"] = xr.DataArray(
                np.array([e["t_index"] for e in entries], dtype=np.int64),
                dims=("time",),
                attrs={"long_name": "RTE timestep index corresponding to DP-SCREAM time index"},
            )

        for v, da in scalar_vars.items():
            ds0[v] = da

        return ds0, scalar_vars


def initialize_zarr_store(
    template_ds: xr.Dataset,
    zarr_path: Path,
    *,
    time_chunk: int,
    xy_chunk: Optional[int],
    replace_existing: bool,
    lock_writes: bool,
    quiet: bool,
    zarr_consolidated: bool,
):
    lock_dir = zarr_path.with_name(zarr_path.name + ".init_lock")
    lock_ctx = _mkdir_lock(lock_dir, quiet=quiet) if lock_writes else nullcontext()

    with lock_ctx:
        if zarr_path.exists():
            if not replace_existing:
                raise RuntimeError(f"Destination Zarr store exists: {zarr_path}")
            _safe_rmtree(zarr_path)

        tmp = _unique_tmp_for(zarr_path)
        if tmp.exists():
            _safe_rmtree(tmp)

        enc = build_zarr_encoding(template_ds, time_chunk=time_chunk, xy_chunk=xy_chunk)
        if not quiet:
            log(f"Initializing Zarr store template -> {tmp}")

        template_ds.to_zarr(tmp, mode="w", consolidated=zarr_consolidated, encoding=enc)

        if not quiet:
            log(f"Committing initialized Zarr store -> {zarr_path}")
        _atomic_commit_dir(tmp, zarr_path, replace_existing=replace_existing)


def write_time_slices_to_zarr(
    entries: List[dict],
    *,
    zarr_path: Path,
    engine: Optional[str],
    quiet: bool,
    keep_t_index: bool,
    zarr_consolidated: bool,
):
    my_entries = [e for i, e in enumerate(entries) if (i % NRANKS) == RANK]

    if not quiet:
        log(f"Assigned {len(my_entries)} time slice(s) for regional Zarr writes", every_rank=True)

    n_local = len(my_entries)
    for j, e in enumerate(my_entries, start=1):
        ds = preprocess_one_file(
            e["path"],
            engine=engine,
            time_hours=e["time_hours"],
            t_index=e["t_index"],
            keep_t_index=keep_t_index,
        )

        region = {"time": slice(e["time_pos"], e["time_pos"] + 1)}

        # For region writes, xarray requires every written variable to share
        # at least one dimension with the region dims, here just "time".
        keep_data_vars = [name for name, var in ds.data_vars.items() if "time" in var.dims]
        ds_region = ds[keep_data_vars].assign_coords(time=ds["time"])

        ds_region.to_zarr(
            zarr_path,
            mode="r+",
            region=region,
            consolidated=zarr_consolidated,
        )

        ds_region.close()
        ds.close()

        if (not quiet) and (j == 1 or j == n_local or j % 10 == 0):
            log(
                f"Wrote slice {j}/{n_local} for this rank -> "
                f"time_pos={e['time_pos']}, t_index={e['t_index']}, file={e['path'].name}",
                every_rank=True,
            )


def export_zarr_to_netcdf(
    zarr_path: Path,
    out_path: Path,
    *,
    engine: Optional[str],
    quiet: bool,
    zarr_consolidated: bool,
    replace_existing: bool,
):
    if out_path.exists():
        if not replace_existing:
            raise RuntimeError(f"Destination NetCDF exists: {out_path}")
        out_path.unlink(missing_ok=True)

    if not quiet:
        log(f"Opening Zarr for final export -> {zarr_path}")

    ds = xr.open_zarr(zarr_path, consolidated=zarr_consolidated)

    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].astype("float64"))
        ds["time"].attrs["units"] = "hours since simulation start"
        ds["time"].attrs["long_name"] = "time"

    encoding = {v: {"zlib": False} for v in ds.data_vars}
    encoding["time"] = {"dtype": "f8"}

    if not quiet:
        log(f"Writing final NetCDF -> {out_path}")

    t0 = time.time()
    ds.to_netcdf(out_path, engine=engine, encoding=encoding)
    elapsed = time.time() - t0
    ds.close()

    if not quiet:
        log(f"Wrote NetCDF: {out_path} ({elapsed:.1f} s, {elapsed/60:.2f} min)")


def combine_one_group(
    files: List[Path],
    *,
    out_path: Path,
    dps_time_map: Dict[str, np.ndarray],
    engine: Optional[str],
    quiet: bool,
    keep_t_index: bool,
    lock_writes: bool,
    replace_existing: bool,
    zarr_consolidated: bool,
    keep_zarr: bool,
    time_chunk: int,
    xy_chunk: Optional[int],
):
    if not files:
        return

    kind = kind_from_name(files[0])
    lr = lr_from_name(files[0])

    t_group0 = time.time()
    entries = build_group_entries(files, dps_time_map=dps_time_map)
    if not entries:
        if not quiet:
            log(f"No valid entries for kind={kind} lr={lr}; skipping")
        return

    first_path = entries[0]["path"]

    if not quiet:
        log(
            f"[{kind} lr_{lr}] Preparing group with {len(entries)} time slice(s), "
            f"time range=[{entries[0]['time_hours']:.6f}, {entries[-1]['time_hours']:.6f}] hours"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_path = out_path.with_suffix(out_path.suffix + ".zarr") if out_path.suffix else out_path.with_suffix(".zarr")

    if RANK == 0:
        t0 = time.time()
        template_ds, _scalar_vars = build_template_from_first(
            first_path,
            engine=engine,
            entries=entries,
            keep_t_index=keep_t_index,
        )
        initialize_zarr_store(
            template_ds,
            zarr_path,
            time_chunk=time_chunk,
            xy_chunk=xy_chunk,
            replace_existing=replace_existing,
            lock_writes=lock_writes,
            quiet=quiet,
            zarr_consolidated=zarr_consolidated,
        )
        template_ds.close()
        if not quiet:
            log(f"[{kind} lr_{lr}] Template/init time: {time.time() - t0:.1f} s")

    barrier()

    t1 = time.time()
    write_time_slices_to_zarr(
        entries,
        zarr_path=zarr_path,
        engine=engine,
        quiet=quiet,
        keep_t_index=keep_t_index,
        zarr_consolidated=zarr_consolidated,
    )
    barrier()
    if RANK == 0 and not quiet:
        log(f"[{kind} lr_{lr}] Parallel Zarr slice-write time: {time.time() - t1:.1f} s")

    if RANK == 0:
        t2 = time.time()
        export_zarr_to_netcdf(
            zarr_path,
            out_path,
            engine=engine,
            quiet=quiet,
            zarr_consolidated=zarr_consolidated,
            replace_existing=replace_existing,
        )
        if not keep_zarr:
            if not quiet:
                log(f"Removing intermediate Zarr store: {zarr_path}")
            _safe_rmtree(zarr_path)
        if not quiet:
            log(f"[{kind} lr_{lr}] NetCDF export time: {time.time() - t2:.1f} s")
            log(f"[{kind} lr_{lr}] Total group time: {time.time() - t_group0:.1f} s")

    barrier()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-dir", action="append", dest="input_dirs", type=Path, required=True,
                    help="Input directory containing separate RTE files. Repeat for multiple directories.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Output directory for combined files.")
    ap.add_argument("--dpscream-file", action="append", dest="dpscream_files", type=Path, required=True,
                    help="DP-SCREAM output file. Repeat for multiple files.")

    ap.add_argument("--lr", default=None, help="Comma-separated lr values, e.g. 02,04,08,16")
    ap.add_argument("--kind", choices=["in", "out", "both"], default="both")
    ap.add_argument("--pattern", default="*.nc")
    ap.add_argument("--engine", default=None)

    ap.add_argument("--flat-output", action="store_true")
    ap.add_argument("--replace-existing", action="store_true")
    ap.add_argument("--lock-writes", action="store_true")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--keep-zarr", action="store_true")
    ap.add_argument("--zarr-consolidated", action="store_true")
    ap.add_argument("--suppress-zarr-v3-warning", action="store_true")
    ap.add_argument("--keep-t-index", action="store_true")

    ap.add_argument("--time-chunk", type=int, default=1,
                    help="Chunk size along time in intermediate Zarr.")
    ap.add_argument("--xy-chunk", type=int, default=None,
                    help="Optional chunk size for x and y dims in intermediate Zarr.")

    ap.add_argument("--scheduler", choices=["threads", "processes", "single-threaded"], default=None)

    args = ap.parse_args()

    if args.lr is not None:
        args.lr = {s.strip() for s in args.lr.split(",") if s.strip()}

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.suppress_zarr_v3_warning:
        warnings.filterwarnings(
            "ignore",
            message=r"Consolidated metadata is currently not part in the Zarr format 3 specification.*",
            category=UserWarning,
        )

    if args.scheduler is not None:
        import dask  # type: ignore
        dask.config.set(scheduler=args.scheduler)

    if not args.quiet:
        log("Initializing MPI workflow.", every_rank=True)

    if RANK == 0:
        dps_time_map = load_dpscream_time_map(args.dpscream_files)
        if not args.quiet:
            log(f"Loaded {len(dps_time_map)} DP-SCREAM file(s)")
            for k, v in sorted(dps_time_map.items()):
                log(f"  DP-SCREAM stem: {k}  len(time)={len(v)}  range_hours=[{v[0]:.6f}, {v[-1]:.6f}]")
    else:
        dps_time_map = None

    if COMM is not None:
        dps_time_map = COMM.bcast(dps_time_map, root=0)

    if RANK == 0:
        groups = build_groups(
            input_dirs=args.input_dirs,
            pattern=args.pattern,
            kind_filter=args.kind,
            lr_filter=args.lr,
            dps_time_map=dps_time_map,
            quiet=args.quiet,
        )
        group_items = list(groups.items())
        group_items.sort(key=lambda kv: (kv[0][0], int(kv[0][1])))
        if not args.quiet:
            total_files = sum(len(v) for _, v in group_items)
            log(f"Found {total_files} valid RTE files in {len(group_items)} group(s)")
    else:
        group_items = None

    if COMM is not None:
        group_items = COMM.bcast(group_items, root=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for (kind, lr), group_files in group_items:
        example = sorted(group_files, key=lambda p: (t_from_name(p), p.name))[0]
        out_path = out_path_for_group(
            example,
            output_dir=args.output_dir,
            input_roots=args.input_dirs,
            flat_output=args.flat_output,
        )

        if not args.quiet:
            log(f"Starting group kind={kind} lr_{lr}: {len(group_files)} files -> {out_path}")

        combine_one_group(
            group_files,
            out_path=out_path,
            dps_time_map=dps_time_map,
            engine=args.engine,
            quiet=args.quiet,
            keep_t_index=args.keep_t_index,
            lock_writes=args.lock_writes,
            replace_existing=args.replace_existing,
            zarr_consolidated=args.zarr_consolidated,
            keep_zarr=args.keep_zarr,
            time_chunk=args.time_chunk,
            xy_chunk=args.xy_chunk,
        )

    barrier()
    if RANK == 0 and not args.quiet:
        log("Done.")


if __name__ == "__main__":
    main()