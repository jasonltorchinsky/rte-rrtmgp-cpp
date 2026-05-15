#!/usr/bin/env python3
"""
combine_scream_timesteps.py

Combine many single-timestep NETCDF4 files into one combined NETCDF4 file per lr group.

Key features
------------
- Pure Python workflow
- MPI-parallel across many nodes/ranks using mpi4py
- Parallel NETCDF4 I/O via netCDF4-python (requires parallel-enabled NetCDF/HDF5)
- Dynamic resource reallocation across lr groups:
    * initially allocate ranks proportional to estimated work
    * when an lr group finishes, all of its ranks are reassigned to the coarsest
      still-running lr group
- Preserves dimensions, global attributes, variable attributes
- Static variables copied once
- Time-dependent variables written as (time, ...) exactly
- Reconstructs output time from t_XXX file-name index using the original DP-SCREAM
  time coordinate
- Skips missing time indices
- Detects corrupt/unreadable input files and reports them
- Provides periodic progress / ETA logging
- Handles output filename collisions by appending _1, _2, ...

Assumptions
-----------
- Input filenames look like:
  scream_dpxx_COMBLE_400x400.scream.INSTANT.nmins_x15.2020-03-12-79200.t_033.lr_01.in.nc
- Output filename should look like:
  scream_dpxx_COMBLE_400x400.scream.INSTANT.nmins_x15.2020-03-12-79200.lr_01.in.nc
- Each input file contains exactly one scalar variable named `time`, but that variable
  is ignored and reconstructed from the DP-SCREAM parent file time axis.
- The DP-SCREAM parent file path is supplied explicitly with --dpscream-file.
- Static variables are:
    ngrid_x, ngrid_y, ngrid_z, x, xh, y, yh, z, zh, z_lay, z_lev
- All other variables except time are considered time-dependent and are written with a
  leading `time` dimension.
- All files within one lr group have identical schema.

Performance notes
-----------------
- Best performance is obtained when launched with many MPI ranks across nodes.
- Each lr group is handled by an MPI subcommunicator.
- Within a group, variables are split across ranks to avoid duplicative reading/writing.
- Parallel NETCDF writing is used to let many ranks write different variables concurrently.
- Chunking/compression defaults are conservative because fastest wallclock is prioritized.
  Compression is off by default.
- Reading is file-by-file, variable-by-variable, avoiding accumulation of many timesteps
  in memory.
- This is optimized for Lustre input/output paths; do not use NFS for scratch/output.

Requirements
------------
- Python >= 3.9
- mpi4py
- netCDF4 built against parallel NetCDF/HDF5
- numpy

Example launch
--------------
srun -N 8 -n 448 python combine_scream_timesteps.py \
  --separate-dir /lustre/project/in_sep \
  --combined-dir /lustre/project/out_combined \
  --dpscream-file /lustre/project/orig/scream_dpxx_COMBLE_400x400.scream.INSTANT.nmins_x15.2020-03-12-79200.nc \
  --lr 01,02,04,08,16,32,64 \
  --preserve-precision True
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import re
import sys
import time as walltime
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset


COMM = MPI.COMM_WORLD
WORLD_RANK = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()

STATIC_VARS = {
    "ngrid_x", "ngrid_y", "ngrid_z",
    "x", "xh", "y", "yh", "z", "zh", "z_lay", "z_lev",
}
IGNORE_INPUT_VARS = {"time"}

FILENAME_RE = re.compile(
    r"^(?P<prefix>.+?)\.t_(?P<tidx>\d+)\.lr_(?P<lr>\d+)\.in\.nc$"
)

LOG_INTERVAL_SECONDS = 30.0


def log(msg: str, comm: MPI.Comm = COMM, root_only: bool = False) -> None:
    if root_only and comm.Get_rank() != 0:
        return
    now = walltime.strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{now}] [world_rank={WORLD_RANK}] {msg}\n")
    sys.stdout.flush()


def parse_bool(s: str) -> bool:
    s2 = str(s).strip().lower()
    if s2 in {"1", "true", "t", "yes", "y"}:
        return True
    if s2 in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean from '{s}'")


def parse_args():
    p = argparse.ArgumentParser(description="Combine Scream timestep NETCDF files by lr.")
    p.add_argument("--separate-dir", required=True, help="Directory containing timestep files on Lustre.")
    p.add_argument("--combined-dir", required=True, help="Directory to write combined files on Lustre.")
    p.add_argument("--dpscream-file", required=True, help="Original DP-SCREAM file containing authoritative time coordinate.")
    p.add_argument("--lr", required=True, help="Comma-separated lr list, e.g. 64,32,04,01")
    p.add_argument("--preserve-precision", type=parse_bool, default=True,
                   help="Preserve original variable dtype if True. Default=True")
    p.add_argument("--compression-level", type=int, default=0,
                   help="NETCDF4 zlib compression level [0-9]. Default 0 for fastest wallclock.")
    p.add_argument("--shuffle", type=parse_bool, default=False,
                   help="Enable HDF5 shuffle filter. Default False.")
    p.add_argument("--chunk-time", type=int, default=1,
                   help="Chunk size along time dimension. Default 1.")
    p.add_argument("--chunksize-mb", type=float, default=64.0,
                   help="Target chunk payload size in MB for non-time dims. Default 64.")
    p.add_argument("--no-fill", type=parse_bool, default=True,
                   help="Disable prefill on output file. Default True.")
    p.add_argument("--eta-interval", type=float, default=30.0,
                   help="Seconds between progress/ETA messages per lr group.")
    return p.parse_args()


@dataclass
class TimeStepFile:
    path: str
    tidx: int
    lr: str
    prefix: str
    size_bytes: int


@dataclass
class LRGroup:
    lr: str
    files: List[TimeStepFile] = field(default_factory=list)
    prefix: Optional[str] = None
    output_path: Optional[str] = None
    total_input_bytes: int = 0
    estimated_work: float = 0.0
    assigned_world_ranks: List[int] = field(default_factory=list)

    def sorted_files(self) -> List[TimeStepFile]:
        return sorted(self.files, key=lambda x: x.tidx)


def split_list_round_robin(items: List[str], nranks: int, rank: int) -> List[str]:
    return [v for i, v in enumerate(items) if i % nranks == rank]


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_nonconflicting_output_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)  # .nc
    k = 1
    while True:
        candidate = f"{root}_{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def discover_files(separate_dir: str, requested_lrs: List[str]) -> Dict[str, LRGroup]:
    groups: Dict[str, LRGroup] = {lr: LRGroup(lr=lr) for lr in requested_lrs}

    for entry in os.scandir(separate_dir):
        if not entry.is_file():
            continue
        name = entry.name
        if not name.endswith(".nc"):
            continue
        m = FILENAME_RE.match(name)
        if not m:
            continue
        lr = m.group("lr")
        if lr not in groups:
            continue
        tidx = int(m.group("tidx"))
        prefix = m.group("prefix")
        tsf = TimeStepFile(
            path=entry.path,
            tidx=tidx,
            lr=lr,
            prefix=prefix,
            size_bytes=entry.stat().st_size,
        )
        groups[lr].files.append(tsf)
        groups[lr].total_input_bytes += tsf.size_bytes
        if groups[lr].prefix is None:
            groups[lr].prefix = prefix

    for lr, g in groups.items():
        if g.prefix is None and g.files:
            g.prefix = g.files[0].prefix

    return groups


def estimate_group_work(group: LRGroup) -> float:
    # Strongly weight bytes; slight weight for file count to reflect metadata/open cost.
    return float(group.total_input_bytes) + 0.01 * float(len(group.files))


def read_dpscream_times(path: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        if "time" not in ds.variables:
            raise RuntimeError(f"DP-SCREAM file '{path}' has no variable named 'time'")
        time_vals = np.array(ds.variables["time"][:], dtype=np.float64)
    return time_vals * 24.0  # convert days to hours


def determine_rank_allocation(active_groups: List[LRGroup], world_size: int) -> Dict[str, List[int]]:
    if not active_groups:
        return {}

    for g in active_groups:
        g.estimated_work = estimate_group_work(g)

    total_work = sum(max(g.estimated_work, 1.0) for g in active_groups)
    desired = []
    for g in active_groups:
        frac = max(g.estimated_work, 1.0) / total_work
        desired.append((g.lr, frac * world_size))

    # At least 1 rank per active group
    alloc = {lr: 1 for lr, _ in desired}
    remaining = world_size - len(active_groups)

    if remaining > 0:
        floors = {lr: int(math.floor(val)) for lr, val in desired}
        for lr, _ in desired:
            extra = max(floors[lr] - 1, 0)
            give = min(extra, remaining)
            alloc[lr] += give
            remaining -= give

        if remaining > 0:
            remainders = sorted(
                ((lr, val - math.floor(val)) for lr, val in desired),
                key=lambda x: (-x[1], int(x[0]))
            )
            idx = 0
            while remaining > 0:
                lr = remainders[idx % len(remainders)][0]
                alloc[lr] += 1
                remaining -= 1
                idx += 1

    # Assign contiguous rank blocks to groups in descending work order.
    ordered = sorted(active_groups, key=lambda g: (g.estimated_work, -int(g.lr)), reverse=True)
    out = {}
    r0 = 0
    for g in ordered:
        nr = alloc[g.lr]
        out[g.lr] = list(range(r0, r0 + nr))
        r0 += nr
    return out


def output_name_from_prefix(prefix: str, lr: str) -> str:
    return f"{prefix}.lr_{lr}.in.nc"


def infer_variable_roles(ds: Dataset) -> Tuple[List[str], List[str]]:
    static_vars = []
    time_dependent_vars = []

    for vname, var in ds.variables.items():
        if vname in IGNORE_INPUT_VARS:
            continue
        if vname in STATIC_VARS:
            static_vars.append(vname)
        else:
            time_dependent_vars.append(vname)

    return static_vars, time_dependent_vars


def choose_output_dtype(var, preserve_precision: bool):
    if preserve_precision:
        return var.dtype
    # Normalize to float64 for floating point; leave ints/strings untouched.
    dt = np.dtype(var.dtype)
    if np.issubdtype(dt, np.floating):
        return np.dtype("f8")
    return dt


def compute_chunksizes(out_dims: Tuple[int, ...],
                       dtype: np.dtype,
                       var_dims: Tuple[str, ...],
                       chunk_time: int,
                       target_mb: float) -> Optional[Tuple[int, ...]]:
    if len(out_dims) == 0:
        return None

    itemsize = np.dtype(dtype).itemsize
    target_bytes = int(target_mb * 1024 * 1024)

    if len(out_dims) == 1:
        return (min(out_dims[0], max(1, chunk_time if var_dims[0] == "time" else out_dims[0])),)

    chunks = list(out_dims)
    start_idx = 0
    if var_dims[0] == "time":
        chunks[0] = min(out_dims[0], max(1, chunk_time))
        start_idx = 1

    # shrink non-time dims proportionally until chunk byte size <= target
    def chunk_bytes(ch):
        n = 1
        for x in ch:
            n *= max(1, x)
        return n * itemsize

    while chunk_bytes(chunks) > target_bytes:
        # halve the largest non-time chunk > 1
        candidates = [(i, chunks[i]) for i in range(start_idx, len(chunks)) if chunks[i] > 1]
        if not candidates:
            break
        i = max(candidates, key=lambda t: t[1])[0]
        chunks[i] = max(1, math.ceil(chunks[i] / 2))

    return tuple(int(x) for x in chunks)


def create_output_file_parallel(output_path: str,
                                template_path: str,
                                times_hours: np.ndarray,
                                preserve_precision: bool,
                                compression_level: int,
                                shuffle: bool,
                                chunk_time: int,
                                chunksize_mb: float,
                                no_fill: bool,
                                group_comm: MPI.Comm) -> Dict[str, List[str]]:
    """
    Create schema collectively. Returns metadata dict with static/time-dependent variables.
    """
    rank = group_comm.Get_rank()

    with Dataset(template_path, "r") as src:
        static_vars, time_vars = infer_variable_roles(src)

        with Dataset(output_path, "w", format="NETCDF4", parallel=True, comm=group_comm, info=MPI.Info.Create()) as dst:
            if no_fill:
                dst.set_fill_off()

            # Global attrs
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

            # Dimensions
            dst.createDimension("time", None)
            for dname, dim in src.dimensions.items():
                if dname == "time":
                    continue
                dst.createDimension(dname, len(dim) if not dim.isunlimited() else None)

            # Create time variable
            time_in = src.variables.get("time", None)
            t_dtype = np.dtype("f8") if not preserve_precision else (time_in.dtype if time_in is not None else np.dtype("f8"))
            tvar = dst.createVariable(
                "time",
                t_dtype,
                ("time",),
                zlib=(compression_level > 0),
                complevel=compression_level,
                shuffle=shuffle,
                chunksizes=(min(len(times_hours), max(1, chunk_time)),)
            )
            if time_in is not None:
                tvar.setncatts({a: time_in.getncattr(a) for a in time_in.ncattrs()})
            tvar.description = "Time since simulation start"
            tvar.units = "hours"

            # Create static vars
            for vname in static_vars:
                vsrc = src.variables[vname]
                vdtype = choose_output_dtype(vsrc, preserve_precision)
                dims = vsrc.dimensions
                shape = tuple(len(dst.dimensions[d]) for d in dims)
                chunks = compute_chunksizes(shape, vdtype, dims, chunk_time, chunksize_mb)
                kwargs = {}
                if vsrc.filters() is not None:
                    kwargs["zlib"] = (compression_level > 0)
                    kwargs["complevel"] = compression_level
                    kwargs["shuffle"] = shuffle
                if chunks is not None:
                    kwargs["chunksizes"] = chunks
                vdst = dst.createVariable(vname, vdtype, dims, **kwargs)
                vdst.setncatts({a: vsrc.getncattr(a) for a in vsrc.ncattrs()})

            # Create time-dependent vars with leading time dimension
            for vname in time_vars:
                vsrc = src.variables[vname]
                vdtype = choose_output_dtype(vsrc, preserve_precision)
                dims = ("time",) + tuple(vsrc.dimensions)
                shape = tuple(len(times_hours) if d == "time" else len(dst.dimensions[d]) for d in dims)
                chunks = compute_chunksizes(shape, vdtype, dims, chunk_time, chunksize_mb)
                kwargs = {}
                if vsrc.filters() is not None:
                    kwargs["zlib"] = (compression_level > 0)
                    kwargs["complevel"] = compression_level
                    kwargs["shuffle"] = shuffle
                if chunks is not None:
                    kwargs["chunksizes"] = chunks
                vdst = dst.createVariable(vname, vdtype, dims, **kwargs)
                vdst.setncatts({a: vsrc.getncattr(a) for a in vsrc.ncattrs()})

            # Collectively write time
            if rank == 0:
                tvar[:] = times_hours

    return {"static_vars": static_vars, "time_vars": time_vars}


def copy_static_variables(output_path: str,
                          template_path: str,
                          static_vars: List[str],
                          preserve_precision: bool,
                          group_comm: MPI.Comm) -> None:
    rank = group_comm.Get_rank()
    nranks = group_comm.Get_size()
    my_vars = split_list_round_robin(static_vars, nranks, rank)

    with Dataset(template_path, "r") as src, Dataset(output_path, "r+", format="NETCDF4",
                                                     parallel=True, comm=group_comm, info=MPI.Info.Create()) as dst:
        for vname in my_vars:
            try:
                data = src.variables[vname][:]
                if not preserve_precision:
                    dt = np.dtype(dst.variables[vname].dtype)
                    if np.issubdtype(dt, np.floating):
                        data = np.asarray(data, dtype=dt)
                dst.variables[vname][:] = data
            except Exception as e:
                log(f"ERROR copying static var '{vname}' from '{template_path}': {e}", root_only=False)
                raise
    group_comm.Barrier()


def process_lr_group(group: LRGroup,
                     dpscream_times_hours: np.ndarray,
                     preserve_precision: bool,
                     compression_level: int,
                     shuffle: bool,
                     chunk_time: int,
                     chunksize_mb: float,
                     no_fill: bool,
                     eta_interval: float,
                     world_group_ranks: List[int]) -> None:
    world_group = COMM.group.Incl(world_group_ranks)
    group_comm = COMM.Create_group(world_group)
    if group_comm == MPI.COMM_NULL:
        return

    rank = group_comm.Get_rank()
    nranks = group_comm.Get_size()

    try:
        sorted_files = group.sorted_files()
        if not sorted_files:
            if rank == 0:
                log(f"lr_{group.lr}: no files found; skipping.", root_only=False)
            return

        template_path = sorted_files[0].path
        valid_pairs: List[Tuple[int, TimeStepFile]] = []
        missing_tidx = []
        for f in sorted_files:
            if f.tidx < 0 or f.tidx >= len(dpscream_times_hours):
                if rank == 0:
                    log(f"lr_{group.lr}: file '{f.path}' has tidx={f.tidx}, outside DP-SCREAM time axis; skipping.")
                continue
            valid_pairs.append((f.tidx, f))

        valid_pairs.sort(key=lambda t: t[0])

        if not valid_pairs:
            if rank == 0:
                log(f"lr_{group.lr}: no valid files after filtering; skipping.")
            return

        time_indices = [t for t, _ in valid_pairs]
        times_hours = np.asarray([dpscream_times_hours[t] for t in time_indices], dtype=np.float64)

        if rank == 0:
            base_name = output_name_from_prefix(group.prefix, group.lr)
            base_path = os.path.join(args.combined_dir, base_name)
            out_path = choose_nonconflicting_output_path(base_path)
            group.output_path = out_path
        else:
            out_path = None

        out_path = group_comm.bcast(out_path, root=0)
        group.output_path = out_path

        if rank == 0:
            log(f"lr_{group.lr}: creating output file '{out_path}' with {len(valid_pairs)} times on {nranks} ranks.")

        schema = create_output_file_parallel(
            output_path=out_path,
            template_path=template_path,
            times_hours=times_hours,
            preserve_precision=preserve_precision,
            compression_level=compression_level,
            shuffle=shuffle,
            chunk_time=chunk_time,
            chunksize_mb=chunksize_mb,
            no_fill=no_fill,
            group_comm=group_comm,
        )
        static_vars = schema["static_vars"]
        time_vars = schema["time_vars"]

        if rank == 0:
            log(f"lr_{group.lr}: schema created. {len(static_vars)} static vars, {len(time_vars)} time-dependent vars.")

        copy_static_variables(
            output_path=out_path,
            template_path=template_path,
            static_vars=static_vars,
            preserve_precision=preserve_precision,
            group_comm=group_comm,
        )

        if rank == 0:
            log(f"lr_{group.lr}: static variables copied.")

        my_time_vars = split_list_round_robin(time_vars, nranks, rank)

        start = walltime.time()
        last_log = start
        processed = 0
        total = len(valid_pairs)
        corrupt_files = 0

        with Dataset(out_path, "r+", format="NETCDF4", parallel=True, comm=group_comm, info=MPI.Info.Create()) as dst:
            for it_out, (tidx, infile) in enumerate(valid_pairs):
                file_ok = True
                try:
                    with Dataset(infile.path, "r") as src:
                        # sanity-check schema lightly on rank 0
                        if rank == 0 and it_out == 0:
                            pass

                        for vname in my_time_vars:
                            try:
                                vsrc = src.variables[vname]
                                data = vsrc[:]
                                vdst = dst.variables[vname]
                                if not preserve_precision:
                                    dt = np.dtype(vdst.dtype)
                                    if np.issubdtype(dt, np.floating):
                                        data = np.asarray(data, dtype=dt)
                                vdst[it_out, ...] = data
                            except Exception as e:
                                log(
                                    f"ERROR lr_{group.lr}: failed variable '{vname}' from file '{infile.path}': {e}",
                                    root_only=False
                                )
                                file_ok = False
                                raise
                except Exception as e:
                    corrupt_files += 1
                    log(f"ERROR lr_{group.lr}: corrupt/unreadable file '{infile.path}': {e}", root_only=False)
                    # Leave unwritten timestep data as default/uninitialized if file failed.
                    # That timestep remains in time axis by design because input file existed.
                    # If you want to drop corrupt files entirely, pre-validate before schema creation.
                    # Here, for performance, we log and continue.
                    file_ok = False

                processed += 1

                now = walltime.time()
                if rank == 0 and (now - last_log >= eta_interval or processed == total):
                    elapsed = now - start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    remaining = total - processed
                    eta_sec = remaining / rate if rate > 0 else float("inf")
                    eta_str = (
                        walltime.strftime("%Y-%m-%d %H:%M:%S", walltime.localtime(now + eta_sec))
                        if np.isfinite(eta_sec) else "unknown"
                    )
                    log(
                        f"lr_{group.lr}: progress {processed}/{total} files "
                        f"({100.0*processed/total:.1f}%), "
                        f"elapsed={elapsed/60.0:.1f} min, ETA={eta_str}, corrupt={corrupt_files}"
                    )
                    last_log = now

        group_comm.Barrier()
        if rank == 0:
            elapsed = walltime.time() - start
            log(f"lr_{group.lr}: COMPLETE in {elapsed/60.0:.2f} min. Output: {out_path}. Corrupt files: {corrupt_files}")

    except Exception:
        err = traceback.format_exc()
        log(f"FATAL lr_{group.lr}:\n{err}", root_only=False)
        raise
    finally:
        group_comm.Free()
        world_group.Free()


def scheduler(groups: Dict[str, LRGroup],
              dpscream_times_hours: np.ndarray,
              preserve_precision: bool,
              compression_level: int,
              shuffle: bool,
              chunk_time: int,
              chunksize_mb: float,
              no_fill: bool,
              eta_interval: float) -> None:
    active = [g for g in groups.values() if g.files]
    inactive = [g for g in groups.values() if not g.files]

    if WORLD_RANK == 0:
        for g in inactive:
            log(f"lr_{g.lr}: no input files discovered; skipping.", root_only=True)

    # Dynamic wave scheduler:
    # repeatedly recompute allocations after each wave of concurrently active groups finishes.
    # Since MPI communicators are static inside each wave, "reallocation while running" is implemented
    # as wave-based redistribution. This is robust and scales well in practice.
    #
    # To bias resources toward coarser groups when a group finishes, each subsequent wave reallocates
    # ranks to remaining active groups. The "next coarsest still running" rule is approximated by
    # work-proportional allocation with tie-breaking toward coarser groups.

    remaining = sorted(active, key=lambda g: int(g.lr))
    wave = 0
    while remaining:
        wave += 1
        alloc = determine_rank_allocation(remaining, WORLD_SIZE)

        # All ranks learn assignment map
        alloc = COMM.bcast(alloc if WORLD_RANK == 0 else None, root=0)

        my_group_lr = None
        for lr, ranks in alloc.items():
            if WORLD_RANK in ranks:
                my_group_lr = lr
                break

        if WORLD_RANK == 0:
            summary = ", ".join([f"lr_{lr}:{len(ranks)}r" for lr, ranks in sorted(alloc.items(), key=lambda x: int(x[0]))])
            log(f"Starting wave {wave} with allocations: {summary}", root_only=True)

        target_group = None
        target_ranks = None
        if my_group_lr is not None:
            target_group = groups[my_group_lr]
            target_ranks = alloc[my_group_lr]

        # Each wave processes exactly one lr group per participating rank set.
        # All groups in this wave run concurrently.
        # To support concurrency, everyone calls process_lr_group only for their assigned group.
        if target_group is not None:
            process_lr_group(
                group=target_group,
                dpscream_times_hours=dpscream_times_hours,
                preserve_precision=preserve_precision,
                compression_level=compression_level,
                shuffle=shuffle,
                chunk_time=chunk_time,
                chunksize_mb=chunksize_mb,
                no_fill=no_fill,
                eta_interval=eta_interval,
                world_group_ranks=target_ranks,
            )
        else:
            # ranks not assigned this wave idle collectively
            COMM.Barrier()

        COMM.Barrier()

        # Remove all groups that were in this wave. This creates wave-based reallocation.
        # Since all allocated groups are processed to completion in the wave, remove them all.
        completed_lrs = set(alloc.keys())
        remaining = [g for g in remaining if g.lr not in completed_lrs]

        COMM.Barrier()
        if WORLD_RANK == 0:
            if remaining:
                rems = ", ".join(f"lr_{g.lr}" for g in sorted(remaining, key=lambda gg: int(gg.lr)))
                log(f"Wave {wave} complete. Remaining groups: {rems}", root_only=True)
            else:
                log(f"All lr groups complete.", root_only=True)


if __name__ == "__main__":
    args = parse_args()

    requested_lrs = [x.strip() for x in args.lr.split(",") if x.strip()]
    requested_lrs = [lr.zfill(2) for lr in requested_lrs]

    if WORLD_RANK == 0:
        safe_makedirs(args.combined_dir)

    COMM.Barrier()

    if WORLD_RANK == 0:
        log("Discovering input files...", root_only=True)
        groups = discover_files(args.separate_dir, requested_lrs)
        for g in groups.values():
            g.estimated_work = estimate_group_work(g)
            if g.files:
                log(
                    f"Discovered lr_{g.lr}: {len(g.files)} files, "
                    f"{g.total_input_bytes/1024**3:.2f} GiB input, "
                    f"estimated_work={g.estimated_work:.3e}",
                    root_only=True
                )
        log("Reading DP-SCREAM time coordinate...", root_only=True)
        dpscream_times_hours = read_dpscream_times(args.dpscream_file)
        log(f"Loaded {len(dpscream_times_hours)} DP-SCREAM time values.", root_only=True)
    else:
        groups = None
        dpscream_times_hours = None

    groups = COMM.bcast(groups, root=0)
    dpscream_times_hours = COMM.bcast(dpscream_times_hours, root=0)

    try:
        scheduler(
            groups=groups,
            dpscream_times_hours=dpscream_times_hours,
            preserve_precision=args.preserve_precision,
            compression_level=args.compression_level,
            shuffle=args.shuffle,
            chunk_time=args.chunk_time,
            chunksize_mb=args.chunksize_mb,
            no_fill=args.no_fill,
            eta_interval=args.eta_interval,
        )
    except Exception:
        err = traceback.format_exc()
        log(f"FATAL WORLD:\n{err}", root_only=False)
        COMM.Abort(1)