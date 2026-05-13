#!/usr/bin/env python3
"""
Combine per-timestep RTE-RRTMGP-CPP NetCDF files into time-stacked outputs
using DP-SCREAM time coordinates.

Key properties:
- MPI-enabled with mpi4py if available; serial fallback if not.
- Memory-safe for large datasets: does not build a full time-expanded xarray Dataset.
- Initializes Zarr from metadata only, then writes per-time slices directly via zarr array assignment.
- Avoids xarray region writes for slice assignment.
- Rank 0 handles discovery, grouping, metadata inference, Zarr initialization, and final NetCDF export.
- All ranks participate in distributed per-time-slice writes.

Requirements:
- Python 3
- xarray
- zarr
- numpy
- netCDF4 or another engine supported by xarray for NetCDF reads
- mpi4py optional
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import math
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import xarray as xr
import zarr


# ----------------------------
# MPI setup with serial fallback
# ----------------------------
try:
    from mpi4py import MPI  # type: ignore
    _COMM = MPI.COMM_WORLD
    RANK = _COMM.Get_rank()
    SIZE = _COMM.Get_size()
    HAVE_MPI = True
except Exception:
    MPI = None
    _COMM = None
    RANK = 0
    SIZE = 1
    HAVE_MPI = False


# ----------------------------
# Logging
# ----------------------------
HOSTNAME = socket.gethostname()


def log(msg: str, quiet: bool = False, rank_only: Optional[int] = None) -> None:
    if quiet:
        return
    if rank_only is not None and RANK != rank_only:
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] [rank {RANK}/{SIZE} @ {HOSTNAME}] {msg}", flush=True)


# ----------------------------
# Dataclasses
# ----------------------------
@dataclass
class ParsedRTEFile:
    path: str
    filename: str
    base: str
    t_index: int
    lr: int
    kind: str  # "in" or "out"


@dataclass
class GroupEntry:
    path: str
    t_index: int
    time_hours: float
    base: str
    time_pos: int
    filename: str


@dataclass
class GroupKey:
    kind: str
    lr: int

    def to_name(self) -> str:
        return f"{self.kind}.lr_{self.lr}"


@dataclass
class GroupInfo:
    key: GroupKey
    output_nc: str
    output_zarr: str
    entries: List[GroupEntry]


# ----------------------------
# Regex / filename parsing
# ----------------------------
RTE_RE = re.compile(
    r"^(?P<base>.+)\.t_(?P<t>\d+)\.lr_(?P<lr>\d+)\.(?P<kind>in|out)\.nc$"
)
TSTEP_RE = re.compile(r"\.t_(\d+)\.")
LR_RE = re.compile(r"\.lr_(\d+)\.")
KIND_RE = re.compile(r"\.(in|out)\.nc$")


def parse_rte_filename(path: Path) -> Optional[ParsedRTEFile]:
    m = RTE_RE.match(path.name)
    if not m:
        return None
    return ParsedRTEFile(
        path=str(path),
        filename=path.name,
        base=m.group("base"),
        t_index=int(m.group("t")),
        lr=int(m.group("lr")),
        kind=m.group("kind"),
    )


def remove_timestep_from_filename(filename: str) -> str:
    # Example:
    #   foo.t_12.lr_7.in.nc -> foo.lr_7.in.nc
    return re.sub(r"\.t_\d+(?=\.lr_\d+\.(?:in|out)\.nc$)", "", filename)


def stem_without_nc(path: Path) -> str:
    # filename without trailing .nc
    if path.name.endswith(".nc"):
        return path.name[:-3]
    return path.stem


# ----------------------------
# Filesystem helpers
# ----------------------------
def safe_rmtree(path: Path, quiet: bool = False) -> None:
    if path.exists():
        log(f"Removing directory tree: {path}", quiet=quiet)
        shutil.rmtree(path, ignore_errors=False)


def atomic_commit_dir(tmp_dir: Path, final_dir: Path, replace_existing: bool, quiet: bool = False) -> None:
    if final_dir.exists():
        if not replace_existing:
            raise FileExistsError(f"Target already exists: {final_dir}")
        safe_rmtree(final_dir, quiet=quiet)
    tmp_dir.rename(final_dir)
    log(f"Committed directory {tmp_dir} -> {final_dir}", quiet=quiet)


@contextlib.contextmanager
def mkdir_lock(lock_dir: Path, enabled: bool, poll_sec: float = 1.0, quiet: bool = False):
    if not enabled:
        yield
        return

    acquired = False
    try:
        while not acquired:
            try:
                lock_dir.mkdir(parents=False, exist_ok=False)
                acquired = True
                log(f"Acquired lock: {lock_dir}", quiet=quiet)
            except FileExistsError:
                log(f"Waiting on lock: {lock_dir}", quiet=quiet)
                time.sleep(poll_sec)
        yield
    finally:
        if acquired and lock_dir.exists():
            try:
                lock_dir.rmdir()
                log(f"Released lock: {lock_dir}", quiet=quiet)
            except Exception as e:
                log(f"Warning: failed to remove lock {lock_dir}: {e}", quiet=quiet)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine per-timestep RTE-RRTMGP-CPP NetCDF files into time-stacked outputs using DP-SCREAM time coordinates."
    )
    p.add_argument("--input-dir", action="append", required=True, help="Input directory root to search recursively. Repeatable.")
    p.add_argument("--dpscream-file", action="append", required=True, help="DP-SCREAM NetCDF file. Repeatable.")
    p.add_argument("--output-dir", required=True, help="Output directory.")
    p.add_argument("--lr", default=None, help="Comma-separated LR values to include, e.g. 1,2,4")
    p.add_argument("--kind", default="both", choices=["in", "out", "both"], help="Which kind(s) to process.")
    p.add_argument("--pattern", default="*.nc", help="Glob pattern for recursive discovery under input dirs.")
    p.add_argument("--engine", default=None, help="xarray engine for reading RTE files; DP-SCREAM uses netcdf4.")
    p.add_argument("--flat-output", action="store_true", help="Write outputs directly in output-dir instead of per-group subdirs.")
    p.add_argument("--replace-existing", action="store_true", help="Replace existing output NetCDF/Zarr.")
    p.add_argument("--lock-writes", action="store_true", help="Use mkdir-based lock around initialization/commit.")
    p.add_argument("--quiet", action="store_true", help="Reduce log output.")
    p.add_argument("--keep-zarr", action="store_true", help="Keep intermediate Zarr stores.")
    p.add_argument("--keep-t-index", action="store_true", help="Store integer t_index coordinate/data array.")
    p.add_argument("--time-chunk", type=int, default=16, help="Chunk size for time dimension.")
    p.add_argument("--xy-chunk", default=None, help="Chunk size for x and y dimensions, or 'None'.")
    p.add_argument(
        "--scheduler",
        default="single-threaded",
        choices=["threads", "processes", "single-threaded"],
        help="Dask scheduler for xarray operations if invoked."
    )
    return p.parse_args(argv)


# ----------------------------
# Utility parsing / scheduler
# ----------------------------
def parse_lr_filter(lr_str: Optional[str]) -> Optional[set]:
    if lr_str is None:
        return None
    vals = [s.strip() for s in lr_str.split(",") if s.strip()]
    if not vals:
        return None
    return {int(v) for v in vals}


def parse_xy_chunk(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s == "none":
        return None
    return int(s)


def barrier() -> None:
    if HAVE_MPI:
        _COMM.Barrier()


# ----------------------------
# DP-SCREAM time mapping
# ----------------------------
def load_dpscream_time_map(dpscream_files: Sequence[str], quiet: bool = False) -> Dict[str, np.ndarray]:
    """
    Returns map:
      dp_stem (filename without .nc) -> time_hours np.ndarray
    """
    time_map: Dict[str, np.ndarray] = {}
    for f in dpscream_files:
        p = Path(f)
        stem = stem_without_nc(p)
        if stem in time_map:
            log(f"Warning: duplicate DP-SCREAM stem '{stem}' encountered, keeping first: {p}", quiet=quiet, rank_only=0)
            continue
        log(f"Reading DP-SCREAM time from {p}", quiet=quiet, rank_only=0)
        with xr.open_dataset(p, engine="netcdf4", decode_times=False) as ds:
            if "time" not in ds.variables:
                raise KeyError(f"DP-SCREAM file missing variable 'time': {p}")
            time_vals = np.asarray(ds["time"].values)
            time_hours = time_vals.astype(np.float64) * 24.0
            if time_hours.ndim != 1:
                raise ValueError(f"DP-SCREAM time variable is not 1D in {p}")
            time_map[stem] = time_hours
    return time_map


# ----------------------------
# Discovery and grouping
# ----------------------------
def discover_rte_files(input_dirs: Sequence[str], pattern: str, quiet: bool = False) -> List[ParsedRTEFile]:
    found: List[ParsedRTEFile] = []
    for root in input_dirs:
        rootp = Path(root)
        if not rootp.exists():
            log(f"Warning: input dir does not exist: {rootp}", quiet=quiet, rank_only=0)
            continue
        for p in rootp.rglob(pattern):
            if not p.is_file():
                continue
            parsed = parse_rte_filename(p)
            if parsed is not None:
                found.append(parsed)
    return found


def select_kind(k: str, requested: str) -> bool:
    if requested == "both":
        return k in ("in", "out")
    return k == requested


def match_dp_base(base: str, dp_time_map: Dict[str, np.ndarray]) -> Optional[str]:
    """
    Match an RTE file base stem to a DP-SCREAM filename stem.

    Conservative strategy:
    - exact match first
    - otherwise suffix match either direction if unique
    """
    if base in dp_time_map:
        return base

    candidates = []
    for stem in dp_time_map.keys():
        if base.endswith(stem) or stem.endswith(base):
            candidates.append(stem)

    if len(candidates) == 1:
        return candidates[0]
    return None


def build_groups(
    parsed_files: Sequence[ParsedRTEFile],
    dp_time_map: Dict[str, np.ndarray],
    lr_filter: Optional[set],
    kind_filter: str,
    output_dir: Path,
    flat_output: bool,
    quiet: bool = False,
) -> List[GroupInfo]:
    by_group: Dict[Tuple[str, int], Dict[int, GroupEntry]] = {}

    for pf in parsed_files:
        if lr_filter is not None and pf.lr not in lr_filter:
            continue
        if not select_kind(pf.kind, kind_filter):
            continue

        matched_dp = match_dp_base(pf.base, dp_time_map)
        if matched_dp is None:
            log(f"Warning: could not match DP-SCREAM stem for {pf.filename}; skipping", quiet=quiet, rank_only=0)
            continue

        time_arr = dp_time_map[matched_dp]
        if pf.t_index < 0 or pf.t_index >= len(time_arr):
            log(f"Warning: timestep out of bounds for {pf.filename}: t_index={pf.t_index}, ntime={len(time_arr)}; skipping",
                quiet=quiet, rank_only=0)
            continue

        key = (pf.kind, pf.lr)
        if key not in by_group:
            by_group[key] = {}

        if pf.t_index in by_group[key]:
            prev = by_group[key][pf.t_index]
            log(f"Warning: duplicate timestep in group {key}, t_index={pf.t_index}; keeping first {prev.filename}, skipping {pf.filename}",
                quiet=quiet, rank_only=0)
            continue

        by_group[key][pf.t_index] = GroupEntry(
            path=pf.path,
            t_index=pf.t_index,
            time_hours=float(time_arr[pf.t_index]),
            base=pf.base,
            time_pos=-1,
            filename=pf.filename,
        )

    groups: List[GroupInfo] = []
    for (kind, lr), entries_by_t in sorted(by_group.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        entries = list(entries_by_t.values())
        entries.sort(key=lambda e: (e.time_hours, e.t_index, e.filename))
        for i, e in enumerate(entries):
            e.time_pos = i

        example_name = remove_timestep_from_filename(entries[0].filename)
        if flat_output:
            output_nc = str(output_dir / example_name)
            zarr_name = re.sub(r"\.nc$", ".zarr", example_name)
            output_zarr = str(output_dir / zarr_name)
        else:
            subdir = output_dir / f"{kind}.lr_{lr}"
            output_nc = str(subdir / example_name)
            zarr_name = re.sub(r"\.nc$", ".zarr", example_name)
            output_zarr = str(subdir / zarr_name)

        groups.append(
            GroupInfo(
                key=GroupKey(kind=kind, lr=lr),
                output_nc=output_nc,
                output_zarr=output_zarr,
                entries=entries,
            )
        )

    return groups


# ----------------------------
# Dataset normalization helpers
# ----------------------------
SCALAR_GRID_NAMES = {"ngrid_x", "ngrid_y", "ngrid_z"}


def maybe_drop_nondim_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" in ds.variables and "time" not in ds.dims:
        ds = ds.drop_vars("time")
    return ds


def extract_scalar_grid_vars(ds: xr.Dataset) -> Tuple[Dict[str, xr.DataArray], xr.Dataset]:
    scalars: Dict[str, xr.DataArray] = {}
    drop_names = []
    for name in SCALAR_GRID_NAMES:
        if name in ds.variables and ds[name].ndim == 0:
            scalars[name] = ds[name]
            drop_names.append(name)
    if drop_names:
        ds = ds.drop_vars(drop_names)
    return scalars, ds


def infer_lev_from_z(z_vals: np.ndarray) -> np.ndarray:
    """
    Compute staggered interfaces around center coordinate z.
    For z length n, return lev length n+1.
    """
    z = np.asarray(z_vals, dtype=np.float64)
    n = z.size
    if n < 1:
        raise ValueError("Cannot infer lev from empty z")
    if n == 1:
        dz = 1.0
        return np.array([z[0] - 0.5 * dz, z[0] + 0.5 * dz], dtype=np.float64)
    mid = 0.5 * (z[:-1] + z[1:])
    lev = np.empty(n + 1, dtype=np.float64)
    lev[1:-1] = mid
    lev[0] = z[0] - 0.5 * (z[1] - z[0])
    lev[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    return lev


def maybe_add_vertical_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    If coord z exists and dim lay has same length as z, and lev is absent, compute lev.
    """
    if "z" in ds.coords and "lay" in ds.dims:
        z = ds["z"]
        if z.ndim == 1 and z.sizes.get(z.dims[0], None) == ds.dims["lay"]:
            if "lev" not in ds.coords and "lev" not in ds.variables:
                lev_vals = infer_lev_from_z(np.asarray(z.values))
                ds = ds.assign_coords({"lev": ("lev", lev_vals)})
    return ds


def open_source_dataset(path: str, engine: Optional[str]) -> xr.Dataset:
    kwargs: Dict[str, Any] = {}
    if engine is not None:
        kwargs["engine"] = engine
    ds = xr.open_dataset(path, decode_times=False, **kwargs)
    ds = maybe_drop_nondim_time(ds)
    _, ds = extract_scalar_grid_vars(ds)
    ds = maybe_add_vertical_coords(ds)
    return ds


# ----------------------------
# Zarr metadata creation
# ----------------------------
def choose_chunks(shape: Tuple[int, ...], dims: Tuple[str, ...], time_chunk: int, xy_chunk: Optional[int]) -> Tuple[int, ...]:
    chunks: List[int] = []
    for dim, size in zip(dims, shape):
        if dim == "time":
            chunks.append(min(time_chunk, size))
        elif dim in ("x", "y"):
            if xy_chunk is None:
                chunks.append(size)
            else:
                chunks.append(min(xy_chunk, size))
        else:
            chunks.append(size)
    return tuple(chunks)


def attrs_to_jsonable(attrs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = str(v)
    return out


def init_group_zarr_metadata_only(
    sample_file: str,
    entries: List[GroupEntry],
    zarr_dir: Path,
    keep_t_index: bool,
    time_chunk: int,
    xy_chunk: Optional[int],
    engine: Optional[str],
    quiet: bool = False,
) -> None:
    """
    Initialize the target zarr store directly from metadata and static arrays,
    without constructing a full time-expanded xarray Dataset.
    """
    ntime = len(entries)
    if ntime < 1:
        raise ValueError("Cannot initialize empty group")

    sample_path = entries[0].path if sample_file is None else sample_file
    with open_source_dataset(sample_path, engine=engine) as ds:
        # Re-extract scalar grid vars from original open to preserve as scalars if present.
        ds_raw = xr.open_dataset(sample_path, decode_times=False, **({"engine": engine} if engine else {}))
        scalar_vars, _ = extract_scalar_grid_vars(maybe_drop_nondim_time(ds_raw))
        ds_raw.close()

        root = zarr.open_group(str(zarr_dir), mode="w")

        # Root attrs
        root.attrs.update(attrs_to_jsonable(ds.attrs))

        # Create dimensions from sample, but output time size is final ntime.
        source_dims = dict(ds.dims)

        # Time coordinate
        time_vals = np.array([e.time_hours for e in entries], dtype=np.float64)
        t_chunks = choose_chunks((ntime,), ("time",), time_chunk, xy_chunk)
        z_time = root.create_dataset(
            "time",
            shape=(ntime,),
            chunks=t_chunks,
            dtype="f8",
            overwrite=True,
        )
        z_time[:] = time_vals
        z_time.attrs.update(attrs_to_jsonable(ds["time"].attrs if "time" in ds.coords else {}))
        z_time.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        if "units" not in z_time.attrs:
            z_time.attrs["units"] = "hours"

        # Optional t_index
        if keep_t_index:
            t_index_vals = np.array([e.t_index for e in entries], dtype=np.int64)
            z_tidx = root.create_dataset(
                "t_index",
                shape=(ntime,),
                chunks=t_chunks,
                dtype="i8",
                overwrite=True,
            )
            z_tidx[:] = t_index_vals
            z_tidx.attrs["_ARRAY_DIMENSIONS"] = ["time"]
            z_tidx.attrs["long_name"] = "original timestep index"

        # Static coordinates
        # Preserve coordinate arrays other than time
        coord_names = set(ds.coords)
        for cname in sorted(coord_names):
            if cname == "time":
                continue
            cda = ds[cname]
            if cda.ndim == 0:
                zc = root.create_dataset(
                    cname,
                    shape=(),
                    dtype=np.asarray(cda.values).dtype,
                    overwrite=True,
                )
                zc[()] = np.asarray(cda.values)
                zc.attrs.update(attrs_to_jsonable(cda.attrs))
                zc.attrs["_ARRAY_DIMENSIONS"] = []
            else:
                shape = tuple(cda.shape)
                dims = tuple(cda.dims)
                chunks = choose_chunks(shape, dims, time_chunk, xy_chunk)
                zc = root.create_dataset(
                    cname,
                    shape=shape,
                    chunks=chunks,
                    dtype=np.asarray(cda.values).dtype,
                    overwrite=True,
                )
                zc[:] = np.asarray(cda.values)
                zc.attrs.update(attrs_to_jsonable(cda.attrs))
                zc.attrs["_ARRAY_DIMENSIONS"] = list(dims)

        # Scalar grid vars preserved as scalar arrays
        for sname, sda in scalar_vars.items():
            zs = root.create_dataset(
                sname,
                shape=(),
                dtype=np.asarray(sda.values).dtype,
                overwrite=True,
            )
            zs[()] = np.asarray(sda.values)
            zs.attrs.update(attrs_to_jsonable(sda.attrs))
            zs.attrs["_ARRAY_DIMENSIONS"] = []

        # Scalar data variables in sample
        for vname, vda in ds.data_vars.items():
            if vname in ("time",):
                continue
            if vname in coord_names:
                continue
            if vda.ndim == 0:
                zv = root.create_dataset(
                    vname,
                    shape=(),
                    dtype=np.asarray(vda.values).dtype,
                    overwrite=True,
                )
                zv[()] = np.asarray(vda.values)
                zv.attrs.update(attrs_to_jsonable(vda.attrs))
                zv.attrs["_ARRAY_DIMENSIONS"] = []
                continue

            src_dims = tuple(vda.dims)
            src_shape = tuple(vda.shape)

            # If source variable lacks time dim, make it time-dependent by prepending time
            if "time" in src_dims:
                out_dims = src_dims
                out_shape = list(src_shape)
                time_axis = src_dims.index("time")
                out_shape[time_axis] = ntime
            else:
                out_dims = ("time",) + src_dims
                out_shape = (ntime,) + src_shape

            chunks = choose_chunks(tuple(out_shape), tuple(out_dims), time_chunk, xy_chunk)
            fill_value = None
            encoding = getattr(vda, "encoding", {})
            if "_FillValue" in encoding:
                fill_value = encoding["_FillValue"]
            elif "_FillValue" in vda.attrs:
                fill_value = vda.attrs["_FillValue"]

            zv = root.create_dataset(
                vname,
                shape=tuple(out_shape),
                chunks=chunks,
                dtype=np.asarray(vda.values).dtype,
                overwrite=True,
                fill_value=fill_value,
            )
            zv.attrs.update(attrs_to_jsonable(vda.attrs))
            zv.attrs["_ARRAY_DIMENSIONS"] = list(out_dims)

        # Add a small manifest for debugging
        root.attrs["history"] = attrs_to_jsonable(root.attrs).get("history", "")
        root.attrs["combine_note"] = "Initialized metadata-only store; per-time slice writes performed directly with zarr assignment."


# ----------------------------
# Direct slice writing
# ----------------------------
def write_one_time_slice(
    zarr_dir: Path,
    entry: GroupEntry,
    keep_t_index: bool,
    engine: Optional[str],
    quiet: bool = False,
) -> None:
    """
    Open one source file and write its contents to the target zarr store at entry.time_pos.
    """
    root = zarr.open_group(str(zarr_dir), mode="r+")
    with xr.open_dataset(entry.path, decode_times=False, **({"engine": engine} if engine else {})) as ds_raw:
        ds = maybe_drop_nondim_time(ds_raw)
        _, ds = extract_scalar_grid_vars(ds)
        ds = maybe_add_vertical_coords(ds)

        # Validate/write static coordinates opportunistically only if missing data would matter.
        # Since metadata init already wrote coords and scalars, do not rewrite them here.

        for vname, vda in ds.data_vars.items():
            if vname == "time":
                continue
            arr = np.asarray(vda.values)

            if vda.ndim == 0:
                # scalar; leave as initialized from sample
                continue

            z = root[vname]
            z_dims = tuple(z.attrs["_ARRAY_DIMENSIONS"])

            if "time" in vda.dims:
                # Assume source file has a singleton time dimension if present.
                # Write one destination time position.
                src_time_axis = vda.dims.index("time")
                if arr.shape[src_time_axis] != 1:
                    raise ValueError(
                        f"Expected singleton time dimension in source for variable {vname} in file {entry.path}, "
                        f"got shape {arr.shape} with dims {vda.dims}"
                    )

                # Remove singleton source time axis and write into destination time_pos.
                arr_no_time = np.take(arr, indices=0, axis=src_time_axis)

                # Build destination indexing tuple
                dst_time_axis = z_dims.index("time")
                idx = [slice(None)] * z.ndim
                idx[dst_time_axis] = slice(entry.time_pos, entry.time_pos + 1)

                expanded = np.expand_dims(arr_no_time, axis=dst_time_axis)
                z[tuple(idx)] = expanded
            else:
                # Variable without time in source becomes time-dependent in output
                if z_dims[0] != "time":
                    raise ValueError(
                        f"Expected output variable {vname} to have prepended time dim, got dims {z_dims}"
                    )
                z[entry.time_pos:entry.time_pos + 1, ...] = np.expand_dims(arr, axis=0)

        if keep_t_index and "t_index" in root:
            root["t_index"][entry.time_pos] = entry.t_index
        if "time" in root:
            root["time"][entry.time_pos] = float(entry.time_hours)

    log(f"Wrote time_pos={entry.time_pos} t_index={entry.t_index} file={entry.filename}", quiet=quiet)


# ----------------------------
# Final export
# ----------------------------
def export_zarr_to_netcdf(zarr_dir: Path, output_nc: Path, replace_existing: bool, quiet: bool = False) -> None:
    if output_nc.exists():
        if not replace_existing:
            raise FileExistsError(f"Output NetCDF exists: {output_nc}")
        output_nc.unlink()

    log(f"Exporting Zarr -> NetCDF: {zarr_dir} -> {output_nc}", quiet=quiet, rank_only=0)
    with xr.open_zarr(str(zarr_dir), consolidated=False) as ds:
        encoding = {}
        for name in ds.data_vars:
            encoding[name] = {"zlib": False}
        if "time" in ds.variables:
            encoding["time"] = {"dtype": "f8"}
        ds.to_netcdf(str(output_nc), encoding=encoding)
    log(f"Finished NetCDF export: {output_nc}", quiet=quiet, rank_only=0)


# ----------------------------
# Main workflow
# ----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    xy_chunk = parse_xy_chunk(args.xy_chunk)

    # Dask scheduler preference if xarray triggers it
    try:
        import dask  # type: ignore
        dask.config.set(scheduler=args.scheduler)
    except Exception:
        pass

    outdir = Path(args.output_dir)
    if RANK == 0:
        ensure_dir(outdir)
    barrier()

    # Rank 0: discover and build plan
    if RANK == 0:
        t0_all = time.time()
        dp_time_map = load_dpscream_time_map(args.dpscream_file, quiet=args.quiet)
        log(f"Loaded {len(dp_time_map)} DP-SCREAM time maps", quiet=args.quiet, rank_only=0)
        if not dp_time_map:
            raise RuntimeError("No DP-SCREAM time maps loaded")

        rte_files = discover_rte_files(args.input_dir, args.pattern, quiet=args.quiet)
        log(f"Discovered {len(rte_files)} RTE candidate files", quiet=args.quiet, rank_only=0)

        groups = build_groups(
            parsed_files=rte_files,
            dp_time_map=dp_time_map,
            lr_filter=parse_lr_filter(args.lr),
            kind_filter=args.kind,
            output_dir=outdir,
            flat_output=args.flat_output,
            quiet=args.quiet,
        )

        log(f"Built {len(groups)} final groups", quiet=args.quiet, rank_only=0)
        for g in groups:
            log(f"Group {g.key.to_name()}: {len(g.entries)} files -> {g.output_nc}", quiet=args.quiet, rank_only=0)
    else:
        groups = None

    # Broadcast groups
    if HAVE_MPI:
        groups = _COMM.bcast(groups, root=0)

    assert groups is not None

    # Process each group
    for group_idx, group in enumerate(groups):
        group_start = time.time()
        zarr_dir = Path(group.output_zarr)
        output_nc = Path(group.output_nc)
        zarr_parent = zarr_dir.parent
        if RANK == 0:
            ensure_dir(zarr_parent)

        barrier()

        # Rank 0 initializes metadata-only zarr store
        if RANK == 0:
            log(f"Starting group {group_idx+1}/{len(groups)}: {group.key.to_name()} ({len(group.entries)} slices)",
                quiet=args.quiet, rank_only=0)

            if output_nc.exists() and not args.replace_existing:
                raise FileExistsError(f"Output NetCDF already exists: {output_nc}")
            if zarr_dir.exists() and not args.replace_existing:
                raise FileExistsError(f"Output Zarr already exists: {zarr_dir}")

            tmp_zarr = Path(str(zarr_dir) + ".init_tmp")
            lock_dir = Path(str(zarr_dir) + ".lock")

            with mkdir_lock(lock_dir, enabled=args.lock_writes, quiet=args.quiet):
                if tmp_zarr.exists():
                    safe_rmtree(tmp_zarr, quiet=args.quiet)
                if zarr_dir.exists() and args.replace_existing:
                    safe_rmtree(zarr_dir, quiet=args.quiet)

                init_group_zarr_metadata_only(
                    sample_file=group.entries[0].path,
                    entries=group.entries,
                    zarr_dir=tmp_zarr,
                    keep_t_index=args.keep_t_index,
                    time_chunk=args.time_chunk,
                    xy_chunk=xy_chunk,
                    engine=args.engine,
                    quiet=args.quiet,
                )
                atomic_commit_dir(tmp_zarr, zarr_dir, replace_existing=args.replace_existing, quiet=args.quiet)

        barrier()

        # All ranks write assigned time slices round-robin
        my_entries = [e for i, e in enumerate(group.entries) if (i % SIZE) == RANK]
        log(f"Assigned {len(my_entries)} slices for group {group.key.to_name()}", quiet=args.quiet)

        for j, entry in enumerate(my_entries, start=1):
            log(f"Writing slice {j}/{len(my_entries)} for group {group.key.to_name()}: {entry.filename}",
                quiet=args.quiet)
            write_one_time_slice(
                zarr_dir=zarr_dir,
                entry=entry,
                keep_t_index=args.keep_t_index,
                engine=args.engine,
                quiet=args.quiet,
            )

        barrier()

        # Rank 0 final export
        if RANK == 0:
            export_zarr_to_netcdf(
                zarr_dir=zarr_dir,
                output_nc=output_nc,
                replace_existing=args.replace_existing,
                quiet=args.quiet,
            )
            if not args.keep_zarr:
                safe_rmtree(zarr_dir, quiet=args.quiet)
            dt = time.time() - group_start
            log(f"Completed group {group.key.to_name()} in {dt:.1f} s", quiet=args.quiet, rank_only=0)

        barrier()

    if RANK == 0:
        log("All groups completed", quiet=args.quiet, rank_only=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())