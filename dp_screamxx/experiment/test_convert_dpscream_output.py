#!/usr/bin/env python3
"""
Convert DP-SCREAM (EAMxx) output to RTE-RRTMGP-CPP input.

Implements:
- Multi-file discovery by directory + file root, concat/sort/de-dup time (keep later file).
- MPI time-sharding -> per-rank Zarr shards -> rank-0 NetCDF consolidation.
- Horizontal grid reconstruction from lon/lat (treated as x/y in meters) with tolerance clustering:
    xy_tol = 2 * np.finfo(np.float64).resolution
- Horizontal coarsening for lr>1 via conservative (aligned-grid) block averaging.
- Uniform-z vertical interpolation on common overlap [z_min, z_max] (no extrapolation allowed).
  Pressure interpolation uses interleaved (z_int,p_int) and (z_mid,p_mid).
- lwp/iwp recomputed from qc/qi and dp/g on output grid.

Caveat:
- Horizontal conservative remap is implemented as aligned-grid block-averaging (conservative for aligned grids).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None  # type: ignore

NP_REAL = np.float64
G_ACCEL: float = 9.80665


# ----------------------------
# MPI helpers
# ----------------------------
@dataclass(frozen=True)
class MpiInfo:
    comm: Any
    rank: int
    size: int


def get_mpi() -> MpiInfo:
    if MPI is None:
        return MpiInfo(comm=None, rank=0, size=1)
    comm: Any = MPI.COMM_WORLD
    return MpiInfo(comm=comm, rank=int(comm.Get_rank()), size=int(comm.Get_size()))


def bcast(mpi: MpiInfo, obj: Any, root: int = 0) -> Any:
    if mpi.size == 1:
        return obj
    return mpi.comm.bcast(obj, root=root)


def barrier(mpi: MpiInfo) -> None:
    if mpi.size > 1:
        mpi.comm.Barrier()


import time as _time
from datetime import datetime as _dt


def log(mpi: MpiInfo, msg: str, *, flush: bool = True) -> None:
    ts: str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    rank: int = mpi.rank
    print(f"[{ts}] [rank {rank:05d}] {msg}", flush=flush)

# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert DP-SCREAM output to RTE-RRTMGP-CPP input using xarray, MPI sharding, and conservative coarsening."
    )
    p.add_argument("--dp-screamxx-output-dir", required=True, type=str)
    p.add_argument("--dp-screamxx-output-file-root", required=True, type=str)
    p.add_argument("--rte-rrtmgp-cpp-input-dir", required=True, type=str)

    p.add_argument("--lr", required=True, nargs="+", type=int, help="Horizontal coarsening factors, e.g. --lr 1 2 4")
    p.add_argument("--nlay", default=None, type=int, help="Number of vertical layers (default: DP lev).")
    p.add_argument("--null-factor", default=10, type=int, help="ngrid scaling factor (default 10).")

    p.add_argument("--emis-sfc", default="1.0", type=str, help="Scalar or list (len band_lw).")
    p.add_argument("--sfc-alb-dir", default="0.0", type=str, help="Scalar or list (len band_sw).")
    p.add_argument("--sfc-alb-dif", default="0.0", type=str, help="Scalar or list (len band_sw).")
    p.add_argument("--tsi", default=1361.0, type=float)
    p.add_argument("--azi", default=0.0, type=float)

    p.add_argument("--tmp-dir", default=None, type=str, help="Temporary dir for Zarr shards.")
    p.add_argument("--chunks-time", default=1, type=int, help="open_mfdataset chunk along time (default 1).")
    p.add_argument("--compress-level", default=1, type=int, help="NetCDF zlib compression level.")
    return p.parse_args(argv)


# ----------------------------
# File discovery / time handling
# ----------------------------
_TIME_FROM_NAME_RE = re.compile(r"\.(\d{4}-\d{2}-\d{2}-\d{5})\.nc$")


def discover_input_files(dp_dir: str, file_root: str) -> List[str]:
    pattern: str = os.path.join(dp_dir, f"{file_root}.*.nc")
    files: List[str] = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No DP-SCREAM files found matching: {pattern}")
    return files


def parse_startdate_from_any_filename(files: Sequence[str]) -> str:
    m = _TIME_FROM_NAME_RE.search(os.path.basename(files[0]))
    if m is None:
        raise ValueError(f"Could not parse startdate token from filename: {files[0]}")
    return m.group(1)


def open_dp_scream_mfdataset(files: Sequence[str], chunks_time: int) -> xr.Dataset:
    ds: xr.Dataset = xr.open_mfdataset(
        list(files),
        combine="nested",
        concat_dim="time",
        parallel=True,
        chunks={"time": chunks_time},
        engine="netcdf4",
        decode_times=True,
    )
    return ds


def dedup_sort_time_keep_last(ds: xr.Dataset) -> xr.Dataset:
    """
    Sort by time and drop duplicates keeping last occurrence (later file wins due to nested concat order).
    """
    tvals: np.ndarray = ds["time"].values
    order: np.ndarray = np.argsort(tvals, kind="stable")
    ds_sorted: xr.Dataset = ds.isel(time=xr.DataArray(order, dims=("time",)))

    t_sorted: np.ndarray = ds_sorted["time"].values
    rev_idx: np.ndarray = np.arange(t_sorted.size - 1, -1, -1, dtype=np.int64)
    t_rev: np.ndarray = t_sorted[rev_idx]

    _uniq: np.ndarray
    first_in_rev: np.ndarray
    _uniq, first_in_rev = np.unique(t_rev, return_index=True)

    keep_rev: np.ndarray = rev_idx[first_in_rev]
    keep_sorted: np.ndarray = np.sort(keep_rev)
    return ds_sorted.isel(time=xr.DataArray(keep_sorted, dims=("time",)))


# ----------------------------
# Horizontal mapping
# ----------------------------
def _xy_tol() -> float:
    return float(2.0 * np.finfo(np.float64).resolution)


def cluster_1d(values: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster 1D values by abs tolerance in sorted order.
    Returns (centers_sorted, inv) where inv maps original elements to cluster index.
    """
    v: np.ndarray = np.asarray(values, dtype=np.float64)
    idx_sort: np.ndarray = np.argsort(v, kind="stable")
    v_sorted: np.ndarray = v[idx_sort]

    bounds: List[Tuple[int, int]] = []
    start: int = 0
    for i in range(1, v_sorted.size):
        if abs(v_sorted[i] - v_sorted[i - 1]) > tol:
            bounds.append((start, i))
            start = i
    bounds.append((start, v_sorted.size))

    centers_sorted: np.ndarray = np.array([float(v_sorted[a:b].mean()) for (a, b) in bounds], dtype=np.float64)

    inv_sorted: np.ndarray = np.empty(v_sorted.size, dtype=np.int64)
    for cid, (a, b) in enumerate(bounds):
        inv_sorted[a:b] = cid

    inv: np.ndarray = np.empty_like(inv_sorted)
    inv[idx_sort] = inv_sorted
    return centers_sorted, inv


def infer_edges_from_centers(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    if c.size < 2:
        raise ValueError("Need at least 2 centers to infer edges.")
    dc: np.ndarray = np.diff(c)
    if not np.all(dc > 0):
        raise ValueError("Centers must be strictly increasing.")
    edges: np.ndarray = np.empty(c.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (c[1:] + c[:-1])
    edges[0] = c[0] - 0.5 * dc[0]
    edges[-1] = c[-1] + 0.5 * dc[-1]
    return edges


@dataclass(frozen=True)
class HorizontalMap:
    x: np.ndarray
    y: np.ndarray
    xh: np.ndarray
    yh: np.ndarray
    nx: int
    ny: int
    ncol_to_yx: np.ndarray  # (ncol,2) -> iy,ix
    rep_ncol: np.ndarray  # (ny,nx) representative ncol index


def build_horizontal_map(ds: xr.Dataset) -> HorizontalMap:
    x_raw: np.ndarray = ds["lon"].values.astype(np.float64)  # treated as x [m]
    y_raw: np.ndarray = ds["lat"].values.astype(np.float64)  # treated as y [m]
    tol: float = _xy_tol()

    x_centers: np.ndarray
    ix: np.ndarray
    x_centers, ix = cluster_1d(x_raw, tol=tol)

    y_centers: np.ndarray
    iy: np.ndarray
    y_centers, iy = cluster_1d(y_raw, tol=tol)

    nx: int = int(x_centers.size)
    ny: int = int(y_centers.size)

    rep_ncol: np.ndarray = -np.ones((ny, nx), dtype=np.int64)
    for n in range(x_raw.size):
        rep_ncol[iy[n], ix[n]] = n  # overwrite => last wins

    if np.any(rep_ncol < 0):
        missing: int = int(np.sum(rep_ncol < 0))
        raise ValueError(f"Reconstructed grid incomplete: missing {missing} of {ny*nx} cells.")

    ncol_to_yx: np.ndarray = np.stack([iy, ix], axis=1).astype(np.int64)
    xh: np.ndarray = infer_edges_from_centers(x_centers)
    yh: np.ndarray = infer_edges_from_centers(y_centers)

    return HorizontalMap(
        x=x_centers,
        y=y_centers,
        xh=xh,
        yh=yh,
        nx=nx,
        ny=ny,
        ncol_to_yx=ncol_to_yx,
        rep_ncol=rep_ncol,
    )


# ----------------------------
# Coarse grid + conservative coarsening (aligned)
# ----------------------------
@dataclass(frozen=True)
class CoarseGrid:
    x: np.ndarray
    y: np.ndarray
    xh: np.ndarray
    yh: np.ndarray
    nx: int
    ny: int
    lr: int


def build_coarse_grid(hmap: HorizontalMap, lr: int) -> CoarseGrid:
    if lr < 1:
        raise ValueError("lr must be >= 1")
    nx_c: int = hmap.nx // lr
    ny_c: int = hmap.ny // lr
    if nx_c < 1 or ny_c < 1:
        raise ValueError(f"lr={lr} too large for grid nx={hmap.nx}, ny={hmap.ny}")

    nx_use: int = nx_c * lr
    ny_use: int = ny_c * lr

    xh_f: np.ndarray = hmap.xh[: nx_use + 1]
    yh_f: np.ndarray = hmap.yh[: ny_use + 1]

    xh_c: np.ndarray = xh_f[::lr].copy()
    yh_c: np.ndarray = yh_f[::lr].copy()
    x_c: np.ndarray = 0.5 * (xh_c[:-1] + xh_c[1:])
    y_c: np.ndarray = 0.5 * (yh_c[:-1] + yh_c[1:])

    return CoarseGrid(x=x_c, y=y_c, xh=xh_c, yh=yh_c, nx=nx_c, ny=ny_c, lr=lr)


def conservative_block_average(arr: np.ndarray, lr: int) -> np.ndarray:
    """
    Block-average last two dims (y,x). Conservative for aligned grid and cell-mean fields.
    """
    if lr == 1:
        return arr
    shape: Tuple[int, ...] = arr.shape
    ny: int = shape[-2]
    nx: int = shape[-1]
    ny_c: int = ny // lr
    nx_c: int = nx // lr
    arr2: np.ndarray = arr[..., : ny_c * lr, : nx_c * lr]
    arr_blk: np.ndarray = arr2.reshape(*shape[:-2], ny_c, lr, nx_c, lr)
    return arr_blk.mean(axis=(-1, -3))


# ----------------------------
# Vertical grid / interpolation
# ----------------------------
@dataclass(frozen=True)
class VerticalGrid:
    z_lev: np.ndarray
    z_lay: np.ndarray
    nlay: int


def compute_common_vertical_domain(ds: xr.Dataset) -> Tuple[float, float]:
    zint: xr.DataArray = ds["z_int"]
    z0: xr.DataArray = zint.isel(ilev=0)
    ztop: xr.DataArray = zint.isel(ilev=-1)

    z_min: float = float(z0.max(skipna=True).compute().values)
    z_max: float = float(ztop.min(skipna=True).compute().values)
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        raise ValueError("Failed to compute finite z_min/z_max from z_int.")
    if z_max <= z_min:
        raise ValueError(f"Invalid common vertical domain: z_max={z_max} <= z_min={z_min}.")
    return z_min, z_max


def build_uniform_vertical_grid(z_min: float, z_max: float, nlay: int) -> VerticalGrid:
    z_lev: np.ndarray = np.linspace(z_min, z_max, nlay + 1, dtype=np.float64)
    z_lay: np.ndarray = 0.5 * (z_lev[:-1] + z_lev[1:])
    return VerticalGrid(z_lev=z_lev, z_lay=z_lay, nlay=nlay)


def _check_no_extrapolation_possible(x_src_min: float, x_src_max: float, x_tgt_min: float, x_tgt_max: float, what: str) -> None:
    if x_tgt_min < x_src_min or x_tgt_max > x_src_max:
        raise ValueError(
            f"Vertical grid would require extrapolation for {what}: target [{x_tgt_min}, {x_tgt_max}] "
            f"not within source [{x_src_min}, {x_src_max}]."
        )


def _interp_1d_no_extrap(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    return np.interp(x_tgt, x_src, y_src, left=np.nan, right=np.nan)


def interp_profile_interleaved_pressure(
    z_int: np.ndarray,
    p_int: np.ndarray,
    z_mid: np.ndarray,
    p_mid: np.ndarray,
    z_lev_tgt: np.ndarray,
    z_lay_tgt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    z_src: np.ndarray = np.concatenate([z_int, z_mid]).astype(np.float64)
    p_src: np.ndarray = np.concatenate([p_int, p_mid]).astype(np.float64)

    order: np.ndarray = np.argsort(z_src, kind="stable")
    z_s: np.ndarray = z_src[order]
    p_s: np.ndarray = p_src[order]

    rev: np.ndarray = np.arange(z_s.size - 1, -1, -1, dtype=np.int64)
    z_rev: np.ndarray = z_s[rev]
    _zuniq: np.ndarray
    first: np.ndarray
    _zuniq, first = np.unique(z_rev, return_index=True)
    keep: np.ndarray = np.sort(rev[first])
    z_s = z_s[keep]
    p_s = p_s[keep]

    _check_no_extrapolation_possible(float(z_s.min()), float(z_s.max()), float(z_lev_tgt.min()), float(z_lev_tgt.max()), "pressure")
    p_lev: np.ndarray = _interp_1d_no_extrap(z_s, p_s, z_lev_tgt)
    p_lay: np.ndarray = _interp_1d_no_extrap(z_s, p_s, z_lay_tgt)
    return p_lev, p_lay


def interp_profile_mid_only(z_mid: np.ndarray, v_mid: np.ndarray, z_tgt: np.ndarray, what: str) -> np.ndarray:
    order: np.ndarray = np.argsort(z_mid, kind="stable")
    z_s: np.ndarray = z_mid[order].astype(np.float64)
    v_s: np.ndarray = v_mid[order].astype(np.float64)

    rev: np.ndarray = np.arange(z_s.size - 1, -1, -1, dtype=np.int64)
    z_rev: np.ndarray = z_s[rev]
    _zuniq: np.ndarray
    first: np.ndarray
    _zuniq, first = np.unique(z_rev, return_index=True)
    keep: np.ndarray = np.sort(rev[first])
    z_s = z_s[keep]
    v_s = v_s[keep]

    _check_no_extrapolation_possible(float(z_s.min()), float(z_s.max()), float(z_tgt.min()), float(z_tgt.max()), what)
    return _interp_1d_no_extrap(z_s, v_s, z_tgt)


# ----------------------------
# Overrides parsing
# ----------------------------
def parse_scalar_or_list(s: str, n: int) -> np.ndarray:
    parts: List[str] = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 1:
        return np.full((n,), float(parts[0]), dtype=np.float64)
    if len(parts) != n:
        raise ValueError(f"Expected scalar or list of length {n}, got {len(parts)} from {s}")
    return np.array([float(p) for p in parts], dtype=np.float64)


# ----------------------------
# DP variable names
# ----------------------------
@dataclass(frozen=True)
class DpVars:
    z_mid: str = "z_mid"
    z_int: str = "z_int"
    p_mid: str = "p_mid"
    p_int: str = "p_int"
    T_mid: str = "T_mid"
    RH_mid: str = "RelativeHumidity"
    qc: str = "qc"
    qi: str = "qi"
    rel: str = "eff_radius_qc"
    dei_radius: str = "eff_radius_qi"
    mu0: str = "cosine_solar_zenith_angle"
    t_sfc: str = "surf_radiative_T"
    vmr_h2o: str = "h2o_volume_mix_ratio"
    vmr_co2: str = "co2_volume_mix_ratio"
    vmr_o3: str = "o3_volume_mix_ratio"
    vmr_n2o: str = "n2o_volume_mix_ratio"
    vmr_co: str = "co_volume_mix_ratio"
    vmr_ch4: str = "ch4_volume_mix_ratio"
    vmr_o2: str = "o2_volume_mix_ratio"
    vmr_n2: str = "n2_volume_mix_ratio"


def reshape_ncol_to_yx_2d(v: np.ndarray, rep_ncol: np.ndarray) -> np.ndarray:
    return v[rep_ncol]


def reshape_ncol_to_yx_3d(v: np.ndarray, rep_ncol: np.ndarray) -> np.ndarray:
    ny, nx = rep_ncol.shape
    nz: int = v.shape[1]
    flat_idx: np.ndarray = rep_ncol.reshape(ny * nx)
    tmp: np.ndarray = v[flat_idx, :]  # (ny*nx,nz)
    return tmp.reshape(ny, nx, nz).transpose(2, 0, 1)  # (nz,ny,nx)


def convert_time_indices_to_rte_shard(
    ds: xr.Dataset,
    hmap: HorizontalMap,
    vgrid: VerticalGrid,
    time_indices: np.ndarray,
    coarse: CoarseGrid,
    band_lw: int,
    band_sw: int,
    overrides: Mapping[str, Any],
) -> xr.Dataset:
    dv: DpVars = DpVars()
    nt: int = int(time_indices.size)
    ny: int = int(coarse.ny)
    nx: int = int(coarse.nx)
    nlay: int = int(vgrid.nlay)
    nlev: int = nlay + 1

    time_hours: np.ndarray = overrides["time_hours"][time_indices]

    emis_band: np.ndarray = overrides["emis_band"]
    alb_dir_band: np.ndarray = overrides["alb_dir_band"]
    alb_dif_band: np.ndarray = overrides["alb_dif_band"]

    tsi_val: float = float(overrides["tsi"])
    azi_val: float = float(overrides["azi"])

    emis_sfc: np.ndarray = np.broadcast_to(emis_band.reshape(1, 1, 1, band_lw), (nt, ny, nx, band_lw)).copy()
    sfc_alb_dir: np.ndarray = np.broadcast_to(alb_dir_band.reshape(1, 1, 1, band_sw), (nt, ny, nx, band_sw)).copy()
    sfc_alb_dif: np.ndarray = np.broadcast_to(alb_dif_band.reshape(1, 1, 1, band_sw), (nt, ny, nx, band_sw)).copy()
    tsi: np.ndarray = np.full((nt, ny, nx), tsi_val, dtype=np.float64)
    azi: np.ndarray = np.full((nt, ny, nx), azi_val, dtype=np.float64)

    p_lay: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    p_lev: np.ndarray = np.empty((nt, nlev, ny, nx), dtype=np.float64)
    t_lay: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    t_lev: np.ndarray = np.empty((nt, nlev, ny, nx), dtype=np.float64)
    rh: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)

    rel: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    dei: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)

    vmr_h2o: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_co2: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_o3: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_n2o: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_co: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_ch4: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_o2: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    vmr_n2: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)

    t_sfc: np.ndarray = np.empty((nt, ny, nx), dtype=np.float64)
    mu0: np.ndarray = np.empty((nt, ny, nx), dtype=np.float64)

    qc_lay: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    qi_lay: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    lwp: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)
    iwp: np.ndarray = np.empty((nt, nlay, ny, nx), dtype=np.float64)

    rep: np.ndarray = hmap.rep_ncol
    lr: int = coarse.lr

    for it_local, tt in enumerate(time_indices.tolist()):
        z_mid_t: np.ndarray = ds[dv.z_mid].isel(time=tt).values.astype(np.float64)
        z_int_t: np.ndarray = ds[dv.z_int].isel(time=tt).values.astype(np.float64)
        p_mid_t: np.ndarray = ds[dv.p_mid].isel(time=tt).values.astype(np.float64)
        p_int_t: np.ndarray = ds[dv.p_int].isel(time=tt).values.astype(np.float64)

        T_mid_t: np.ndarray = ds[dv.T_mid].isel(time=tt).values.astype(np.float64)
        RH_t: np.ndarray = ds[dv.RH_mid].isel(time=tt).values.astype(np.float64)

        qc_t: np.ndarray = ds[dv.qc].isel(time=tt).values.astype(np.float64)
        qi_t: np.ndarray = ds[dv.qi].isel(time=tt).values.astype(np.float64)

        rel_t: np.ndarray = ds[dv.rel].isel(time=tt).values.astype(np.float64)
        dei_r_t: np.ndarray = ds[dv.dei_radius].isel(time=tt).values.astype(np.float64)

        mu0_t: np.ndarray = ds[dv.mu0].isel(time=tt).values.astype(np.float64)
        tsfc_t: np.ndarray = ds[dv.t_sfc].isel(time=tt).values.astype(np.float64)

        vmr_h2o_t: np.ndarray = ds[dv.vmr_h2o].isel(time=tt).values.astype(np.float64)
        vmr_co2_t: np.ndarray = ds[dv.vmr_co2].isel(time=tt).values.astype(np.float64)
        vmr_o3_t: np.ndarray = ds[dv.vmr_o3].isel(time=tt).values.astype(np.float64)
        vmr_n2o_t: np.ndarray = ds[dv.vmr_n2o].isel(time=tt).values.astype(np.float64)
        vmr_co_t: np.ndarray = ds[dv.vmr_co].isel(time=tt).values.astype(np.float64)
        vmr_ch4_t: np.ndarray = ds[dv.vmr_ch4].isel(time=tt).values.astype(np.float64)
        vmr_o2_t: np.ndarray = ds[dv.vmr_o2].isel(time=tt).values.astype(np.float64)
        vmr_n2_t: np.ndarray = ds[dv.vmr_n2].isel(time=tt).values.astype(np.float64)

        z_mid_zyx: np.ndarray = reshape_ncol_to_yx_3d(z_mid_t, rep)
        z_int_zyx: np.ndarray = reshape_ncol_to_yx_3d(z_int_t, rep)
        p_mid_zyx: np.ndarray = reshape_ncol_to_yx_3d(p_mid_t, rep)
        p_int_zyx: np.ndarray = reshape_ncol_to_yx_3d(p_int_t, rep)

        T_mid_zyx: np.ndarray = reshape_ncol_to_yx_3d(T_mid_t, rep)
        RH_zyx: np.ndarray = reshape_ncol_to_yx_3d(RH_t, rep)
        qc_zyx: np.ndarray = reshape_ncol_to_yx_3d(qc_t, rep)
        qi_zyx: np.ndarray = reshape_ncol_to_yx_3d(qi_t, rep)
        rel_zyx: np.ndarray = reshape_ncol_to_yx_3d(rel_t, rep)
        dei_r_zyx: np.ndarray = reshape_ncol_to_yx_3d(dei_r_t, rep)

        vmr_h2o_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_h2o_t, rep)
        vmr_co2_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_co2_t, rep)
        vmr_o3_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_o3_t, rep)
        vmr_n2o_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_n2o_t, rep)
        vmr_co_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_co_t, rep)
        vmr_ch4_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_ch4_t, rep)
        vmr_o2_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_o2_t, rep)
        vmr_n2_zyx: np.ndarray = reshape_ncol_to_yx_3d(vmr_n2_t, rep)

        tsfc_yx: np.ndarray = reshape_ncol_to_yx_2d(tsfc_t, rep)
        mu0_yx: np.ndarray = reshape_ncol_to_yx_2d(mu0_t, rep)

        t_sfc[it_local, :, :] = conservative_block_average(tsfc_yx[None, ...], lr=lr)[0]
        mu0[it_local, :, :] = conservative_block_average(mu0_yx[None, ...], lr=lr)[0]

        # Coarsen profiles first (conservative)
        def coarsen(field_zyx: np.ndarray) -> np.ndarray:
            return conservative_block_average(field_zyx[None, ...], lr=lr)[0]

        z_mid_c: np.ndarray = coarsen(z_mid_zyx)
        z_int_c: np.ndarray = coarsen(z_int_zyx)
        p_mid_c: np.ndarray = coarsen(p_mid_zyx)
        p_int_c: np.ndarray = coarsen(p_int_zyx)

        T_mid_c: np.ndarray = coarsen(T_mid_zyx)
        RH_c: np.ndarray = coarsen(RH_zyx)
        qc_c: np.ndarray = coarsen(qc_zyx)
        qi_c: np.ndarray = coarsen(qi_zyx)
        rel_c: np.ndarray = coarsen(rel_zyx)
        dei_r_c: np.ndarray = coarsen(dei_r_zyx)

        vmr_h2o_c: np.ndarray = coarsen(vmr_h2o_zyx)
        vmr_co2_c: np.ndarray = coarsen(vmr_co2_zyx)
        vmr_o3_c: np.ndarray = coarsen(vmr_o3_zyx)
        vmr_n2o_c: np.ndarray = coarsen(vmr_n2o_zyx)
        vmr_co_c: np.ndarray = coarsen(vmr_co_zyx)
        vmr_ch4_c: np.ndarray = coarsen(vmr_ch4_zyx)
        vmr_o2_c: np.ndarray = coarsen(vmr_o2_zyx)
        vmr_n2_c: np.ndarray = coarsen(vmr_n2_zyx)

        for jy in range(ny):
            for ix2 in range(nx):
                zmid_1: np.ndarray = z_mid_c[:, jy, ix2]
                zint_1: np.ndarray = z_int_c[:, jy, ix2]
                pmid_1: np.ndarray = p_mid_c[:, jy, ix2]
                pint_1: np.ndarray = p_int_c[:, jy, ix2]

                plev_1: np.ndarray
                play_1: np.ndarray
                plev_1, play_1 = interp_profile_interleaved_pressure(
                    z_int=zint_1, p_int=pint_1, z_mid=zmid_1, p_mid=pmid_1, z_lev_tgt=vgrid.z_lev, z_lay_tgt=vgrid.z_lay
                )
                if np.any(~np.isfinite(plev_1)) or np.any(~np.isfinite(play_1)):
                    raise ValueError(f"Non-finite pressure after interpolation at time={tt}, y={jy}, x={ix2}, lr={lr}")
                p_lev[it_local, :, jy, ix2] = plev_1
                p_lay[it_local, :, jy, ix2] = play_1

                Tmid_1: np.ndarray = T_mid_c[:, jy, ix2]
                tlay_1: np.ndarray = interp_profile_mid_only(zmid_1, Tmid_1, vgrid.z_lay, what="temperature(lay)")
                tlev_1: np.ndarray = interp_profile_mid_only(zmid_1, Tmid_1, vgrid.z_lev, what="temperature(lev)")
                t_lay[it_local, :, jy, ix2] = tlay_1
                t_lev[it_local, :, jy, ix2] = tlev_1

                rh[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, RH_c[:, jy, ix2], vgrid.z_lay, what="rh")
                rel[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, rel_c[:, jy, ix2], vgrid.z_lay, what="rel")
                dei[it_local, :, jy, ix2] = 2.0 * interp_profile_mid_only(zmid_1, dei_r_c[:, jy, ix2], vgrid.z_lay, what="dei_radius")

                qc1: np.ndarray = interp_profile_mid_only(zmid_1, qc_c[:, jy, ix2], vgrid.z_lay, what="qc")
                qi1: np.ndarray = interp_profile_mid_only(zmid_1, qi_c[:, jy, ix2], vgrid.z_lay, what="qi")
                qc_lay[it_local, :, jy, ix2] = qc1
                qi_lay[it_local, :, jy, ix2] = qi1

                vmr_h2o[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_h2o_c[:, jy, ix2], vgrid.z_lay, what="vmr_h2o")
                vmr_co2[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_co2_c[:, jy, ix2], vgrid.z_lay, what="vmr_co2")
                vmr_o3[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_o3_c[:, jy, ix2], vgrid.z_lay, what="vmr_o3")
                vmr_n2o[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_n2o_c[:, jy, ix2], vgrid.z_lay, what="vmr_n2o")
                vmr_co[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_co_c[:, jy, ix2], vgrid.z_lay, what="vmr_co")
                vmr_ch4[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_ch4_c[:, jy, ix2], vgrid.z_lay, what="vmr_ch4")
                vmr_o2[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_o2_c[:, jy, ix2], vgrid.z_lay, what="vmr_o2")
                vmr_n2[it_local, :, jy, ix2] = interp_profile_mid_only(zmid_1, vmr_n2_c[:, jy, ix2], vgrid.z_lay, what="vmr_n2")

        dp: np.ndarray = p_lev[it_local, 1:, :, :] - p_lev[it_local, :-1, :, :]
        lwp[it_local, :, :, :] = qc_lay[it_local, :, :, :] * dp / G_ACCEL
        iwp[it_local, :, :, :] = qi_lay[it_local, :, :, :] * dp / G_ACCEL

    ds_out: xr.Dataset = xr.Dataset(
        data_vars=dict(
            emis_sfc=(("time", "y", "x", "band_lw"), emis_sfc),
            sfc_alb_dir=(("time", "y", "x", "band_sw"), sfc_alb_dir),
            sfc_alb_dif=(("time", "y", "x", "band_sw"), sfc_alb_dif),
            tsi=(("time", "y", "x"), tsi),
            azi=(("time", "y", "x"), azi),
            p_lay=(("time", "lay", "y", "x"), p_lay),
            p_lev=(("time", "lev", "y", "x"), p_lev),
            t_lay=(("time", "lay", "y", "x"), t_lay),
            t_lev=(("time", "lev", "y", "x"), t_lev),
            rh=(("time", "lay", "y", "x"), rh),
            lwp=(("time", "lay", "y", "x"), lwp),
            iwp=(("time", "lay", "y", "x"), iwp),
            rel=(("time", "lay", "y", "x"), rel),
            dei=(("time", "lay", "y", "x"), dei),
            vmr_ch4=(("time", "lay", "y", "x"), vmr_ch4),
            vmr_co=(("time", "lay", "y", "x"), vmr_co),
            vmr_co2=(("time", "lay", "y", "x"), vmr_co2),
            vmr_h2o=(("time", "lay", "y", "x"), vmr_h2o),
            vmr_n2=(("time", "lay", "y", "x"), vmr_n2),
            vmr_n2o=(("time", "lay", "y", "x"), vmr_n2o),
            vmr_o2=(("time", "lay", "y", "x"), vmr_o2),
            vmr_o3=(("time", "lay", "y", "x"), vmr_o3),
            t_sfc=(("time", "y", "x"), t_sfc),
            mu0=(("time", "y", "x"), mu0),
            ngrid_x=((), np.int64(overrides["ngrid_x"][coarse.lr])),
            ngrid_y=((), np.int64(overrides["ngrid_y"][coarse.lr])),
            ngrid_z=((), np.int64(overrides["ngrid_z"])),
        ),
        coords=dict(
            time=(("time",), time_hours.astype(np.float64)),
            x=(("x",), coarse.x.astype(np.float64)),
            y=(("y",), coarse.y.astype(np.float64)),
            xh=(("xh",), coarse.xh.astype(np.float64)),
            yh=(("yh",), coarse.yh.astype(np.float64)),
            z=(("z",), vgrid.z_lay.astype(np.float64)),
            zh=(("zh",), vgrid.z_lev.astype(np.float64)),
            z_lay=(("z_lay",), vgrid.z_lay.astype(np.float64)),
            z_lev=(("z_lev",), vgrid.z_lev.astype(np.float64)),
            lay=(("lay",), np.arange(nlay, dtype=np.int64)),
            lev=(("lev",), np.arange(nlev, dtype=np.int64)),
            band_lw=(("band_lw",), np.arange(band_lw, dtype=np.int64)),
            band_sw=(("band_sw",), np.arange(band_sw, dtype=np.int64)),
        ),
    )

    # Minimal attrs + correct VMR units
    ds_out["time"].attrs["units"] = "hours since simulation start"
    for vname in ["vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o", "vmr_o2", "vmr_o3"]:
        ds_out[vname].attrs["units"] = "mol/mol"

    return ds_out


def write_netcdf(ds: xr.Dataset, path: str, compress_level: int) -> None:
    encoding: Dict[str, Dict[str, Any]] = {}
    for v in ds.data_vars:
        if ds[v].ndim == 0:
            continue
        encoding[v] = dict(zlib=True, complevel=int(compress_level), _FillValue=np.nan, dtype="float64")
    encoding["time"] = dict(dtype="float64")
    ds.to_netcdf(path, mode="w", format="NETCDF4", engine="netcdf4", encoding=encoding)


def consolidate_zarr_shards_to_netcdf(shard_paths: Sequence[str], out_path: str, compress_level: int) -> None:
    dsets: List[xr.Dataset] = [xr.open_zarr(p, consolidated=False) for p in shard_paths]
    ds_all: xr.Dataset = xr.concat(dsets, dim="time")
    order: np.ndarray = np.argsort(ds_all["time"].values, kind="stable")
    ds_all = ds_all.isel(time=xr.DataArray(order, dims=("time",)))
    write_netcdf(ds_all, out_path, compress_level=compress_level)
    for d in dsets:
        d.close()  # type: ignore[attr-defined]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args: argparse.Namespace = parse_args(argv)
    mpi: MpiInfo = get_mpi()

    log(mpi, "Starting dp_scream_to_rte conversion")

    out_dir: str = args.rte_rrtmgp_cpp_input_dir
    os.makedirs(out_dir, exist_ok=True)

    tmp_dir: str = args.tmp_dir or os.path.join(out_dir, ".tmp_dp2rte")
    if mpi.rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    barrier(mpi)

    if mpi.rank == 0:
        files0: List[str] = discover_input_files(args.dp_screamxx_output_dir, args.dp_screamxx_output_file_root)
        startdate0: str = parse_startdate_from_any_filename(files0)
    else:
        files0 = []
        startdate0 = ""

    files: List[str] = bcast(mpi, files0, root=0)
    startdate: str = bcast(mpi, startdate0, root=0)

    ds_raw: xr.Dataset = open_dp_scream_mfdataset(files, chunks_time=int(args.chunks_time))
    ds: xr.Dataset = dedup_sort_time_keep_last(ds_raw)

    tvals: np.ndarray = ds["time"].values
    t0: np.datetime64 = np.min(tvals)
    time_hours: np.ndarray = ((tvals - t0) / np.timedelta64(1, "h")).astype(np.float64)

    hmap: HorizontalMap = build_horizontal_map(ds)

    z_min: float
    z_max: float
    z_min, z_max = compute_common_vertical_domain(ds)

    dp_nlay_default: int = int(ds.dims.get("lev", 0))
    nlay: int = int(args.nlay) if args.nlay is not None else dp_nlay_default
    if nlay < 1:
        raise ValueError("nlay must be >= 1")
    vgrid: VerticalGrid = build_uniform_vertical_grid(z_min=z_min, z_max=z_max, nlay=nlay)

    band_lw: int = int(ds.dims.get("lwband", 16))
    band_sw: int = int(ds.dims.get("swband", 14))

    emis_band: np.ndarray = parse_scalar_or_list(str(args.emis_sfc), n=band_lw)
    alb_dir_band: np.ndarray = parse_scalar_or_list(str(args.sfc_alb_dir), n=band_sw)
    alb_dif_band: np.ndarray = parse_scalar_or_list(str(args.sfc_alb_dif), n=band_sw)

    null_factor: int = int(args.null_factor)
    if null_factor <= 0:
        raise ValueError("--null-factor must be positive")

    ngrid_x: Dict[int, int] = {}
    ngrid_y: Dict[int, int] = {}
    for lr in args.lr:
        coarse0: CoarseGrid = build_coarse_grid(hmap, int(lr))
        ngrid_x[int(lr)] = int(math.ceil(coarse0.nx / null_factor))
        ngrid_y[int(lr)] = int(math.ceil(coarse0.ny / null_factor))
    ngrid_z: int = int(math.ceil(nlay / null_factor))

    overrides: Dict[str, Any] = dict(
        emis_band=emis_band,
        alb_dir_band=alb_dir_band,
        alb_dif_band=alb_dif_band,
        tsi=float(args.tsi),
        azi=float(args.azi),
        time_hours=time_hours,
        ngrid_x=ngrid_x,
        ngrid_y=ngrid_y,
        ngrid_z=ngrid_z,
    )

    nt_total: int = int(ds.dims["time"])
    counts: List[int] = [nt_total // mpi.size] * mpi.size
    for r in range(nt_total % mpi.size):
        counts[r] += 1
    offsets: List[int] = [0]
    for r in range(1, mpi.size):
        offsets.append(offsets[r - 1] + counts[r - 1])
    i0: int = offsets[mpi.rank]
    i1: int = i0 + counts[mpi.rank]
    local_time_idx: np.ndarray = np.arange(i0, i1, dtype=np.int64)

    if mpi.rank == 0:
        print(
            f"[rank 0] nt_total={nt_total}, ranks={mpi.size}, z_common=[{z_min:.6g},{z_max:.6g}], "
            f"dp_nlay={dp_nlay_default}, out_nlay={nlay}, grid(nx,ny)=({hmap.nx},{hmap.ny})",
            flush=True,
        )

    for lr in args.lr:
        lr_i: int = int(lr)
        coarse: CoarseGrid = build_coarse_grid(hmap, lr_i)

        ds_shard: xr.Dataset = convert_time_indices_to_rte_shard(
            ds=ds,
            hmap=hmap,
            vgrid=vgrid,
            time_indices=local_time_idx,
            coarse=coarse,
            band_lw=band_lw,
            band_sw=band_sw,
            overrides=overrides,
        )

        shard_path: str = os.path.join(tmp_dir, f"lr_{lr_i:02d}.rank_{mpi.rank:05d}.zarr")
        if os.path.exists(shard_path):
            shutil.rmtree(shard_path)
        ds_shard.to_zarr(shard_path, mode="w")
        ds_shard.close()  # type: ignore[attr-defined]

        barrier(mpi)

        if mpi.rank == 0:
            shard_paths: List[str] = [os.path.join(tmp_dir, f"lr_{lr_i:02d}.rank_{r:05d}.zarr") for r in range(mpi.size)]
            for sp in shard_paths:
                if not os.path.exists(sp):
                    raise FileNotFoundError(f"Missing shard: {sp}")

            out_name: str = f"{args.dp_screamxx_output_file_root}.{startdate}.lr_{lr_i:02d}.in.nc"
            out_path: str = os.path.join(out_dir, out_name)
            print(f"[rank 0] Consolidating lr={lr_i} -> {out_path}", flush=True)

            consolidate_zarr_shards_to_netcdf(shard_paths=shard_paths, out_path=out_path, compress_level=int(args.compress_level))

            for sp in shard_paths:
                shutil.rmtree(sp)

        barrier(mpi)

    if mpi.rank == 0:
        try:
            if os.path.isdir(tmp_dir) and len(os.listdir(tmp_dir)) == 0:
                os.rmdir(tmp_dir)
        except Exception:
            pass

    ds.close()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())