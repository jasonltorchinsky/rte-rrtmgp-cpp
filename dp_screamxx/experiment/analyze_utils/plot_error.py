#!/usr/bin/env python3
import os
import re
import glob
import argparse
import sys
import time as pytime
import traceback
import gc

import numpy as np
import xarray as xr
from mpi4py import MPI
import matplotlib.pyplot as plt


PLOT_COLORS = [
    "#332288", "#117733", "#44AA99", "#88CCEE",
    "#DDCC77", "#CC6677", "#AA4499", "#882255"
]


def log(msg, comm=None, root_only=True):
    ts = pytime.strftime("%Y-%m-%d %H:%M:%S")
    if comm is None:
        print(f"[{ts}] {msg}", flush=True)
        return
    rank = comm.Get_rank()
    if (not root_only) or rank == 0:
        print(f"[{ts}] [rank {rank}] {msg}", flush=True)


def arr_gb(a):
    if a is None:
        return 0.0
    try:
        return float(a.nbytes) / (1024.0 ** 3)
    except Exception:
        return 0.0


def decompose_1d(n, size, rank):
    counts = np.full(size, n // size, dtype=int)
    counts[: n % size] += 1
    starts = np.zeros(size, dtype=int)
    starts[1:] = np.cumsum(counts[:-1])
    i0 = int(starts[rank])
    i1 = int(i0 + counts[rank])
    return i0, i1


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {v!r}")


def parse_error_types(s):
    allowed = ("rmse", "mae", "mbe")
    vals = [x.strip().lower() for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one error type.")
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unsupported error type(s): {bad}. Allowed values are {allowed}."
        )
    out = []
    seen = set()
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def find_pairs(input_dir, output_dir):
    in_files = sorted(glob.glob(os.path.join(input_dir, "*.in.nc")))
    out_files = sorted(glob.glob(os.path.join(output_dir, "*.out.nc")))

    lr_re = re.compile(r"(lr_\d+)")
    in_map = {}
    out_map = {}

    for p in in_files:
        m = lr_re.search(os.path.basename(p))
        if m:
            in_map[m.group(1)] = p

    for p in out_files:
        m = lr_re.search(os.path.basename(p))
        if m:
            out_map[m.group(1)] = p

    lrs = sorted(set(in_map.keys()) & set(out_map.keys()))
    return {lr: (in_map[lr], out_map[lr]) for lr in lrs}


def find_day_segments(time_hours, mu0_1d, night_eps=1.0e-3):
    time_hours = np.asarray(time_hours, dtype=np.float64)
    mu0_1d = np.asarray(mu0_1d, dtype=np.float64)

    is_day = np.isfinite(mu0_1d) & (mu0_1d > night_eps)
    segments = []

    if is_day.size == 0:
        return segments

    in_seg = False
    start = None
    for i, flag in enumerate(is_day):
        if flag and not in_seg:
            start = i
            in_seg = True
        elif not flag and in_seg:
            end = i
            if end > start:
                segments.append(
                    {
                        "i0": int(start),
                        "i1": int(end),
                        "t0": float(time_hours[start]),
                        "t1": float(time_hours[end - 1]),
                        "tmid": float(0.5 * (time_hours[start] + time_hours[end - 1])),
                    }
                )
            in_seg = False
            start = None

    if in_seg:
        end = len(is_day)
        if end > start:
            segments.append(
                {
                    "i0": int(start),
                    "i1": int(end),
                    "t0": float(time_hours[start]),
                    "t1": float(time_hours[end - 1]),
                    "tmid": float(0.5 * (time_hours[start] + time_hours[end - 1])),
                }
            )

    return segments


def map_output_time_to_input_index(time_out, ntime_in):
    time_out = np.asarray(time_out, dtype=np.float64)
    idx = np.rint(time_out).astype(np.int64)

    if np.any(~np.isfinite(time_out)):
        raise ValueError("Output time coordinate contains non-finite values.")
    if np.any(np.abs(time_out - idx) > 1.0e-12):
        raise ValueError("Output time coordinate is not integer-valued as expected.")
    if np.any(idx < 0) or np.any(idx >= ntime_in):
        raise ValueError(
            f"Output time indices are out of bounds for input time array: "
            f"min idx={idx.min()}, max idx={idx.max()}, input size={ntime_in}"
        )

    return idx


def safe_profile_label_from_input(ds_in):
    if "xh" not in ds_in.variables:
        return "unknown"

    xh = ds_in["xh"].values.astype(np.float64)
    if xh.ndim != 1 or xh.size < 2:
        return "unknown"

    dx = xh[1] - xh[0]
    if dx < 1000.0:
        return r"{:0.0f} $m$".format(dx)
    return r"{:0.2f} $km$".format(dx / 1000.0)


def reduce_error_timeseries(comm, ray_loc, ts_loc, error_type):
    valid = np.isfinite(ray_loc) & np.isfinite(ts_loc)

    if error_type == "rmse":
        diff = ray_loc - ts_loc
        val = np.where(valid, diff * diff, 0.0)
        local_num = np.sum(val, axis=1, dtype=np.float64)
        local_den = np.sum(valid, axis=1, dtype=np.int64)

        global_num = comm.allreduce(local_num, op=MPI.SUM)
        global_den = comm.allreduce(local_den, op=MPI.SUM)

        if comm.Get_rank() != 0:
            return None

        out = np.full(global_num.shape, np.nan, dtype=np.float64)
        good = global_den > 0
        out[good] = np.sqrt(global_num[good] / global_den[good])
        return out

    if error_type == "mae":
        diff = np.abs(ts_loc - ray_loc)
        val = np.where(valid, diff, 0.0)
        local_num = np.sum(val, axis=1, dtype=np.float64)
        local_den = np.sum(valid, axis=1, dtype=np.int64)

        global_num = comm.allreduce(local_num, op=MPI.SUM)
        global_den = comm.allreduce(local_den, op=MPI.SUM)

        if comm.Get_rank() != 0:
            return None

        out = np.full(global_num.shape, np.nan, dtype=np.float64)
        good = global_den > 0
        out[good] = global_num[good] / global_den[good]
        return out

    if error_type == "mbe":
        ts_val = np.where(valid, ts_loc, 0.0)
        ray_val = np.where(valid, ray_loc, 0.0)

        local_ts_sum = np.sum(ts_val, axis=1, dtype=np.float64)
        local_ray_sum = np.sum(ray_val, axis=1, dtype=np.float64)
        local_den = np.sum(valid, axis=1, dtype=np.int64)

        global_ts_sum = comm.allreduce(local_ts_sum, op=MPI.SUM)
        global_ray_sum = comm.allreduce(local_ray_sum, op=MPI.SUM)
        global_den = comm.allreduce(local_den, op=MPI.SUM)

        if comm.Get_rank() != 0:
            return None

        out = np.full(global_ts_sum.shape, np.nan, dtype=np.float64)
        good = global_den > 0
        out[good] = (global_ray_sum[good] / global_den[good]) - (global_ts_sum[good] / global_den[good])
        return out

    raise ValueError(f"Unsupported error_type={error_type}")


def get_first_dim_name(da, candidates):
    for d in candidates:
        if d in da.dims:
            return d
    raise ValueError(f"None of candidate dims {candidates} found in {da.dims}")


def subset_cols_2d(a, nt, loc_a, loc_b):
    return a.reshape(nt, -1)[:, loc_a:loc_b]


def subset_cols_3d_layers(a, nt, nlay, loc_a, loc_b):
    return a.reshape(nt, nlay, -1)[:, :, loc_a:loc_b].reshape(nt, -1)


def choose_case_layout(ncases, nranks, min_group=8, max_group=32, serial_case_processing=False):
    if ncases <= 0:
        return 0, 0

    if serial_case_processing:
        return 1, np.array([nranks], dtype=int)

    best = None
    max_concurrent = min(ncases, nranks)

    for nactive_cases in range(max_concurrent, 0, -1):
        base = nranks // nactive_cases
        rem = nranks % nactive_cases
        sizes = np.array([base + (1 if i < rem else 0) for i in range(nactive_cases)], dtype=int)

        if np.any(sizes <= 0):
            continue

        score = 0
        for s in sizes:
            if min_group <= s <= max_group:
                score += 3
            elif s < min_group:
                score += 1
            else:
                score += 2

        score += 0.01 * nactive_cases

        cand = (score, nactive_cases, sizes)
        if best is None or cand[0] > best[0]:
            best = cand

    _, nactive_cases, sizes = best
    return nactive_cases, sizes


def assign_case_and_subrank(ncases, nranks, rank, min_group=8, max_group=32, serial_case_processing=False):
    nactive_cases, sizes = choose_case_layout(
        ncases,
        nranks,
        min_group=min_group,
        max_group=max_group,
        serial_case_processing=serial_case_processing,
    )

    if nactive_cases == 0:
        return MPI.UNDEFINED, -1, None

    case_ids = list(range(nactive_cases))
    offsets = np.zeros(nactive_cases + 1, dtype=int)
    offsets[1:] = np.cumsum(sizes)

    my_case = MPI.UNDEFINED
    my_subrank = -1

    for c in case_ids:
        if offsets[c] <= rank < offsets[c + 1]:
            my_case = c
            my_subrank = rank - offsets[c]
            break

    return my_case, my_subrank, sizes


def process_one_case(case_idx, lr_tag, in_path, out_path, args, subcomm, world_rank):
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()

    t0_wall = MPI.Wtime()
    if subrank == 0:
        log(f"Processing {lr_tag} with subgroup size {subsize} (world root rank {world_rank})")
        log(f"{lr_tag}: opening input={in_path}")
        log(f"{lr_tag}: opening output={out_path}")

    fluxes = [
        ("sw_sfc_up", r"Upwelling Surface Flux $\left[W\,m^{-2}\right]$"),
        ("sw_sfc_dn", r"Downwelling Surface Flux $\left[W\,m^{-2}\right]$"),
        ("sw_tod_up", r"Upwelling Top-of-Domain Flux $\left[W\,m^{-2}\right]$"),
        ("flux_abs", r"Absorbed Flux $\left[W\,m^{-3}\right]$"),
        ("flux_abs_vi", r"Vertically-Integrated Absorbed Flux $\left[W\,m^{-2}\right]$"),
    ]

    local_results = {
        error_type: {flux_tag: [] for flux_tag, _ in fluxes}
        for error_type in args.error_type
    }

    with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in:
        time_in = ds_in["time"].astype("float64").load().values

        if "mu0" not in ds_in.variables:
            raise KeyError(f"{in_path} missing variable 'mu0' needed for daytime masking")
        mu0 = ds_in["mu0"].astype("float64").load().values

        if "z_lev" not in ds_in.variables:
            raise KeyError(f"{in_path} missing variable 'z_lev' needed for absorbed-flux diagnostics")
        z_lev_in = ds_in["z_lev"].astype("float64").load().values

        profile_label = safe_profile_label_from_input(ds_in)

    if subrank == 0:
        log(f"{lr_tag}: time_in shape={time_in.shape}, mu0 shape={mu0.shape}, z_lev shape={z_lev_in.shape}")

    if args.zmax is None:
        ilev_max = len(z_lev_in)
    else:
        zmax_m = 1000.0 * args.zmax
        ilev_max = int(np.searchsorted(z_lev_in, zmax_m, side="right"))
        ilev_max = max(2, min(ilev_max, len(z_lev_in)))
        if subrank == 0:
            iz = min(ilev_max - 1, len(z_lev_in) - 1)
            log(f"{lr_tag}: input zmax={args.zmax}, actual zmax={z_lev_in[iz]}")

    z_lev_use = np.asarray(z_lev_in[:ilev_max], dtype=np.float64)

    with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out:
        time_out = ds_out["time"].astype("float64").load().values
        nt_out = int(ds_out.sizes["time"])
        ny = int(ds_out.sizes["y"])
        nx = int(ds_out.sizes["x"])
        ncol = ny * nx

        if subrank == 0:
            log(f"{lr_tag}: time_out shape={time_out.shape}, nx={nx}, ny={ny}, ncol={ncol}")

        if time_out.size != nt_out:
            raise ValueError(
                f"{out_path}: output time coordinate length {time_out.size} "
                f"does not match time dimension {nt_out}"
            )

        time_idx = map_output_time_to_input_index(time_out, time_in.size)
        time_use = time_in[time_idx]
        nt = time_use.size

        mu0_use = mu0[time_idx, :, :]
        day_valid_3d = np.isfinite(mu0_use) & (mu0_use > args.night_eps)

        del mu0_use
        gc.collect()

        if args.time_units == "hours":
            time_x = time_use
            xlabel = "Time Since Simulation Start [Hours]"
        else:
            time_x = time_idx.astype(np.float64)
            xlabel = "Input time-step index"

        if subrank == 0:
            mu0_1d = mu0[time_idx, 0, 0]
            day_segments = find_day_segments(time_x, mu0_1d, night_eps=args.night_eps)
            common_xlabel = xlabel
            log(f"{lr_tag}: detected {len(day_segments)} daytime segment(s)")
        else:
            day_segments = None
            common_xlabel = None

        i0, i1 = decompose_1d(ncol, subsize, subrank)
        log(f"{lr_tag}: subgroup rank {subrank} assigned col slab [{i0}:{i1}) (nloc={i1-i0})", root_only=False)

        if i1 > i0:
            j0 = i0 // nx
            j1 = (i1 - 1) // nx
            y0 = j0
            y1 = j1 + 1
            slab_start_col = y0 * nx
            loc_a = i0 - slab_start_col
            loc_b = i1 - slab_start_col

            day_loc_col = day_valid_3d[:, y0:y1, :].reshape(nt, -1)[:, loc_a:loc_b]
        else:
            y0 = 0
            y1 = 0
            loc_a = 0
            loc_b = 0
            day_loc_col = np.empty((nt, 0), dtype=bool)

        del day_valid_3d
        gc.collect()

        if i1 > i0:
            rt_flux_sfc_up = (
                ds_out["rt_flux_sfc_up"]
                .transpose("time", "y", "x")
                .isel(time=slice(0, nt), y=slice(y0, y1))
                .astype("float64")
                .load()
                .values
            )
            rt_flux_sfc_dir = (
                ds_out["rt_flux_sfc_dir"]
                .transpose("time", "y", "x")
                .isel(time=slice(0, nt), y=slice(y0, y1))
                .astype("float64")
                .load()
                .values
            )
            rt_flux_sfc_dif = (
                ds_out["rt_flux_sfc_dif"]
                .transpose("time", "y", "x")
                .isel(time=slice(0, nt), y=slice(y0, y1))
                .astype("float64")
                .load()
                .values
            )
            rt_flux_tod_up = (
                ds_out["rt_flux_tod_up"]
                .transpose("time", "y", "x")
                .isel(time=slice(0, nt), y=slice(y0, y1))
                .astype("float64")
                .load()
                .values
            )

            sw_up_da = ds_out["sw_flux_up"]
            sw_dn_da = ds_out["sw_flux_dn"]
            sw_lev_dim = get_first_dim_name(sw_up_da, ("lev", "z", "z_lev"))

            sw_up_all = (
                sw_up_da
                .transpose("time", sw_lev_dim, "y", "x")
                .isel(time=slice(0, nt), y=slice(y0, y1), **{sw_lev_dim: slice(0, ilev_max)})
                .astype("float64")
                .load()
                .values
            )

            sw_dn_all = (
                sw_dn_da
                .transpose("time", sw_lev_dim, "y", "x")
                .isel(time=slice(0, nt), y=slice(y0, y1), **{sw_lev_dim: slice(0, ilev_max)})
                .astype("float64")
                .load()
                .values
            )

            nlev = sw_up_all.shape[1]
            if not (-nlev <= args.lev_sfc_idx < nlev):
                raise IndexError(f"{lr_tag}: lev_sfc_idx={args.lev_sfc_idx} out of bounds for nlev={nlev}")
            if not (-nlev <= args.lev_top_idx < nlev):
                raise IndexError(f"{lr_tag}: lev_top_idx={args.lev_top_idx} out of bounds for nlev={nlev}")

            if "rt_flux_abs_dif" in ds_out.variables and "rt_flux_abs_dir" in ds_out.variables:
                rt_abs_dif_da = ds_out["rt_flux_abs_dif"]
                rt_abs_dir_da = ds_out["rt_flux_abs_dir"]
                rt_lay_dim = get_first_dim_name(rt_abs_dif_da, ("lay", "z", "z_lay"))

                rt_abs_dif_all = (
                    rt_abs_dif_da
                    .transpose("time", rt_lay_dim, "y", "x")
                    .isel(time=slice(0, nt), y=slice(y0, y1), **{rt_lay_dim: slice(0, ilev_max - 1)})
                    .astype("float64")
                    .load()
                    .values
                )

                rt_abs_dir_all = (
                    rt_abs_dir_da
                    .transpose("time", rt_lay_dim, "y", "x")
                    .isel(time=slice(0, nt), y=slice(y0, y1), **{rt_lay_dim: slice(0, ilev_max - 1)})
                    .astype("float64")
                    .load()
                    .values
                )
            else:
                rt_abs_dif_all = None
                rt_abs_dir_all = None

            if subrank == 0:
                log(f"{lr_tag}: rt_flux_sfc_up shape={rt_flux_sfc_up.shape}, {arr_gb(rt_flux_sfc_up):.3f} GiB")
                log(f"{lr_tag}: sw_up_all shape={sw_up_all.shape}, {arr_gb(sw_up_all):.3f} GiB")
                log(f"{lr_tag}: sw_dn_all shape={sw_dn_all.shape}, {arr_gb(sw_dn_all):.3f} GiB")
                if rt_abs_dif_all is not None:
                    log(f"{lr_tag}: rt_abs_dif_all shape={rt_abs_dif_all.shape}, {arr_gb(rt_abs_dif_all):.3f} GiB")
                    log(f"{lr_tag}: rt_abs_dir_all shape={rt_abs_dir_all.shape}, {arr_gb(rt_abs_dir_all):.3f} GiB")

            sw_sfc_up_ts_loc = subset_cols_2d(sw_up_all[:, args.lev_sfc_idx, :, :], nt, loc_a, loc_b)
            sw_sfc_dn_ts_loc = subset_cols_2d(sw_dn_all[:, args.lev_sfc_idx, :, :], nt, loc_a, loc_b)
            sw_tod_up_ts_loc = subset_cols_2d(sw_up_all[:, args.lev_top_idx, :, :], nt, loc_a, loc_b)

            sw_sfc_up_ray_loc = subset_cols_2d(rt_flux_sfc_up, nt, loc_a, loc_b)
            sw_sfc_dn_ray_loc = subset_cols_2d(rt_flux_sfc_dir + rt_flux_sfc_dif, nt, loc_a, loc_b)
            sw_tod_up_ray_loc = subset_cols_2d(rt_flux_tod_up, nt, loc_a, loc_b)

            sw_sfc_up_ray_loc = np.where(day_loc_col, sw_sfc_up_ray_loc, np.nan)
            sw_sfc_up_ts_loc = np.where(day_loc_col, sw_sfc_up_ts_loc, np.nan)
            sw_sfc_dn_ray_loc = np.where(day_loc_col, sw_sfc_dn_ray_loc, np.nan)
            sw_sfc_dn_ts_loc = np.where(day_loc_col, sw_sfc_dn_ts_loc, np.nan)
            sw_tod_up_ray_loc = np.where(day_loc_col, sw_tod_up_ray_loc, np.nan)
            sw_tod_up_ts_loc = np.where(day_loc_col, sw_tod_up_ts_loc, np.nan)

            del rt_flux_sfc_up, rt_flux_sfc_dir, rt_flux_sfc_dif, rt_flux_tod_up
            gc.collect()

            z_lev = z_lev_use[:sw_up_all.shape[1]]
            if z_lev.ndim != 1:
                raise ValueError(f"Expected 1D z_lev, got shape {z_lev.shape}")
            if z_lev.size < sw_up_all.shape[1]:
                raise ValueError(
                    f"z_lev has size {z_lev.size}, but sw_flux_up/sw_flux_dn need at least {sw_up_all.shape[1]} levels"
                )

            dz = np.diff(z_lev)
            nlay = dz.size

            ts_flux_diff_all = (
                (sw_up_all[:, :-1, :, :] - sw_dn_all[:, :-1, :, :]) +
                (sw_dn_all[:, 1:, :, :] - sw_up_all[:, 1:, :, :])
            )
            ts_abs_all = ts_flux_diff_all / dz[None, :, None, None]

            del ts_flux_diff_all
            gc.collect()

            if rt_abs_dif_all is None or rt_abs_dir_all is None:
                raise KeyError(f"{out_path} missing rt_flux_abs_dif and/or rt_flux_abs_dir")

            rt_abs_all = rt_abs_dif_all + rt_abs_dir_all
            del rt_abs_dif_all, rt_abs_dir_all
            gc.collect()

            if rt_abs_all.shape[1] != nlay:
                raise ValueError(
                    f"Ray absorbed-flux layer count ({rt_abs_all.shape[1]}) does not match dz layer count ({nlay})"
                )

            day_loc_cell = np.repeat(day_loc_col[:, None, :], nlay, axis=1).reshape(nt, -1)

            flux_abs_ts_loc = subset_cols_3d_layers(ts_abs_all, nt, nlay, loc_a, loc_b)
            flux_abs_ray_loc = subset_cols_3d_layers(rt_abs_all, nt, nlay, loc_a, loc_b)
            flux_abs_ts_loc = np.where(day_loc_cell, flux_abs_ts_loc, np.nan)
            flux_abs_ray_loc = np.where(day_loc_cell, flux_abs_ray_loc, np.nan)

            ts_vi_all = np.sum(ts_abs_all * dz[None, :, None, None], axis=1)
            rt_vi_all = np.sum(rt_abs_all * dz[None, :, None, None], axis=1)

            del ts_abs_all, rt_abs_all
            gc.collect()

            flux_abs_vi_ts_loc = subset_cols_2d(ts_vi_all, nt, loc_a, loc_b)
            flux_abs_vi_ray_loc = subset_cols_2d(rt_vi_all, nt, loc_a, loc_b)
            flux_abs_vi_ts_loc = np.where(day_loc_col, flux_abs_vi_ts_loc, np.nan)
            flux_abs_vi_ray_loc = np.where(day_loc_col, flux_abs_vi_ray_loc, np.nan)

            del ts_vi_all, rt_vi_all, sw_up_all, sw_dn_all
            gc.collect()

        else:
            sw_sfc_up_ray_loc = np.empty((nt, 0), dtype=np.float64)
            sw_sfc_up_ts_loc = np.empty((nt, 0), dtype=np.float64)
            sw_sfc_dn_ray_loc = np.empty((nt, 0), dtype=np.float64)
            sw_sfc_dn_ts_loc = np.empty((nt, 0), dtype=np.float64)
            sw_tod_up_ray_loc = np.empty((nt, 0), dtype=np.float64)
            sw_tod_up_ts_loc = np.empty((nt, 0), dtype=np.float64)
            flux_abs_ray_loc = np.empty((nt, 0), dtype=np.float64)
            flux_abs_ts_loc = np.empty((nt, 0), dtype=np.float64)
            flux_abs_vi_ray_loc = np.empty((nt, 0), dtype=np.float64)
            flux_abs_vi_ts_loc = np.empty((nt, 0), dtype=np.float64)

        del day_loc_col
        gc.collect()

        flux_arrays = {
            "sw_sfc_up": (sw_sfc_up_ray_loc, sw_sfc_up_ts_loc),
            "sw_sfc_dn": (sw_sfc_dn_ray_loc, sw_sfc_dn_ts_loc),
            "sw_tod_up": (sw_tod_up_ray_loc, sw_tod_up_ts_loc),
            "flux_abs": (flux_abs_ray_loc, flux_abs_ts_loc),
            "flux_abs_vi": (flux_abs_vi_ray_loc, flux_abs_vi_ts_loc),
        }

        for flux_tag, flux_ylabel in fluxes:
            if subrank == 0:
                log(f"{lr_tag}: flux={flux_tag}")

            log(f"{lr_tag}: entering reductions for flux={flux_tag}", subcomm, root_only=False)

            ray_loc, ts_loc = flux_arrays[flux_tag]

            for error_type in args.error_type:
                metric = reduce_error_timeseries(subcomm, ray_loc, ts_loc, error_type)

                if subrank == 0:
                    local_results[error_type][flux_tag].append(
                        {
                            "label": profile_label,
                            "metric": metric,
                            "time_x": time_x.copy(),
                            "flux_ylabel": flux_ylabel,
                            "lr_tag": lr_tag,
                            "day_segments": day_segments,
                            "xlabel": common_xlabel,
                        }
                    )

            del ray_loc, ts_loc
            gc.collect()

        del flux_arrays
        gc.collect()

    del mu0, z_lev_use, time_in
    gc.collect()

    if subrank == 0:
        log(f"{lr_tag}: done (walltime {MPI.Wtime() - t0_wall:.2f} s).")
        return {
            "case_idx": case_idx,
            "lr_tag": lr_tag,
            "results": local_results,
        }

    return None


def metric_label(error_type):
    if error_type == "rmse":
        return "Root-Mean-Square Error"
    if error_type == "mae":
        return "Mean Absolute Error"
    if error_type == "mbe":
        return "Mean Bias Error (Ray-Tracer - Two-Stream)"
    return error_type.upper()


def should_use_log_scale(error_type, log_scale, series_list):
    if not log_scale:
        return False
    if error_type != "mbe":
        return True
    for s in series_list:
        y = np.asarray(s["metric"], dtype=np.float64)
        finite = np.isfinite(y)
        if np.any(finite & (y <= 0.0)):
            return False
    return True


def format_day_ticks(seg):
    return [seg["t0"], seg["tmid"], seg["t1"]]


def plot_error_figure(error_type, valid_case_results, fluxes, args, viz_dir):
    all_results = {flux_tag: [] for flux_tag, _ in fluxes}

    for cres in valid_case_results:
        for flux_tag, _ in fluxes:
            all_results[flux_tag].extend(cres["results"][error_type][flux_tag])

    reference_series = None
    for flux_tag, _ in fluxes:
        if all_results[flux_tag]:
            reference_series = all_results[flux_tag][0]
            break

    if reference_series is None:
        return

    day_segments = reference_series.get("day_segments", [])
    xlabel = reference_series.get("xlabel", "Time Since Simulation Start [Hours]")

    nrows = len(fluxes)
    ndays = len(day_segments)

    if ndays == 0:
        log(f"No daytime segments found for error_type={error_type}; skipping figure.")
        return

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ndays,
        figsize=(3.2 * ndays + 1.5, 2.6 * nrows + 1.6),
        sharex="col",
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )

    legend_handles = None
    legend_labels = None

    for row, (flux_tag, flux_ylabel) in enumerate(fluxes):
        series_list = all_results[flux_tag]
        use_log = should_use_log_scale(error_type, args.log_scale, series_list)

        for col, seg in enumerate(day_segments):
            ax = axes[row, col]
            ax.set_axisbelow(True)

            for i, s in enumerate(series_list):
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                tx = np.asarray(s["time_x"], dtype=np.float64)
                yy = np.asarray(s["metric"], dtype=np.float64)
                sl = slice(seg["i0"], seg["i1"])

                ax.plot(
                    tx[sl],
                    yy[sl],
                    linewidth=2,
                    color=color,
                    label=s["label"],
                )

            if legend_handles is None:
                legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()

            ax.set_xlim(seg["t0"], seg["t1"])
            xticks = format_day_ticks(seg)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{xticks[0]:g}", f"{xticks[1]:g}", f"{xticks[2]:g}"])
            ax.grid(False)
            ax.axvline(seg["tmid"], color="0.7", linewidth=0.8, zorder=0)

            if error_type == "mbe":
                ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="-", zorder=0)

            if use_log:
                ax.set_yscale("log")

            if col == 0:
                ax.set_ylabel(flux_ylabel)
            if row == 0:
                ax.set_title(f"Day {col + 1}")

            ax.set_xlabel("")

    fig.supylabel(metric_label(error_type))
    fig.supxlabel(xlabel)

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 6),
            bbox_to_anchor=(0.5, 1.025),
            frameon=True,
        )

    out_png = os.path.join(viz_dir, f"{error_type}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    log(f"Wrote: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rte_rrtmgp_cpp_input_dir_path",
        action="store",
        nargs=1,
        type=str,
        required=True,
        help="Directory containing RTE-RRTMGP-CPP input files (*.in).",
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_output_dir_path",
        action="store",
        nargs=1,
        type=str,
        required=True,
        help="Directory containing RTE-RRTMGP-CPP output files (*.out).",
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_viz_dir_path",
        action="store",
        nargs=1,
        type=str,
        required=True,
        help="Directory to write plots.",
    )
    parser.add_argument(
        "--night-eps",
        type=float,
        default=1.0e-3,
        help="Night when mu0 <= night_eps (default: 1.e-3).",
    )
    parser.add_argument(
        "--lev-sfc-idx",
        type=int,
        default=0,
        help="lev index to use for surface fluxes in sw_flux_* arrays (default: 0).",
    )
    parser.add_argument(
        "--lev-top-idx",
        type=int,
        default=-1,
        help="lev index to use for top-of-domain in sw_flux_* arrays (default: -1).",
    )
    parser.add_argument(
        "--time-units",
        type=str,
        default="hours",
        choices=["raw", "hours"],
        help="X-axis units derived from input time: raw or hours since start (default: hours).",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default=None,
        help="Comma-separated list of lr tags to plot, e.g. '01,04,16'. "
             "If omitted, plot all available lr_XX pairs.",
    )
    parser.add_argument(
        "--error-type",
        type=parse_error_types,
        default=["rmse"],
        help="Comma-separated list of error types to plot. Allowed: rmse,mae,mbe. "
             "Example: --error-type rmse,mae,mbe",
    )
    parser.add_argument(
        "--log-scale",
        type=str2bool,
        default=True,
        help="Use log scale on y-axis when supported (default: true).",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Maximum height to include [km]. Default: use all heights.",
    )
    parser.add_argument(
        "--min-case-ranks",
        type=int,
        default=8,
        help="Preferred minimum ranks per active case (default: 8).",
    )
    parser.add_argument(
        "--max-case-ranks",
        type=int,
        default=32,
        help="Preferred maximum ranks per active case (default: 32).",
    )
    parser.add_argument(
        "--serial-case-processing",
        type=str2bool,
        default=False,
        help="If true, run only one case at a time using all ranks.",
    )

    args = parser.parse_args()

    input_dir = args.rte_rrtmgp_cpp_input_dir_path[0]
    output_dir = args.rte_rrtmgp_cpp_output_dir_path[0]
    viz_dir = args.rte_rrtmgp_cpp_viz_dir_path[0]

    world = MPI.COMM_WORLD
    wrank = world.Get_rank()
    wsize = world.Get_size()

    if wrank == 0:
        log(f"Starting job with {wsize} MPI ranks", world)
        log(f"Input dir:  {input_dir}", world)
        log(f"Output dir: {output_dir}", world)
        log(f"Plot dir:   {viz_dir}", world)
        log(f"Error types: {args.error_type}", world)

    pairs = find_pairs(input_dir, output_dir)

    if wrank == 0:
        log(f"Found {len(pairs)} lr_XX pairs", world)
        for lr, (pin, pout) in pairs.items():
            log(f"  {lr}: in={os.path.basename(pin)} out={os.path.basename(pout)}", world)

    if not pairs:
        if wrank == 0:
            log("ERROR: No matching lr_XX pairs found.", world)
        sys.exit(1)

    if args.lr is not None:
        requested = [s.strip() for s in args.lr.split(",") if s.strip()]
        requested_tags = [s if s.startswith("lr_") else f"lr_{s}" for s in requested]
        pairs = {lr: pairs[lr] for lr in requested_tags if lr in pairs}

        if wrank == 0:
            log(f"After --lr filtering, {len(pairs)} pairs remain: {sorted(pairs.keys())}", world)

        if not pairs:
            if wrank == 0:
                log("ERROR: --lr requested resolutions not found in available pairs.", world)
            sys.exit(1)

    os.makedirs(viz_dir, exist_ok=True)

    case_items = sorted(pairs.items())
    ncases = len(case_items)

    if args.serial_case_processing:
        if wrank == 0:
            log("Serial case processing enabled: one case at a time.", world)

        gathered_all = []

        for active_case_idx in range(ncases):
            has_case = True
            subcomm = world.Split(color=0, key=wrank)

            my_lr_tag, my_paths = case_items[active_case_idx]
            my_in_path, my_out_path = my_paths

            case_result = process_one_case(
                active_case_idx, my_lr_tag, my_in_path, my_out_path, args, subcomm, wrank
            )

            gathered = world.gather(
                case_result if subcomm.Get_rank() == 0 else None,
                root=0
            )

            if wrank == 0:
                valid = [g for g in gathered if g is not None]
                gathered_all.extend(valid)

        if wrank != 0:
            return

        valid_case_results = gathered_all
        valid_case_results.sort(key=lambda d: d["case_idx"])

    else:
        my_case_idx, my_subrank, subgroup_sizes = assign_case_and_subrank(
            ncases=ncases,
            nranks=wsize,
            rank=wrank,
            min_group=args.min_case_ranks,
            max_group=args.max_case_ranks,
            serial_case_processing=False,
        )

        if wrank == 0:
            active_cases = len(subgroup_sizes) if subgroup_sizes is not None else 0
            log(f"Active concurrent cases: {active_cases}", world)
            if subgroup_sizes is not None:
                log(f"Subgroup sizes: {list(map(int, subgroup_sizes))}", world)

        has_case = (my_case_idx != MPI.UNDEFINED and my_case_idx < ncases)

        color = my_case_idx if has_case else MPI.UNDEFINED
        subcomm = world.Split(color=color, key=wrank)

        case_result = None
        if has_case:
            my_lr_tag, my_paths = case_items[my_case_idx]
            my_in_path, my_out_path = my_paths

            case_result = process_one_case(
                my_case_idx, my_lr_tag, my_in_path, my_out_path, args, subcomm, wrank
            )

        gathered = world.gather(
            case_result if (has_case and subcomm.Get_rank() == 0) else None,
            root=0
        )

        if wrank != 0:
            return

        valid_case_results = [g for g in gathered if g is not None]
        valid_case_results.sort(key=lambda d: d["case_idx"])

    fluxes = [
        ("sw_sfc_up", r"Upwelling Surface Flux $\left[W\,m^{-2}\right]$"),
        ("sw_sfc_dn", r"Downwelling Surface Flux $\left[W\,m^{-2}\right]$"),
        ("sw_tod_up", r"Upwelling Top-of-Domain Flux $\left[W\,m^{-2}\right]$"),
        ("flux_abs", r"Absorbed Flux $\left[W\,m^{-3}\right]$"),
        ("flux_abs_vi", r"Vertically-Integrated Absorbed Flux $\left[W\,m^{-2}\right]$"),
    ]

    for error_type in args.error_type:
        plot_error_figure(error_type, valid_case_results, fluxes, args, viz_dir)

    log("All resolutions complete.", world)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"\n===== Unhandled exception on rank {rank} =====", flush=True)
        traceback.print_exc()
        print(f"===== End exception on rank {rank} =====\n", flush=True)
        try:
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            sys.exit(1)