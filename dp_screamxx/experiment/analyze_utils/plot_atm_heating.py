#!/usr/bin/env python3
import os
import re
import glob
import argparse
import sys
import time as pytime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpi4py import MPI


def log(msg, comm=None, root_only=True):
    ts = pytime.strftime("%Y-%m-%d %H:%M:%S")
    if comm is None:
        print(f"[{ts}] {msg}", flush=True)
        return
    rank = comm.Get_rank()
    if (not root_only) or rank == 0:
        print(f"[{ts}] [rank {rank}] {msg}", flush=True)


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

    lrs = sorted(set(in_map.keys()) & set(out_map.keys()), key=lambda s: int(s.split("_")[1]), reverse=True)
    return {lr: (in_map[lr], out_map[lr]) for lr in lrs}


def decompose_1d(n, size, rank):
    counts = np.full(size, n // size, dtype=int)
    counts[: n % size] += 1
    starts = np.zeros(size, dtype=int)
    starts[1:] = np.cumsum(counts[:-1])
    i0 = int(starts[rank])
    i1 = int(i0 + counts[rank])
    return i0, i1


def find_day_night_transition_indices(mu0_1d, night_eps=1.0e-3):
    """
    Returns list of (start_idx, end_idx) for each daytime segment.
    Convention matches prior scripts:
      - night-to-day transition starts at i
      - day-to-night transition ends at i-1
    """
    segments = []
    n = len(mu0_1d)
    if n == 0:
        return segments

    is_day = np.isfinite(mu0_1d) & (mu0_1d > night_eps)

    start = None
    if is_day[0]:
        start = 0

    for i in range(1, n):
        if (not is_day[i - 1]) and is_day[i]:
            start = i
        elif is_day[i - 1] and (not is_day[i]):
            if start is not None:
                segments.append((start, i - 1))
                start = None

    if start is not None:
        segments.append((start, n - 1))

    return segments


def choose_representative_day_segments(day_segments):
    if not day_segments:
        raise ValueError("No daytime segments found.")
    if len(day_segments) == 1:
        return [day_segments[0]]
    return day_segments


def choose_times_within_day(start_idx, end_idx):
    """
    Morning: shortly after sunrise
    Noon: near midpoint of daytime interval
    Evening: about halfway between noon and sunset
    """
    if end_idx < start_idx:
        raise ValueError(f"Invalid daytime interval: {start_idx}, {end_idx}")

    npts = end_idx - start_idx + 1
    if npts == 1:
        return [start_idx, start_idx, start_idx]

    morning = start_idx + max(1, int(round(0.125 * (npts - 1))))
    noon = start_idx + int(round(0.5 * (npts - 1)))
    evening = noon + int(round(0.5 * (end_idx - noon)))

    morning = min(max(morning, start_idx), end_idx)
    noon = min(max(noon, start_idx), end_idx)
    evening = min(max(evening, start_idx), end_idx)

    return [morning, noon, evening]


def get_time_hours_from_output_time(time_in, time_out):
    idx = np.rint(np.asarray(time_out)).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= len(time_in)):
        raise ValueError("Output time indices are out of bounds for input time.")
    return np.asarray(time_in, dtype=np.float64)[idx]


def get_profile_label(ds_in):
    if "xh" not in ds_in.variables:
        return "unknown"

    xh = ds_in["xh"].values.astype(np.float64)
    if xh.ndim != 1 or xh.size < 2:
        return "unknown"

    dx = xh[1] - xh[0]
    if dx < 1000.0:
        return r"{:0.0f} $m$".format(dx)
    return r"{:0.2f} $km$".format(dx / 1000.0)


def global_nanmin(local_val, comm):
    return comm.allreduce(local_val, op=MPI.MIN)


def global_nanmax(local_val, comm):
    return comm.allreduce(local_val, op=MPI.MAX)


def local_minmax(arr):
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.inf, -np.inf
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def gather_x_slices(local_slices, comm):
    rank = comm.Get_rank()
    gathered = comm.gather(local_slices, root=0)
    if rank != 0:
        return None

    nslices = len(local_slices)
    full = []
    for s in range(nslices):
        parts = [g[s] for g in gathered]
        full.append(np.concatenate(parts, axis=-1))
    return full


def get_first_dim_name(da, candidates):
    for d in candidates:
        if d in da.dims:
            return d
    raise ValueError(f"None of candidate dims {candidates} found in {da.dims}")


def compute_mixture_R_cp_from_vmr(ds_in_local, input_idx, y_mid, i0, i1, ilay_max):
    """
    Compute mixture gas constant R_mix [J kg-1 K-1] and cp_mix [J kg-1 K-1]
    from volume mixing ratios (mole fractions) using:
      Mbar   = sum_i x_i M_i / sum_i x_i
      R_mix  = R_u / Mbar
      w_i    = x_i M_i / sum_j(x_j M_j)
      cp_mix = sum_i w_i cp_i

    Returns arrays of shape (lay, x_local).
    """
    Ru = 8.314462618  # J mol-1 K-1

    # Molecular weights [kg/mol]
    M = {
        "vmr_n2":  28.0134e-3,
        "vmr_o2":  31.9988e-3,
        "vmr_h2o": 18.01528e-3,
        "vmr_co2": 44.0095e-3,
        "vmr_ch4": 16.0425e-3,
        "vmr_n2o": 44.0128e-3,
        "vmr_co":  28.0101e-3,
        "vmr_o3":  47.9982e-3,
    }

    # Approximate mass-specific cp [J/kg/K]
    cp_i = {
        "vmr_n2":  1039.0,
        "vmr_o2":   918.0,
        "vmr_h2o": 1859.0,
        "vmr_co2":  846.0,
        "vmr_ch4": 2220.0,
        "vmr_n2o":  880.0,
        "vmr_co":  1040.0,
        "vmr_o3":   920.0,
    }

    species = list(M.keys())

    x_i = {}
    for s in species:
        x_i[s] = ds_in_local[s].isel(
            time=input_idx, lay=slice(0, ilay_max), y=y_mid, x=slice(i0, i1)
        ).astype("float64").load().values

    x_sum = np.zeros_like(next(iter(x_i.values())))
    for s in species:
        x_sum += x_i[s]

    bad = (~np.isfinite(x_sum)) | (x_sum <= 0.0)

    Mbar = np.zeros_like(x_sum)
    for s in species:
        Mbar += x_i[s] * M[s]
    Mbar = np.where(bad, np.nan, Mbar / x_sum)

    R_mix = np.where(bad, np.nan, Ru / Mbar)

    mass_denom = np.zeros_like(x_sum)
    for s in species:
        mass_denom += x_i[s] * M[s]

    cp_mix = np.zeros_like(x_sum)
    for s in species:
        w_i = np.where(bad, np.nan, (x_i[s] * M[s]) / mass_denom)
        cp_mix += w_i * cp_i[s]

    cp_mix = np.where(bad, np.nan, cp_mix)

    return R_mix, cp_mix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rte_rrtmgp_cpp_input_dir_path",
        nargs=1,
        type=str,
        required=True,
        help="Directory containing RTE-RRTMGP-CPP input files (*.in.nc).",
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_output_dir_path",
        nargs=1,
        type=str,
        required=True,
        help="Directory containing RTE-RRTMGP-CPP output files (*.out.nc).",
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_viz_dir_path",
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
        "--lr",
        type=str,
        default=None,
        help="Comma-separated list of lr tags to plot, e.g. '01,04'. If omitted, plot all available lr_XX pairs.",
    )
    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Maximum height to include [km]. Default: use all heights.",
    )
    parser.add_argument(
        "--detailed-calc",
        action="store_true",
        help=(
            "If set, compute R and cp from volume mixing ratios in the input file. "
            "If not set, use dry-air constants."
        ),
    )

    args = parser.parse_args()

    input_dir = args.rte_rrtmgp_cpp_input_dir_path[0]
    output_dir = args.rte_rrtmgp_cpp_output_dir_path[0]
    viz_dir = args.rte_rrtmgp_cpp_viz_dir_path[0]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Dry-air defaults
    Rd_dry = 287.05     # J kg-1 K-1
    cp_dry = 1004.0     # J kg-1 K-1
    sec_per_hour = 3600.0

    if rank == 0:
        os.makedirs(viz_dir, exist_ok=True)

    pairs = find_pairs(input_dir, output_dir)
    if rank == 0:
        log(f"Found {len(pairs)} lr_XX pairs", comm)

    if not pairs:
        if rank == 0:
            log("ERROR: No matching lr_XX pairs found.", comm)
        sys.exit(1)

    if args.lr is not None:
        requested = [s.strip() for s in args.lr.split(",") if s.strip()]
        requested_tags = [s if s.startswith("lr_") else f"lr_{s}" for s in requested]
        pairs = {lr: pairs[lr] for lr in requested_tags if lr in pairs}
        if rank == 0:
            log(f"After --lr filtering, {len(pairs)} pairs remain: {list(pairs.keys())}", comm)

        if not pairs:
            if rank == 0:
                log("ERROR: --lr requested resolutions not found in available pairs.", comm)
            sys.exit(1)

    for lr_tag, (in_path, out_path) in pairs.items():
        t0 = MPI.Wtime()
        if rank == 0:
            calc_mode = "detailed VMR-based cp/R" if args.detailed_calc else "dry-air cp/R"
            log(f"Processing {lr_tag} using {calc_mode}", comm)

        with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in:
            if rank == 0:
                profile_label = get_profile_label(ds_in)
                time_in = ds_in["time"].astype("float64").load().values
                x = ds_in["x"].astype("float64").load().values
                mu0 = ds_in["mu0"].astype("float64").load().values
            else:
                profile_label = None
                time_in = None
                x = None
                mu0 = None

        profile_label = comm.bcast(profile_label, root=0)
        time_in = comm.bcast(time_in, root=0)
        x = comm.bcast(x, root=0)
        mu0 = comm.bcast(mu0, root=0)

        with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out:
            if rank == 0:
                time_out = ds_out["time"].astype("float64").load().values
                time_hours = get_time_hours_from_output_time(time_in, time_out)

                ny = int(ds_out.sizes["y"])
                nx = int(ds_out.sizes["x"])
                y_mid = ny // 2

                z_out = ds_out["z"].astype("float64").load().values
                ilay_max_avail = len(z_out)

                sw_lev_dim_name = get_first_dim_name(ds_out["sw_flux_up"], ("lev", "z", "z_lev"))
                ilev_max_avail = int(ds_out["sw_flux_up"].sizes[sw_lev_dim_name])

                if args.zmax is None:
                    ilay_max = ilay_max_avail
                else:
                    zmax_m = 1000.0 * args.zmax
                    ilay_max = int(np.searchsorted(z_out, zmax_m, side="right"))
                    ilay_max = max(1, min(ilay_max, ilay_max_avail))

                ilev_max = min(ilev_max_avail, ilay_max + 1)

                mu0_use = mu0[np.rint(time_out).astype(np.int64), :, :]
                mu0_1d = mu0_use[:, y_mid, 0]

                day_segments = find_day_night_transition_indices(mu0_1d, night_eps=args.night_eps)
                chosen_days = choose_representative_day_segments(day_segments)

                day_choices = []
                for start_idx, end_idx in chosen_days:
                    day_choices.extend(choose_times_within_day(start_idx, end_idx))

                unique_choices = sorted(set(day_choices))
                if len(unique_choices) < 3:
                    raise ValueError(f"{lr_tag}: could not determine three distinct daytime samples.")

                if len(unique_choices) > 3:
                    first_day_start, first_day_end = chosen_days[0]
                    unique_choices = choose_times_within_day(first_day_start, first_day_end)

                t_morning, t_noon, t_evening = unique_choices
                t_indices = [t_morning, t_noon, t_evening]

                sza_deg = np.degrees(np.arccos(np.clip(mu0_1d[t_indices], -1.0, 1.0)))
                t_titles = [
                    rf"{time_hours[t_morning]:.2f} Hours - Solar Zenith Angle {sza_deg[0]:.1f}$^{{\circ}}$",
                    rf"{time_hours[t_noon]:.2f} Hours - Solar Zenith Angle {sza_deg[1]:.1f}$^{{\circ}}$",
                    rf"{time_hours[t_evening]:.2f} Hours - Solar Zenith Angle {sza_deg[2]:.1f}$^{{\circ}}$",
                ]
            else:
                time_out = None
                nx = None
                y_mid = None
                t_indices = None
                t_titles = None
                ilay_max = None
                ilev_max = None
                z_out = None

            time_out = comm.bcast(time_out, root=0)
            nx = comm.bcast(nx, root=0)
            y_mid = comm.bcast(y_mid, root=0)
            t_indices = comm.bcast(t_indices, root=0)
            t_titles = comm.bcast(t_titles, root=0)
            ilay_max = comm.bcast(ilay_max, root=0)
            ilev_max = comm.bcast(ilev_max, root=0)
            z_out = comm.bcast(z_out, root=0)

            i0, i1 = decompose_1d(nx, comm.Get_size(), rank)
            x_loc = x[i0:i1]

            with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in_local:
                lwp_local = ds_in_local["lwp"].isel(y=y_mid, x=slice(i0, i1)).astype("float64")
                iwp_local = ds_in_local["iwp"].isel(y=y_mid, x=slice(i0, i1)).astype("float64")
                t_lay_local = ds_in_local["t_lay"].isel(y=y_mid, x=slice(i0, i1), lay=slice(0, ilay_max)).astype("float64")
                p_lay_local = ds_in_local["p_lay"].isel(y=y_mid, x=slice(i0, i1), lay=slice(0, ilay_max)).astype("float64")

                lwp_iwp_slices_local = []
                ray_hr_slices_local = []
                ts_hr_slices_local = []
                diff_hr_slices_local = []

                rt_abs_dif_da = ds_out["rt_flux_abs_dif"]
                rt_abs_dir_da = ds_out["rt_flux_abs_dir"]
                sw_up_da = ds_out["sw_flux_up"]
                sw_dn_da = ds_out["sw_flux_dn"]

                rt_abs_dim = get_first_dim_name(rt_abs_dif_da, ("lay", "z"))
                sw_lev_dim = get_first_dim_name(sw_up_da, ("lev", "z", "z_lev"))

                rt_abs_dif = rt_abs_dif_da.isel({rt_abs_dim: slice(0, ilay_max), "y": y_mid, "x": slice(i0, i1)}).astype("float64")
                rt_abs_dir = rt_abs_dir_da.isel({rt_abs_dim: slice(0, ilay_max), "y": y_mid, "x": slice(i0, i1)}).astype("float64")
                sw_up = sw_up_da.isel({sw_lev_dim: slice(0, ilev_max), "y": y_mid, "x": slice(i0, i1)}).astype("float64")
                sw_dn = sw_dn_da.isel({sw_lev_dim: slice(0, ilev_max), "y": y_mid, "x": slice(i0, i1)}).astype("float64")

                # Obtain z and Δz from the output file directly
                z_lay_use = z_out[:ilay_max]

                z_ifc = np.empty(ilay_max + 1, dtype=np.float64)
                z_ifc[1:-1] = 0.5 * (z_lay_use[:-1] + z_lay_use[1:])
                z_ifc[0] = z_lay_use[0] - 0.5 * (z_lay_use[1] - z_lay_use[0])
                z_ifc[-1] = z_lay_use[-1] + 0.5 * (z_lay_use[-1] - z_lay_use[-2])
                dz = np.diff(z_ifc)

                ray_abs_slices_local = []
                ts_abs_slices_local = []
                diff_abs_slices_local = []

                for t_idx in t_indices:
                    input_idx = int(np.rint(time_out[t_idx]))

                    cloud_slice_local = (
                        lwp_local.isel(time=input_idx, lay=slice(0, ilay_max)).load().values
                        + iwp_local.isel(time=input_idx, lay=slice(0, ilay_max)).load().values
                    )
                    lwp_iwp_slices_local.append(cloud_slice_local)

                    T_local = t_lay_local.isel(time=input_idx).load().values
                    p_lay_now = p_lay_local.isel(time=input_idx).load().values

                    if args.detailed_calc:
                        R_mix_local, cp_mix_local = compute_mixture_R_cp_from_vmr(
                            ds_in_local=ds_in_local,
                            input_idx=input_idx,
                            y_mid=y_mid,
                            i0=i0,
                            i1=i1,
                            ilay_max=ilay_max,
                        )
                    else:
                        R_mix_local = np.full_like(T_local, Rd_dry, dtype=np.float64)
                        cp_mix_local = np.full_like(T_local, cp_dry, dtype=np.float64)

                    # Ray-tracer: volumetric absorption -> heating rate
                    Qrt_local = (
                        rt_abs_dif.isel(time=t_idx).load().values
                        + rt_abs_dir.isel(time=t_idx).load().values
                    )
                    ray_abs_slices_local.append(Qrt_local)

                    ray_hr_local = 86400 * Qrt_local * R_mix_local * T_local / (p_lay_now * cp_mix_local) # Convert to per-day
                    ray_hr_slices_local.append(ray_hr_local)

                    # Two-stream absorbed flux: match legacy script formula, but use Δz from output z
                    sw_up_slice_local = sw_up.isel(time=t_idx).load().values
                    sw_dn_slice_local = sw_dn.isel(time=t_idx).load().values

                    ts_flux_diff_local = (
                        (sw_up_slice_local[:-1, :] - sw_dn_slice_local[:-1, :]) +
                        (sw_dn_slice_local[1:, :] - sw_up_slice_local[1:, :])
                    )
                    Qts_local = ts_flux_diff_local / dz[:, None]
                    ts_abs_slices_local.append(Qts_local)

                    ts_hr_local = 86400 * Qts_local * R_mix_local * T_local / (p_lay_now * cp_mix_local) # Convert to per-day
                    ts_hr_slices_local.append(ts_hr_local)

                    diff_hr_slices_local.append(ray_hr_local - ts_hr_local)
                    diff_abs_slices_local.append(Qrt_local - Qts_local)

        local_cloud_min = min(local_minmax(a)[0] for a in lwp_iwp_slices_local)
        local_cloud_max = max(local_minmax(a)[1] for a in lwp_iwp_slices_local)
        lwp_iwp_min = global_nanmin(local_cloud_min, comm)
        lwp_iwp_max = global_nanmax(local_cloud_max, comm)

        local_hr_min = min(
            min(local_minmax(a)[0] for a in ray_hr_slices_local),
            min(local_minmax(a)[0] for a in ts_hr_slices_local),
        )
        local_hr_max = max(
            max(local_minmax(a)[1] for a in ray_hr_slices_local),
            max(local_minmax(a)[1] for a in ts_hr_slices_local),
        )
        hr_min = global_nanmin(local_hr_min, comm)
        hr_max = global_nanmax(local_hr_max, comm)

        hr_min = min(0.0, hr_min)

        local_diff_hr_max = max(np.nanmax(np.abs(a)) if np.any(np.isfinite(a)) else 0.0 for a in diff_hr_slices_local)
        diff_hr_max = global_nanmax(local_diff_hr_max, comm)

        all_x_parts = comm.gather(x_loc, root=0)
        lwp_iwp_slices = gather_x_slices(lwp_iwp_slices_local, comm)
        ray_hr_slices = gather_x_slices(ray_hr_slices_local, comm)
        ts_hr_slices = gather_x_slices(ts_hr_slices_local, comm)
        diff_hr_slices = gather_x_slices(diff_hr_slices_local, comm)

        if rank == 0:
            x_full = np.concatenate(all_x_parts)
            diff_norm = TwoSlopeNorm(vmin=-diff_hr_max, vcenter=0.0, vmax=diff_hr_max)

            fig, axes = plt.subplots(
                4, 3,
                figsize=(14, 14),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            for j in range(3):
                axes[0, j].set_title(t_titles[j])

            pcm_cloud = None
            pcm_hr = None
            pcm_diff = None

            for j in range(3):
                pcm_cloud = axes[0, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, lwp_iwp_slices[j],
                    shading="auto",
                    cmap="Blues",
                    vmin=lwp_iwp_min,
                    vmax=lwp_iwp_max,
                )

                heating_cmap = "hot"
                flux_cmap = "magma"
                
                pcm_hr = axes[1, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, ray_hr_slices[j],
                    shading="auto",
                    cmap=heating_cmap,
                    vmin=hr_min,
                    vmax=hr_max,
                )

                axes[2, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, ts_hr_slices[j],
                    shading="auto",
                    cmap=heating_cmap,
                    vmin=hr_min,
                    vmax=hr_max,
                )

                pcm_diff = axes[3, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, diff_hr_slices[j],
                    shading="auto",
                    cmap="RdBu_r",
                    norm=diff_norm,
                )

            axes[1, 0].set_ylabel("Ray-Tracer")
            axes[2, 0].set_ylabel("Two-Stream")
            axes[3, 0].set_ylabel("Ray-Tracer - Two-Stream")

            fig.supxlabel(r"$x \; [km]$")
            fig.supylabel(r"$z \; [km]$")

            cbar_cloud = fig.colorbar(
                pcm_cloud, ax=axes[0, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            cbar_cloud.set_label(r"Liquid + Ice Water Path $\left[kg\,m^{-2}\right]$")

            cbar_hr = fig.colorbar(
                pcm_hr, ax=axes[1:3, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            cbar_hr.set_label(r"Heating Rate $\left[K\,d^{-1}\right]$")

            cbar_diff = fig.colorbar(
                pcm_diff, ax=axes[3, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            cbar_diff.set_label(r"Heating Rate Difference $\left[K\,d^{-1}\right]$")

            calc_label = "VMR-based $c_p$, $R$" if args.detailed_calc else "Dry-air $c_p$, $R_d$"
            fig.suptitle(f"Horizontal Resolution - {profile_label}", fontsize=14)

            out_png = os.path.join(viz_dir, f"{lr_tag}_atm_heating.png")
            fig.savefig(out_png, dpi=200)
            plt.close(fig)

            log(f"Wrote: {out_png}", comm)

        comm.Barrier()
        if rank == 0:
            log(f"Finished {lr_tag} in {MPI.Wtime() - t0:.2f} s", comm)

    if rank == 0:
        log("All resolutions complete.", comm)


if __name__ == "__main__":
    main()