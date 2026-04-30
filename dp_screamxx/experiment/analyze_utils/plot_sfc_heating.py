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
    segments = []
    n = len(mu0_1d)
    if n == 0:
        return segments

    is_day = np.isfinite(mu0_1d) & (mu0_1d > night_eps)

    start = 0 if is_day[0] else None
    for i in range(1, n):
        if (not is_day[i - 1]) and is_day[i]:
            start = i
        elif is_day[i - 1] and (not is_day[i]) and start is not None:
            segments.append((start, i - 1))
            start = None

    if start is not None:
        segments.append((start, n - 1))

    return segments


def choose_times_within_day(start_idx, end_idx):
    if end_idx < start_idx:
        raise ValueError(f"Invalid daytime interval: {start_idx}, {end_idx}")

    npts = end_idx - start_idx + 1
    if npts == 1:
        return [start_idx, start_idx, start_idx]

    morning = start_idx + max(1, int(round(0.125 * (npts - 1))))
    noon = start_idx + int(round(0.5 * (npts - 1)))
    evening = noon + int(round(0.5 * (end_idx - noon)))

    return [
        min(max(morning, start_idx), end_idx),
        min(max(noon, start_idx), end_idx),
        min(max(evening, start_idx), end_idx),
    ]


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
    return r"{:0.0f} $m$".format(dx) if dx < 1000.0 else r"{:0.2f} $km$".format(dx / 1000.0)


def global_nanmin(local_val, comm):
    return comm.allreduce(local_val, op=MPI.MIN)


def global_nanmax(local_val, comm):
    return comm.allreduce(local_val, op=MPI.MAX)


def local_minmax(arr):
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.inf, -np.inf
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def gather_field(local_field, comm):
    gathered = comm.gather(local_field, root=0)
    if comm.Get_rank() != 0:
        return None
    return np.concatenate(gathered, axis=-1)


def choose_three_times(mu0_1d, night_eps):
    day_segments = find_day_night_transition_indices(mu0_1d, night_eps=night_eps)
    if not day_segments:
        raise ValueError("No daytime segments found.")

    day_choices = []
    for seg in day_segments:
        day_choices.extend(choose_times_within_day(*seg))

    unique_choices = sorted(set(day_choices))
    if len(unique_choices) < 3:
        raise ValueError("Could not determine three distinct daytime samples.")

    if len(unique_choices) > 3:
        unique_choices = choose_times_within_day(*day_segments[0])

    return unique_choices


def compute_grid_spacing(ds_in, ds_out):
    dx = dy = None

    if "xh" in ds_in.variables:
        xh = ds_in["xh"].values.astype(np.float64)
        if xh.ndim == 1 and xh.size >= 2:
            dx = float(np.nanmean(np.diff(xh)))

    if "yh" in ds_in.variables:
        yh = ds_in["yh"].values.astype(np.float64)
        if yh.ndim == 1 and yh.size >= 2:
            dy = float(np.nanmean(np.diff(yh)))

    if dx is None and "x" in ds_in.variables:
        x = ds_in["x"].values.astype(np.float64)
        if x.ndim == 1 and x.size >= 2:
            dx = float(np.nanmean(np.diff(x)))

    if dy is None and "y" in ds_in.variables:
        y = ds_in["y"].values.astype(np.float64)
        if y.ndim == 1 and y.size >= 2:
            dy = float(np.nanmean(np.diff(y)))

    if dx is None and "x" in ds_out.variables:
        x = ds_out["x"].values.astype(np.float64)
        if x.ndim == 1 and x.size >= 2:
            dx = float(np.nanmean(np.diff(x)))

    if dy is None and "y" in ds_out.variables:
        y = ds_out["y"].values.astype(np.float64)
        if y.ndim == 1 and y.size >= 2:
            dy = float(np.nanmean(np.diff(y)))

    if dx is None or dy is None:
        raise ValueError("Could not determine dx and/or dy from input/output files.")

    return dx, dy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rte_rrtmgp_cpp_input_dir_path", nargs=1, type=str, required=True)
    parser.add_argument("--rte_rrtmgp_cpp_output_dir_path", nargs=1, type=str, required=True)
    parser.add_argument("--rte_rrtmgp_cpp_viz_dir_path", nargs=1, type=str, required=True)
    parser.add_argument("--night-eps", type=float, default=1.0e-3)
    parser.add_argument("--lr", type=str, default=None)
    parser.add_argument("--zmax", type=float, default=None, help="Unused in this XY surface-flux script.")
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional case selector. If --case GATEIII, plot surface heating rate instead of net surface shortwave flux.",
    )

    args = parser.parse_args()

    input_dir = args.rte_rrtmgp_cpp_input_dir_path[0]
    output_dir = args.rte_rrtmgp_cpp_output_dir_path[0]
    viz_dir = args.rte_rrtmgp_cpp_viz_dir_path[0]

    case_name = None if args.case is None else args.case.strip().upper()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
            log(f"Processing {lr_tag}", comm)

        with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in:
            with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out_for_grid:
                if rank == 0:
                    profile_label = get_profile_label(ds_in)
                    time_in = ds_in["time"].values.astype(np.float64)
                    x = ds_in["x"].values.astype(np.float64)
                    y = ds_in["y"].values.astype(np.float64)
                    mu0 = ds_in["mu0"].values.astype(np.float64)
                    dx, dy = compute_grid_spacing(ds_in, ds_out_for_grid)
                else:
                    profile_label = time_in = x = y = mu0 = dx = dy = None

        profile_label = comm.bcast(profile_label, root=0)
        time_in = comm.bcast(time_in, root=0)
        x = comm.bcast(x, root=0)
        y = comm.bcast(y, root=0)
        mu0 = comm.bcast(mu0, root=0)
        dx = comm.bcast(dx, root=0)
        dy = comm.bcast(dy, root=0)

        #
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Adjustable constants for GATEIII surface heating rate calculation
        # Units should be:
        #   rho_w : kg m^-3
        #   c_pw  : J kg^-1 K^-1
        #   h_m   : m
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #
        rho_w = None
        c_pw = None
        h_m = None
        if case_name == "GATEIII":
            rho_w = 1.027e3 # Reference density of sea water (doi:10.1017/9781107588417)
            c_pw  = 3986    # Reference specific heat capacity at constant pressure for sea water (doi:10.1017/9781107588417)
            h_m   = 19.753  # Average mixing layer depth in August in observation region (doi:10.17882/91774)

        with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out:
            time_out = ds_out["time"].values.astype(np.float64)
            ny = int(ds_out.sizes["y"])
            nx = int(ds_out.sizes["x"])

            time_hours = get_time_hours_from_output_time(time_in, time_out)
            time_idx = np.rint(time_out).astype(np.int64)

            if rank == 0:
                y_mid = ny // 2
                mu0_1d = mu0[time_idx, y_mid, 0]
                t_indices = choose_three_times(mu0_1d, args.night_eps)
                sza_deg = np.degrees(np.arccos(np.clip(mu0_1d[t_indices], -1.0, 1.0)))
                t_titles = [
                    rf"{time_hours[t_indices[0]]:.2f} Hours - Solar Zenith Angle {sza_deg[0]:.1f}$^{{\circ}}$",
                    rf"{time_hours[t_indices[1]]:.2f} Hours - Solar Zenith Angle {sza_deg[1]:.1f}$^{{\circ}}$",
                    rf"{time_hours[t_indices[2]]:.2f} Hours - Solar Zenith Angle {sza_deg[2]:.1f}$^{{\circ}}$",
                ]
            else:
                t_indices = t_titles = None

            t_indices = comm.bcast(t_indices, root=0)
            t_titles = comm.bcast(t_titles, root=0)

            i0, i1 = decompose_1d(nx, comm.Get_size(), rank)

            with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in_local:
                lwp = ds_in_local["lwp"].isel(x=slice(i0, i1)).astype("float64").load().values
                iwp = ds_in_local["iwp"].isel(x=slice(i0, i1)).astype("float64").load().values

            #
            # Net surface shortwave flux:
            # Qnr = downwelling - upwelling
            #
            rt_sfc_dn = (
                ds_out["rt_flux_sfc_dir"].isel(x=slice(i0, i1)).astype("float64").load().values
                + ds_out["rt_flux_sfc_dif"].isel(x=slice(i0, i1)).astype("float64").load().values
            )
            rt_sfc_up = ds_out["rt_flux_sfc_up"].isel(x=slice(i0, i1)).astype("float64").load().values
            rt_sfc_net = rt_sfc_dn - rt_sfc_up

            #
            # Two-stream net surface shortwave flux:
            # use lev=0 for surface, consistent with original script
            #
            ts_sfc_dn = ds_out["sw_flux_dn"].isel(lev=0, x=slice(i0, i1)).astype("float64").load().values
            ts_sfc_up = ds_out["sw_flux_up"].isel(lev=0, x=slice(i0, i1)).astype("float64").load().values
            ts_sfc_net = ts_sfc_dn - ts_sfc_up

            cloud_local = []
            ray_local = []
            ts_local = []
            diff_local = []

            for t_idx in t_indices:
                input_idx = int(time_idx[t_idx])

                cloud_xy = np.sum(lwp[input_idx] + iwp[input_idx], axis=0)
                cloud_local.append(cloud_xy)

                ray_xy = rt_sfc_net[t_idx]
                ts_xy = ts_sfc_net[t_idx]

                if case_name == "GATEIII":
                    #
                    # Requested formula:
                    # heating_rate = Qnr / (rho_w * c_pw * h_m)
                    #
                    denom = rho_w * c_pw * h_m 
                    ray_xy = (ray_xy / denom) * 86400 # convert to per-day
                    ts_xy = (ts_xy / denom) * 86400 # convert to per-day

                ray_local.append(ray_xy)
                ts_local.append(ts_xy)
                diff_local.append(ray_xy - ts_xy)

        cloud_min = global_nanmin(min(local_minmax(a)[0] for a in cloud_local), comm)
        cloud_max = global_nanmax(max(local_minmax(a)[1] for a in cloud_local), comm)

        field_min = global_nanmin(
            min(
                min(local_minmax(a)[0] for a in ray_local),
                min(local_minmax(a)[0] for a in ts_local),
            ),
            comm,
        )
        field_max = global_nanmax(
            max(
                max(local_minmax(a)[1] for a in ray_local),
                max(local_minmax(a)[1] for a in ts_local),
            ),
            comm,
        )

        diff_max = global_nanmax(
            max(np.nanmax(np.abs(a)) if np.any(np.isfinite(a)) else 0.0 for a in diff_local),
            comm,
        )

        cloud = [gather_field(a, comm) for a in cloud_local]
        ray = [gather_field(a, comm) for a in ray_local]
        ts = [gather_field(a, comm) for a in ts_local]
        diff = [gather_field(a, comm) for a in diff_local]

        if rank == 0:
            x_km = x / 1000.0
            y_km = y / 1000.0
            X, Y = np.meshgrid(x_km, y_km)
            diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0.0, vmax=diff_max)

            fig, axes = plt.subplots(
                4, 3,
                figsize=(14, 14),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )

            for j in range(3):
                axes[0, j].set_title(t_titles[j])

            pcm_cloud = pcm_field = pcm_diff = None

            for j in range(3):
                pcm_cloud = axes[0, j].pcolormesh(
                    X, Y, cloud[j],
                    shading="auto",
                    cmap="Blues",
                    vmin=cloud_min,
                    vmax=cloud_max,
                )

                heating_cmap = "hot"
                flux_cmap = "magma"

                if case_name == "GATEIII":
                    cmap = heating_cmap
                else:
                    cmap = flux_cmap

                pcm_field = axes[1, j].pcolormesh(
                    X, Y, ray[j],
                    shading="auto",
                    cmap=cmap,
                    vmin=field_min,
                    vmax=field_max,
                )

                axes[2, j].pcolormesh(
                    X, Y, ts[j],
                    shading="auto",
                    cmap=cmap,
                    vmin=field_min,
                    vmax=field_max,
                )

                pcm_diff = axes[3, j].pcolormesh(
                    X, Y, diff[j],
                    shading="auto",
                    cmap="RdBu_r",
                    norm=diff_norm,
                )

            axes[1, 0].set_ylabel("Ray-Tracer")
            axes[2, 0].set_ylabel("Two-Stream")
            axes[3, 0].set_ylabel("Ray-Tracer - Two-Stream")

            fig.supxlabel(r"$x \; [km]$")
            fig.supylabel(r"$y \; [km]$")

            cbar_cloud = fig.colorbar(
                pcm_cloud, ax=axes[0, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            cbar_cloud.set_label(r"Vertically-Integrated Liquid + Ice Water Path $\left[kg\,m^{-1}\right]$")

            cbar_field = fig.colorbar(
                pcm_field, ax=axes[1:3, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            if case_name == "GATEIII":
                cbar_field.set_label(r"Surface Heating Rate $\left[K\,d^{-1}\right]$")
            else:
                cbar_field.set_label(r"Net Surface Shortwave Flux $\left[W\,m^{-2}\right]$")

            cbar_diff = fig.colorbar(
                pcm_diff, ax=axes[3, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            if case_name == "GATEIII":
                cbar_diff.set_label(r"Surface Heating Rate $\left[K\,d^{-1}\right]$")
            else:
                cbar_diff.set_label(r"Net Surface Shortwave Flux $\left[W\,m^{-2}\right]$")

            if case_name == "GATEIII":
                fig.suptitle(
                    f"Horizontal Resolution - {profile_label}",
                    fontsize=12,
                )
                out_png = os.path.join(viz_dir, f"{lr_tag}_sfc_heating.png")
            else:
                fig.suptitle(f"Horizontal Resolution - {profile_label} | Net Surface Shortwave Flux", fontsize=14)
                out_png = os.path.join(viz_dir, f"{lr_tag}_net_sfc_sw_flux.png")

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