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

    args = parser.parse_args()

    input_dir = args.rte_rrtmgp_cpp_input_dir_path[0]
    output_dir = args.rte_rrtmgp_cpp_output_dir_path[0]
    viz_dir = args.rte_rrtmgp_cpp_viz_dir_path[0]

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
            if rank == 0:
                profile_label = get_profile_label(ds_in)
                time_in = ds_in["time"].astype("float64").load().values
                x = ds_in["x"].astype("float64").load().values
                z_lay = ds_in["z_lay"].astype("float64").load().values
                z_lev = ds_in["z_lev"].astype("float64").load().values
                mu0 = ds_in["mu0"].astype("float64").load().values

                if args.zmax is None:
                    ilay_max = len(z_lay)
                    ilev_max = len(z_lev)
                else:
                    zmax_m = 1000.0 * args.zmax
                    ilay_max = int(np.searchsorted(z_lay, zmax_m, side="right"))
                    ilev_max = int(np.searchsorted(z_lev, zmax_m, side="right"))

                    ilay_max = max(1, min(ilay_max, len(z_lay)))
                    ilev_max = max(2, min(ilev_max, len(z_lev)))
            else:
                profile_label = None
                time_in = None
                x = None
                z_lay = None
                z_lev = None
                mu0 = None
                ilay_max = None
                ilev_max = None

        profile_label = comm.bcast(profile_label, root=0)
        time_in = comm.bcast(time_in, root=0)
        x = comm.bcast(x, root=0)
        z_lay = comm.bcast(z_lay, root=0)
        z_lev = comm.bcast(z_lev, root=0)
        mu0 = comm.bcast(mu0, root=0)
        ilay_max = comm.bcast(ilay_max, root=0)
        ilev_max = comm.bcast(ilev_max, root=0)

        with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out:
            if rank == 0:
                time_out = ds_out["time"].astype("float64").load().values
                time_hours = get_time_hours_from_output_time(time_in, time_out)

                ny = int(ds_out.sizes["y"])
                nx = int(ds_out.sizes["x"])
                y_mid = ny // 2

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

            time_out = comm.bcast(time_out, root=0)
            nx = comm.bcast(nx, root=0)
            y_mid = comm.bcast(y_mid, root=0)
            t_indices = comm.bcast(t_indices, root=0)
            t_titles = comm.bcast(t_titles, root=0)

            i0, i1 = decompose_1d(nx, comm.Get_size(), rank)
            x_loc = x[i0:i1]

            with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in_local:
                lwp_local = ds_in_local["lwp"].isel(y=y_mid, x=slice(i0, i1)).astype("float64")
                iwp_local = ds_in_local["iwp"].isel(y=y_mid, x=slice(i0, i1)).astype("float64")

                lwp_iwp_slices_local = []
                for t_idx in t_indices:
                    input_idx = int(np.rint(time_out[t_idx]))
                    cloud_slice_local = (
                        lwp_local.isel(time=input_idx, lay=slice(0, ilay_max)).load().values
                        + iwp_local.isel(time=input_idx, lay=slice(0, ilay_max)).load().values
                    )
                    lwp_iwp_slices_local.append(cloud_slice_local)

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

            z_lay_use = z_lay[:ilay_max]
            z_lev_use = z_lev[:ilev_max]
            dz = np.diff(z_lev_use)

            ray_abs_slices_local = []
            ts_abs_slices_local = []
            diff_abs_slices_local = []

            for t_idx in t_indices:
                ray_abs_slice_local = (
                    rt_abs_dif.isel(time=t_idx).load().values
                    + rt_abs_dir.isel(time=t_idx).load().values
                )
                ray_abs_slices_local.append(ray_abs_slice_local)

                sw_up_slice_local = sw_up.isel(time=t_idx).load().values
                sw_dn_slice_local = sw_dn.isel(time=t_idx).load().values

                ts_flux_diff_local = (
                    (sw_up_slice_local[:-1, :] - sw_dn_slice_local[:-1, :]) +
                    (sw_dn_slice_local[1:, :] - sw_up_slice_local[1:, :])
                )
                ts_abs_slice_local = ts_flux_diff_local / dz[:, None]
                ts_abs_slices_local.append(ts_abs_slice_local)

                diff_abs_slices_local.append(ray_abs_slice_local - ts_abs_slice_local)

        local_cloud_min = min(local_minmax(a)[0] for a in lwp_iwp_slices_local)
        local_cloud_max = max(local_minmax(a)[1] for a in lwp_iwp_slices_local)
        lwp_iwp_min = global_nanmin(local_cloud_min, comm)
        lwp_iwp_max = global_nanmax(local_cloud_max, comm)

        local_abs_min = min(
            min(local_minmax(a)[0] for a in ray_abs_slices_local),
            min(local_minmax(a)[0] for a in ts_abs_slices_local),
        )
        local_abs_max = max(
            max(local_minmax(a)[1] for a in ray_abs_slices_local),
            max(local_minmax(a)[1] for a in ts_abs_slices_local),
        )
        abs_min = global_nanmin(local_abs_min, comm)
        abs_max = global_nanmax(local_abs_max, comm)

        local_diff_abs_max = max(np.nanmax(np.abs(a)) if np.any(np.isfinite(a)) else 0.0 for a in diff_abs_slices_local)
        diff_abs_max = global_nanmax(local_diff_abs_max, comm)

        all_x_parts = comm.gather(x_loc, root=0)
        lwp_iwp_slices = gather_x_slices(lwp_iwp_slices_local, comm)
        ray_abs_slices = gather_x_slices(ray_abs_slices_local, comm)
        ts_abs_slices = gather_x_slices(ts_abs_slices_local, comm)
        diff_abs_slices = gather_x_slices(diff_abs_slices_local, comm)

        if rank == 0:
            x_full = np.concatenate(all_x_parts)
            diff_norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0.0, vmax=diff_abs_max)

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
            pcm_abs = None
            pcm_diff = None

            for j in range(3):
                pcm_cloud = axes[0, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, lwp_iwp_slices[j],
                    shading="auto",
                    cmap="Blues",
                    vmin=lwp_iwp_min,
                    vmax=lwp_iwp_max,
                )

                pcm_abs = axes[1, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, ray_abs_slices[j],
                    shading="auto",
                    cmap="magma",
                    vmin=abs_min,
                    vmax=abs_max,
                )

                axes[2, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, ts_abs_slices[j],
                    shading="auto",
                    cmap="magma",
                    vmin=abs_min,
                    vmax=abs_max,
                )

                pcm_diff = axes[3, j].pcolormesh(
                    x_full / 1000.0, z_lay_use / 1000.0, diff_abs_slices[j],
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

            cbar_abs = fig.colorbar(
                pcm_abs, ax=axes[1:3, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            cbar_abs.set_label(r"Absorbed Shortwave Flux $\left[W\,m^{-3}\right]$")

            cbar_diff = fig.colorbar(
                pcm_diff, ax=axes[3, :], location="right", shrink=0.95, fraction=0.046, pad=0.04
            )
            cbar_diff.set_label(r"Absorbed Shortwave Flux $\left[W\,m^{-3}\right]$")

            fig.suptitle(f"Horizontal Resolution - {profile_label}", fontsize=14)

            out_png = os.path.join(viz_dir, f"{lr_tag}_flux_abs.png")
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