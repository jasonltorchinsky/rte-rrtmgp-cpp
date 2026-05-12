#!/usr/bin/env python3
import os
import re
import glob
import argparse
import sys
import time as pytime

import numpy as np
import xarray as xr
from mpi4py import MPI

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from calc_atm_heating import calc_atm_heating
from consts import rho_w, cp_sw, h_m, sec_per_day


def log(msg, comm=None, root_only=True):
    ts = pytime.strftime("%Y-%m-%d %H:%M:%S")
    if comm is None:
        print(f"[{ts}] {msg}", flush=True)
        return
    rank = comm.Get_rank()
    if (not root_only) or rank == 0:
        print(f"[{ts}] [rank {rank}] {msg}", flush=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {v!r}")


def decompose_1d(n, size, rank):
    counts = np.full(size, n // size, dtype=np.int64)
    counts[: n % size] += 1
    starts = np.zeros(size, dtype=np.int64)
    starts[1:] = np.cumsum(counts[:-1])
    i0 = int(starts[rank])
    i1 = int(i0 + counts[rank])
    return i0, i1


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


def time_edges_from_centers(t):
    t = np.asarray(t, dtype=np.float64)
    nt = t.size
    edges = np.empty(nt + 1, dtype=np.float64)
    if nt == 1:
        edges[0] = t[0] - 0.5
        edges[1] = t[0] + 0.5
        return edges
    edges[1:-1] = 0.5 * (t[1:] + t[:-1])
    edges[0] = t[0] - (edges[1] - t[0])
    edges[-1] = t[-1] + (t[-1] - edges[-2])
    return edges


def finite_minmax(a):
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.inf, -np.inf
    x = a[finite]
    return float(np.min(x)), float(np.max(x))


def reduce_minmax(comm, a):
    local_min, local_max = finite_minmax(a)
    gmin = comm.allreduce(local_min, op=MPI.MIN)
    gmax = comm.allreduce(local_max, op=MPI.MAX)
    return gmin, gmax


def find_day_segments(mu0_1d, night_eps=1.0e-3):
    is_day = np.isfinite(mu0_1d) & (mu0_1d > night_eps)
    if is_day.size == 0:
        return []
    changes = np.diff(is_day.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if is_day[0]:
        starts = np.r_[0, starts]
    if is_day[-1]:
        ends = np.r_[ends, is_day.size]
    return list(zip(starts.tolist(), ends.tolist()))


def get_first_dim_name(da, candidates):
    for d in candidates:
        if d in da.dims:
            return d
    raise ValueError(f"None of candidate dims {candidates} found in {da.dims}")


def build_histograms_fast(x_2d, bins):
    nt, _ = x_2d.shape
    nbins = bins.size - 1
    xmin = float(bins[0])
    xmax = float(bins[-1])

    finite = np.isfinite(x_2d)
    if not np.any(finite):
        return np.zeros((nt, nbins), dtype=np.int64)

    t_idx, _ = np.nonzero(finite)
    vals = np.clip(x_2d[finite], xmin, xmax)

    bin_width = (xmax - xmin) / nbins
    if bin_width <= 0.0 or not np.isfinite(bin_width):
        raise ValueError("Invalid histogram bin width.")

    b_idx = np.floor((vals - xmin) / bin_width).astype(np.int64)
    b_idx = np.clip(b_idx, 0, nbins - 1)

    flat_idx = t_idx * nbins + b_idx
    return np.bincount(flat_idx, minlength=nt * nbins).reshape(nt, nbins).astype(np.int64, copy=False)


def reduce_histogram_to_root(comm, H_local, bin_min=1):
    rank = comm.Get_rank()
    H = np.empty_like(H_local) if rank == 0 else None
    comm.Reduce(H_local, H, op=MPI.SUM, root=0)
    if rank != 0:
        return None
    H_plot = H.T.astype(np.float32, copy=False)
    H_plot[H_plot < float(max(1, bin_min))] = np.nan
    return H_plot


def gather_distribution_data_local(x_2d, subsample=0, seed=0):
    nt, _ = x_2d.shape
    rng = np.random.default_rng(seed=seed)
    out = []
    for t in range(nt):
        vals = x_2d[t]
        vals = vals[np.isfinite(vals)].astype(np.float32, copy=False)
        if subsample > 0 and vals.size > subsample:
            vals = vals[rng.choice(vals.size, size=subsample, replace=False)]
        out.append(vals)
    return out


def merge_gathered_timewise_arrays(gathered, nt, size):
    merged = []
    for t in range(nt):
        pieces = [gathered[r][t] for r in range(size) if gathered[r][t].size > 0]
        if len(pieces) == 0:
            merged.append(np.empty(0, dtype=np.float32))
        elif len(pieces) == 1:
            merged.append(pieces[0])
        else:
            merged.append(np.concatenate(pieces))
    return merged


def make_stats(merged):
    nt = len(merged)
    keys = ["min", "p05", "p20", "p40", "p50", "p60", "p80", "p95", "max"]
    stats = {k: np.full(nt, np.nan, dtype=np.float64) for k in keys}
    for i, x in enumerate(merged):
        if x.size == 0:
            continue
        stats["min"][i] = float(np.min(x))
        stats["p05"][i] = float(np.percentile(x, 5.0))
        stats["p20"][i] = float(np.percentile(x, 20.0))
        stats["p40"][i] = float(np.percentile(x, 40.0))
        stats["p50"][i] = float(np.percentile(x, 50.0))
        stats["p60"][i] = float(np.percentile(x, 60.0))
        stats["p80"][i] = float(np.percentile(x, 80.0))
        stats["p95"][i] = float(np.percentile(x, 95.0))
        stats["max"][i] = float(np.max(x))
    return stats


def get_evenly_spaced_violin_indices(i_start, i_end, nmax):
    nseg = i_end - i_start
    if nseg <= 0:
        return np.empty(0, dtype=np.int64)
    if nmax <= 0 or nseg <= nmax:
        return np.arange(i_start, i_end, dtype=np.int64)
    return np.rint(np.linspace(i_start, i_end - 1, nmax)).astype(np.int64)


def format_day_axis(ax, x0, x1, show_labels=True):
    xticks = [x0] if x0 == x1 else [x0, 0.5 * (x0 + x1), x1]
    ax.set_xlim(x0, x1)
    ax.set_xticks(xticks)
    if show_labels:
        ax.set_xticklabels([f"{v:g}" for v in xticks])
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)


def compute_shared_norm(H_plots, cbar_log, bin_min):
    finite_chunks = [H[np.isfinite(H)].ravel() for H in H_plots if np.any(np.isfinite(H))]
    if not finite_chunks:
        raise RuntimeError("No histogram bins left to plot after bin_min masking.")
    finite_vals = np.concatenate(finite_chunks)
    if cbar_log:
        return LogNorm(vmin=max(1.0, float(bin_min)), vmax=float(np.nanmax(finite_vals)))
    return None


def make_triptych_figure(nseg, figure_width_per_day, figure_height, ribbons=False):
    nrows = 3
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=nseg,
        figsize=(figure_width_per_day * nseg + 1.5, figure_height),
        sharex="col",
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )

    for axrow in axs:
        for ax in axrow:
            ax.set_axisbelow(True)

    return fig, axs


def finalize_triptych_axes(axs, day_segments, time_x, flux_ylim, diff_ylim, titles):
    nseg = len(day_segments)
    for row in range(3):
        for iseg, (i_start, i_end) in enumerate(day_segments, start=1):
            ax = axs[row, iseg - 1]
            seg_times = time_x[i_start:i_end]
            if seg_times.size == 0:
                continue
            x0 = float(seg_times[0])
            x1 = float(seg_times[-1])
            format_day_axis(ax, x0, x1, show_labels=(row == 2))
            ax.grid(False)
            if x1 > x0:
                ax.axvline(0.5 * (x0 + x1), color="0.7", linewidth=0.8, zorder=0)
            if row == 0:
                ax.set_title(f"Day {iseg}")
            if iseg == 1:
                ax.set_ylabel(titles[row])

    for iseg in range(nseg):
        axs[0, iseg].set_ylim(*flux_ylim)
        axs[1, iseg].set_ylim(*flux_ylim)
        axs[2, iseg].set_ylim(*diff_ylim)
        axs[2, iseg].axhline(0.0, color="k", linewidth=0.8, alpha=0.7)


def plot_triptych_pcolormesh(
    time_x, day_segments, flux_bins, diff_bins, t_edges,
    ray_H_plot, ts_H_plot, diff_H_plot, norm_all,
    cbar_log, count_kind, figure_height, figure_width_per_day, flux_ylabel,
):
    nseg = len(day_segments)
    fig, axs = make_triptych_figure(nseg, figure_width_per_day, figure_height, ribbons=False)

    pcm = None
    titles = ["Ray-Tracer", "Two-Stream", "Ray-Tracer - Two-Stream"]
    Hs = [ray_H_plot, ts_H_plot, diff_H_plot]
    ybins = [flux_bins, flux_bins, diff_bins]

    for row in range(3):
        for iseg, (i_start, i_end) in enumerate(day_segments, start=1):
            ax = axs[row, iseg - 1]
            seg_edges = t_edges[i_start:i_end + 1]
            seg_H = Hs[row][:, i_start:i_end]

            flux_cmap = "magma"

            if np.any(np.isfinite(seg_H)):
                pcm = ax.pcolormesh(
                    seg_edges, ybins[row], seg_H,
                    shading="auto", cmap=flux_cmap, norm=norm_all, rasterized=True,
                )

    finalize_triptych_axes(
        axs, day_segments, time_x,
        (float(flux_bins[0]), float(flux_bins[-1])),
        (float(diff_bins[0]), float(diff_bins[-1])),
        titles,
    )

    fig.supxlabel("Time Since Simulation Start [Hours]")
    fig.supylabel(flux_ylabel)

    if pcm is not None:
        cb = fig.colorbar(pcm, ax=axs.ravel().tolist(), location="right")
        cb.set_label(
            (r"$\log$(Number of Cells)" if cbar_log else "Number of Cells")
            if count_kind == "cells"
            else (r"$\log$(Number of Columns)" if cbar_log else "Number of Columns")
        )
    return fig


def plot_triptych_boxplot(
    time_x, day_segments, ray_data, ts_data, diff_data,
    flux_ylim, diff_ylim, figure_height, figure_width_per_day, flux_ylabel,
):
    nseg = len(day_segments)
    fig, axs = make_triptych_figure(nseg, figure_width_per_day, figure_height, ribbons=False)

    titles = ["Ray-Tracer", "Two-Stream", "Ray-Tracer - Two-Stream"]
    datasets = [ray_data, ts_data, diff_data]

    for row, data_all in enumerate(datasets):
        for iseg, (i_start, i_end) in enumerate(day_segments, start=1):
            ax = axs[row, iseg - 1]
            seg_times = time_x[i_start:i_end]
            seg_data = data_all[i_start:i_end]

            clean_pos = [float(p) for p, d in zip(seg_times, seg_data) if d.size > 0]
            clean_data = [d for d in seg_data if d.size > 0]

            if clean_data:
                if len(clean_pos) >= 2:
                    dt = np.diff(clean_pos)
                    pos_dt = dt[dt > 0]
                    widths = 0.6 * float(np.min(pos_dt)) if pos_dt.size > 0 else 0.25
                else:
                    widths = 0.25

                ax.boxplot(
                    clean_data,
                    positions=clean_pos,
                    widths=widths,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    boxprops=dict(facecolor="tab:blue", edgecolor="black", alpha=0.45),
                    medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(color="black", linewidth=0.9),
                    capprops=dict(color="black", linewidth=0.9),
                )

    finalize_triptych_axes(axs, day_segments, time_x, flux_ylim, diff_ylim, titles)
    fig.supxlabel("Time Since Simulation Start [Hours]")
    fig.supylabel(flux_ylabel)
    return fig


def plot_triptych_ribbons(
    time_x, day_segments, ray_stats, ts_stats, diff_stats,
    flux_ylim, diff_ylim, figure_height, figure_width_per_day, flux_ylabel,
):
    nseg = len(day_segments)
    fig, axs = make_triptych_figure(nseg, figure_width_per_day, figure_height, ribbons=True)

    titles = ["Ray-Tracer", "Two-Stream", "Ray-Tracer - Two-Stream"]
    stats_all = [ray_stats, ts_stats, diff_stats]
    c = "tab:blue"

    for row, stats in enumerate(stats_all):
        for iseg, (i_start, i_end) in enumerate(day_segments, start=1):
            ax = axs[row, iseg - 1]
            seg_times = time_x[i_start:i_end]

            ax.fill_between(seg_times, stats["p05"][i_start:i_end], stats["p95"][i_start:i_end], color=c, alpha=0.12, linewidth=0.0)
            ax.fill_between(seg_times, stats["p20"][i_start:i_end], stats["p80"][i_start:i_end], color=c, alpha=0.22, linewidth=0.0)
            ax.fill_between(seg_times, stats["p40"][i_start:i_end], stats["p60"][i_start:i_end], color=c, alpha=0.35, linewidth=0.0)
            ax.plot(seg_times, stats["p50"][i_start:i_end], color="black", linewidth=1.2)
            ax.plot(seg_times, stats["min"][i_start:i_end], color="tab:red", linewidth=0.8, alpha=0.75, linestyle="--")
            ax.plot(seg_times, stats["max"][i_start:i_end], color="tab:red", linewidth=0.8, alpha=0.75, linestyle="--")

    finalize_triptych_axes(axs, day_segments, time_x, flux_ylim, diff_ylim, titles)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.2),
        Patch(facecolor=c, edgecolor="none", alpha=0.35),
        Patch(facecolor=c, edgecolor="none", alpha=0.22),
        Patch(facecolor=c, edgecolor="none", alpha=0.12),
        Line2D([0], [0], color="tab:red", linewidth=0.8, linestyle="--"),
    ]
    legend_labels = ["Median", "40-60%", "20-80%", "5-95%", "Min / Max"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=min(len(legend_labels), 6),
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
    )

    fig.supxlabel("Time Since Simulation Start [Hours]")
    fig.supylabel(flux_ylabel)
    return fig


def plot_triptych_violin(
    time_x, day_segments, ray_data, ts_data, diff_data,
    flux_ylim, diff_ylim, figure_height, figure_width_per_day,
    flux_ylabel, violin_max_per_day,
):
    nseg = len(day_segments)
    fig, axs = make_triptych_figure(nseg, figure_width_per_day, figure_height, ribbons=False)

    titles = ["Ray-Tracer", "Two-Stream", "Ray-Tracer - Two-Stream"]
    datasets = [ray_data, ts_data, diff_data]

    for row, data_all in enumerate(datasets):
        for iseg, (i_start, i_end) in enumerate(day_segments, start=1):
            ax = axs[row, iseg - 1]
            seg_idx = get_evenly_spaced_violin_indices(i_start, i_end, violin_max_per_day)

            clean_positions = []
            clean_data = []
            for i in seg_idx:
                d = data_all[int(i)]
                if d.size > 0:
                    clean_positions.append(float(time_x[i]))
                    clean_data.append(d)

            if clean_data:
                if len(clean_positions) >= 2:
                    dt = np.diff(clean_positions)
                    pos_dt = dt[dt > 0]
                    widths = 0.7 * float(np.min(pos_dt)) if pos_dt.size > 0 else 0.25
                else:
                    widths = 0.25

                parts = ax.violinplot(
                    clean_data,
                    positions=clean_positions,
                    widths=widths,
                    showmeans=False,
                    showmedians=True,
                    showextrema=True,
                )
                for body in parts["bodies"]:
                    body.set_facecolor("tab:blue")
                    body.set_edgecolor("black")
                    body.set_alpha(0.35)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("black")
                    parts["cmedians"].set_linewidth(1.0)
                for key in ("cbars", "cmins", "cmaxes"):
                    if key in parts:
                        parts[key].set_color("black")
                        parts[key].set_linewidth(0.8)

    finalize_triptych_axes(axs, day_segments, time_x, flux_ylim, diff_ylim, titles)
    fig.supxlabel("Time Since Simulation Start [Hours]")
    fig.supylabel(flux_ylabel)
    return fig


def save_visualization(
    visualization,
    out_dir,
    file_prefix,
    time_x,
    day_segments,
    flux_bins,
    diff_bins,
    t_edges,
    ray_H_plot,
    ts_H_plot,
    diff_H_plot,
    norm_all,
    cbar_log,
    count_kind,
    flux_ylabel,
    ray_data,
    ts_data,
    diff_data,
    ray_stats,
    ts_stats,
    diff_stats,
    violin_max_per_day,
    flux_ylim,
    diff_ylim,
    figure_height,
    figure_width_per_day,
    figure_dpi,
):
    if visualization == "pcolormesh":
        fig = plot_triptych_pcolormesh(
            time_x, day_segments, flux_bins, diff_bins, t_edges,
            ray_H_plot, ts_H_plot, diff_H_plot, norm_all,
            cbar_log, count_kind, figure_height, figure_width_per_day, flux_ylabel,
        )
    elif visualization == "boxplot":
        fig = plot_triptych_boxplot(
            time_x, day_segments, ray_data, ts_data, diff_data,
            flux_ylim, diff_ylim, figure_height, figure_width_per_day, flux_ylabel,
        )
    elif visualization == "ribbons":
        fig = plot_triptych_ribbons(
            time_x, day_segments, ray_stats, ts_stats, diff_stats,
            flux_ylim, diff_ylim, figure_height, figure_width_per_day, flux_ylabel,
        )
    elif visualization == "violin":
        fig = plot_triptych_violin(
            time_x, day_segments, ray_data, ts_data, diff_data,
            flux_ylim, diff_ylim, figure_height, figure_width_per_day,
            flux_ylabel, violin_max_per_day,
        )
    else:
        raise ValueError(f"Unsupported visualization: {visualization}")

    out_png = os.path.join(out_dir, f"{file_prefix}_{visualization}.png")
    fig.savefig(out_png, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rte_rrtmgp_cpp_input_dir_path", action="store", nargs=1, type=str, required=True)
    parser.add_argument("--rte_rrtmgp_cpp_output_dir_path", action="store", nargs=1, type=str, required=True)
    parser.add_argument("--rte_rrtmgp_cpp_viz_dir_path", action="store", nargs=1, type=str, required=True)

    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--cbar-log", type=str2bool, default=True)
    parser.add_argument("--bin-min", type=int, default=1)
    parser.add_argument("--night-eps", type=float, default=1.0e-3)
    parser.add_argument("--lev-sfc-idx", type=int, default=0)
    parser.add_argument("--lev-top-idx", type=int, default=-1)
    parser.add_argument("--time-units", type=str, default="hours", choices=["hours"])
    parser.add_argument("--lr", type=str, default=None)
    parser.add_argument("--zmax", type=float, default=None)
    parser.add_argument("--distribution-visualization", action="store", type=str, required=False,
        choices=("pcolormesh", "boxplot", "ribbons", "violin"), default=None)
    parser.add_argument("--figure-dpi", action="store", type=int, required=False, default=200)
    parser.add_argument("--figure-height", action="store", type=float, required=False, default=9.0)
    parser.add_argument("--figure-width-per-day", action="store", type=float, required=False, default=3.6)
    parser.add_argument("--distribution-subsample", action="store", type=int, required=False, default=0)
    parser.add_argument("--violin-max-per-day", action="store", type=int, required=False, default=12)
    parser.add_argument("--timings", action="store", type=str2bool, required=False, default=True)
    parser.add_argument("--case", action="store", type=str, required=False, default="")

    args = parser.parse_args()

    if args.bin_min < 1:
        raise ValueError("--bin-min must be >= 1")
    if args.nbins < 1:
        raise ValueError("--nbins must be >= 1")
    if args.distribution_subsample < 0:
        raise ValueError("--distribution-subsample must be >= 0")
    if args.violin_max_per_day < 1:
        raise ValueError("--violin-max-per-day must be >= 1")

    input_dir = args.rte_rrtmgp_cpp_input_dir_path[0]
    output_dir = args.rte_rrtmgp_cpp_output_dir_path[0]
    viz_dir = args.rte_rrtmgp_cpp_viz_dir_path[0]

    visualizations_to_make = (
        ["pcolormesh", "boxplot", "ribbons", "violin"]
        if args.distribution_visualization is None
        else [args.distribution_visualization]
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        log(f"Starting job with {size} MPI ranks", comm)
        log(f"Input dir:  {input_dir}", comm)
        log(f"Output dir: {output_dir}", comm)
        log(f"Plot dir:   {viz_dir}", comm)
        log(f"Visualizations: {visualizations_to_make}", comm)
        log(f"Case: {args.case}", comm)

    pairs = find_pairs(input_dir, output_dir)
    if rank == 0:
        log(f"Found {len(pairs)} lr_XX pairs", comm)
        for lr, (pin, pout) in pairs.items():
            log(f"  {lr}: in={os.path.basename(pin)} out={os.path.basename(pout)}", comm)

    if not pairs:
        if rank == 0:
            log("ERROR: No matching lr_XX pairs found.", comm)
        sys.exit(1)

    if args.lr is not None:
        requested = [s.strip() for s in args.lr.split(",") if s.strip()]
        requested_tags = [s if s.startswith("lr_") else f"lr_{s}" for s in requested]
        pairs = {lr: pairs[lr] for lr in requested_tags if lr in pairs}
        if rank == 0:
            log(f"After --lr filtering, {len(pairs)} pairs remain: {sorted(pairs.keys())}", comm)
        if not pairs:
            if rank == 0:
                log("ERROR: --lr requested resolutions not found in available pairs.", comm)
            sys.exit(1)

    os.makedirs(viz_dir, exist_ok=True)

    diagnostics = [
        ("sw_sfc_up", r"Upwelling Surface Flux $\left[W\,m^{-2}\right]$", "columns"),
        ("sw_sfc_dn", r"Downwelling Surface Flux $\left[W\,m^{-2}\right]$", "columns"),
        ("sw_tod_up", r"Upwelling Top-of-Domain Flux $\left[W\,m^{-2}\right]$", "columns"),
        ("flux_abs", r"Absorbed Flux $\left[W\,m^{-3}\right]$", "cells"),
        ("flux_abs_vi", r"Vertically-Integrated Absorbed Flux $\left[W\,m^{-2}\right]$", "columns"),
        ("net_sfc_flux", r"Net Surface Flux $\left[W\,m^{-2}\right]$", "columns"),
    ]

    if args.case == "GATEIII":
        diagnostics.extend([
            ("sfc_heating", r"Surface Heating Rate $\left[K\,d^{-1}\right]$", "columns"),
            ("atm_heating", r"Atmospheric Heating Rate $\left[K\,d^{-1}\right]$", "cells"),
        ])

    for lr_tag, (in_path, out_path) in pairs.items():
        t0_wall = MPI.Wtime()
        if rank == 0:
            log(f"Processing {lr_tag}", comm)

        t_meta0 = MPI.Wtime()
        with xr.open_dataset(in_path, engine="netcdf4", decode_times=False) as ds_in:
            time_in = ds_in["time"].astype("float64").load().values

            if "mu0" not in ds_in.variables:
                raise KeyError(f"{in_path} missing variable 'mu0' needed for daytime masking")
            mu0 = ds_in["mu0"].astype("float32").load().values

            if "z_lev" not in ds_in.variables:
                raise KeyError(f"{in_path} missing variable 'z_lev' needed for absorbed-flux diagnostics")
            z_lev_in = ds_in["z_lev"].astype("float64").load().values

            if args.zmax is None:
                ilev_max = len(z_lev_in)
            else:
                zmax_m = 1000.0 * args.zmax
                ilev_max = int(np.searchsorted(z_lev_in, zmax_m, side="right"))
                ilev_max = max(2, min(ilev_max, len(z_lev_in)))
        t_meta1 = MPI.Wtime()

        ilev_max = comm.bcast(ilev_max, root=0)
        z_lev_use = z_lev_in[:ilev_max]

        t_out0 = MPI.Wtime()
        with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out:
            time_out = ds_out["time"].astype("float64").load().values
            nt_out = int(ds_out.sizes["time"])
            ny = int(ds_out.sizes["y"])
            nx = int(ds_out.sizes["x"])
            ncol = ny * nx

            if time_out.size != nt_out:
                raise ValueError(
                    f"{out_path}: output time coordinate length {time_out.size} "
                    f"does not match time dimension {nt_out}"
                )

            time_idx = np.rint(time_out).astype(np.int64)
            if np.any(time_idx < 0) or np.any(time_idx >= time_in.size):
                raise ValueError(
                    f"{out_path}: output time indices are out of bounds for input time array."
                )

            time_use = time_in[time_idx]
            nt = time_use.size

            mu0_use = mu0[time_idx, :, :]
            day_valid_3d = np.isfinite(mu0_use) & (mu0_use > args.night_eps)

            if mu0_use.ndim != 3:
                raise ValueError(f"Unexpected mu0 shape for day segmentation: {mu0_use.shape}")

            if rank == 0:
                mu0_1d = mu0_use[:, 0, 0]
                mu0_min = np.nanmin(mu0_use, axis=(1, 2))
                mu0_max = np.nanmax(mu0_use, axis=(1, 2))
                max_spread = np.nanmax(np.abs(mu0_max - mu0_min))
                if np.isfinite(max_spread) and max_spread > 1.0e-12:
                    log(
                        f"Warning: mu0 is not perfectly uniform across x,y. Maximum spread = {max_spread:.3e}",
                        comm,
                    )
                day_segments = find_day_segments(mu0_1d, night_eps=args.night_eps)
                log(f"{lr_tag}: found {len(day_segments)} daytime segments", comm)
            else:
                day_segments = None

            day_segments = comm.bcast(day_segments, root=0)
            if len(day_segments) == 0:
                if rank == 0:
                    log(f"{lr_tag}: no daytime segments found; skipping.", comm)
                continue

            i0, i1 = decompose_1d(ncol, size, rank)
            log(f"{lr_tag}: Assigned col slab [{i0}:{i1}) (nloc={i1-i0})", comm)

            j0 = i0 // nx
            j1 = (i1 - 1) // nx
            y0 = j0
            y1 = j1 + 1

            slab_start_col = y0 * nx
            loc_a = i0 - slab_start_col
            loc_b = i1 - slab_start_col

            day_loc_col = day_valid_3d[:, y0:y1, :].reshape(nt, -1)[:, loc_a:loc_b]
        t_out1 = MPI.Wtime()

        for flux_tag, flux_ylabel, count_kind in diagnostics:
            if rank == 0:
                log(f"{lr_tag}: diagnostic={flux_tag}", comm)

            t_flux0 = MPI.Wtime()
            with xr.open_dataset(out_path, engine="netcdf4", decode_times=False) as ds_out:
                if flux_tag == "sw_sfc_up":
                    ray_loc = (
                        ds_out["rt_flux_sfc_up"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )
                    ts_loc = (
                        ds_out["sw_flux_up"]
                        .isel(time=slice(0, nt), lev=args.lev_sfc_idx, y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )
                    ray_loc = np.where(day_loc_col, ray_loc, np.nan)
                    ts_loc = np.where(day_loc_col, ts_loc, np.nan)

                elif flux_tag == "sw_sfc_dn":
                    ray_loc = (
                        ds_out["rt_flux_sfc_dir"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                        + ds_out["rt_flux_sfc_dif"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                    ).reshape(nt, -1)[:, loc_a:loc_b]

                    ts_loc = (
                        ds_out["sw_flux_dn"]
                        .isel(time=slice(0, nt), lev=args.lev_sfc_idx, y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )
                    ray_loc = np.where(day_loc_col, ray_loc, np.nan)
                    ts_loc = np.where(day_loc_col, ts_loc, np.nan)

                elif flux_tag == "sw_tod_up":
                    ray_loc = (
                        ds_out["rt_flux_tod_up"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )
                    ts_loc = (
                        ds_out["sw_flux_up"]
                        .isel(time=slice(0, nt), lev=args.lev_top_idx, y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )
                    ray_loc = np.where(day_loc_col, ray_loc, np.nan)
                    ts_loc = np.where(day_loc_col, ts_loc, np.nan)

                elif flux_tag in ("net_sfc_flux", "sfc_heating"):
                    ts_sfc_dn = (
                        ds_out["sw_flux_dn"]
                        .isel(time=slice(0, nt), lev=args.lev_sfc_idx, y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )
                    ts_sfc_up = (
                        ds_out["sw_flux_up"]
                        .isel(time=slice(0, nt), lev=args.lev_sfc_idx, y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )

                    rt_sfc_dn = (
                        ds_out["rt_flux_sfc_dir"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                        + ds_out["rt_flux_sfc_dif"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                    ).reshape(nt, -1)[:, loc_a:loc_b]

                    rt_sfc_up = (
                        ds_out["rt_flux_sfc_up"]
                        .isel(time=slice(0, nt), y=slice(y0, y1))
                        .astype("float32").load().values
                        .reshape(nt, -1)[:, loc_a:loc_b]
                    )

                    ts_loc = ts_sfc_dn - ts_sfc_up
                    ray_loc = rt_sfc_dn - rt_sfc_up

                    if flux_tag == "sfc_heating":
                        denom = rho_w * cp_sw * h_m
                        ts_loc = (ts_loc / denom) * sec_per_day
                        ray_loc = (ray_loc / denom) * sec_per_day

                    ray_loc = np.where(day_loc_col, ray_loc, np.nan)
                    ts_loc = np.where(day_loc_col, ts_loc, np.nan)

                elif flux_tag in ("flux_abs", "flux_abs_vi"):
                    if "rt_flux_abs_dif" not in ds_out.variables or "rt_flux_abs_dir" not in ds_out.variables:
                        raise KeyError(f"{out_path} missing rt_flux_abs_dif and/or rt_flux_abs_dir")

                    sw_up_da = ds_out["sw_flux_up"]
                    sw_dn_da = ds_out["sw_flux_dn"]
                    rt_abs_dif_da = ds_out["rt_flux_abs_dif"]
                    rt_abs_dir_da = ds_out["rt_flux_abs_dir"]

                    sw_lev_dim = get_first_dim_name(sw_up_da, ("lev", "z", "z_lev"))
                    rt_lay_dim = get_first_dim_name(rt_abs_dif_da, ("lay", "z", "z_lay"))

                    sw_up = sw_up_da.isel(
                        time=slice(0, nt), y=slice(y0, y1), **{sw_lev_dim: slice(0, ilev_max)}
                    ).astype("float32").load()
                    sw_dn = sw_dn_da.isel(
                        time=slice(0, nt), y=slice(y0, y1), **{sw_lev_dim: slice(0, ilev_max)}
                    ).astype("float32").load()
                    rt_abs_dif = rt_abs_dif_da.isel(
                        time=slice(0, nt), y=slice(y0, y1), **{rt_lay_dim: slice(0, ilev_max - 1)}
                    ).astype("float32").load()
                    rt_abs_dir = rt_abs_dir_da.isel(
                        time=slice(0, nt), y=slice(y0, y1), **{rt_lay_dim: slice(0, ilev_max - 1)}
                    ).astype("float32").load()

                    z_lev = np.asarray(z_lev_use, dtype=np.float64)
                    nlev_needed = sw_up.sizes[sw_lev_dim]
                    if z_lev.size < nlev_needed:
                        raise ValueError(
                            f"z_lev has size {z_lev.size}, but sw_flux_up/sw_flux_dn need at least {nlev_needed} levels"
                        )

                    z_lev = z_lev[:nlev_needed]
                    dz = np.diff(z_lev)
                    nlay = dz.size

                    sw_up_v = sw_up.values
                    sw_dn_v = sw_dn.values
                    ts_flux_diff = (
                        (sw_up_v[:, :-1, :, :] - sw_dn_v[:, :-1, :, :]) +
                        (sw_dn_v[:, 1:, :, :] - sw_up_v[:, 1:, :, :])
                    )
                    ts_abs = ts_flux_diff / dz[None, :, None, None]
                    rt_abs = rt_abs_dif.values + rt_abs_dir.values

                    if rt_abs.shape[1] != nlay:
                        raise ValueError(
                            f"Ray absorbed-flux layer count ({rt_abs.shape[1]}) does not match dz layer count ({nlay})"
                        )

                    if flux_tag == "flux_abs":
                        ts_loc = ts_abs.reshape(nt, nlay, -1)[:, :, loc_a:loc_b].reshape(nt, -1).astype(np.float32, copy=False)
                        ray_loc = rt_abs.reshape(nt, nlay, -1)[:, :, loc_a:loc_b].reshape(nt, -1).astype(np.float32, copy=False)

                        day_loc_cell = np.repeat(day_loc_col[:, None, :], nlay, axis=1).reshape(nt, -1)
                        ray_loc = np.where(day_loc_cell, ray_loc, np.nan)
                        ts_loc = np.where(day_loc_cell, ts_loc, np.nan)
                    else:
                        ts_vi = np.sum(ts_abs * dz[None, :, None, None], axis=1)
                        rt_vi = np.sum(rt_abs * dz[None, :, None, None], axis=1)

                        ts_loc = ts_vi.reshape(nt, -1)[:, loc_a:loc_b].astype(np.float32, copy=False)
                        ray_loc = rt_vi.reshape(nt, -1)[:, loc_a:loc_b].astype(np.float32, copy=False)
                        ray_loc = np.where(day_loc_col, ray_loc, np.nan)
                        ts_loc = np.where(day_loc_col, ts_loc, np.nan)

                elif flux_tag == "atm_heating":
                    in_time_index = xr.DataArray(
                        time_idx[:nt],
                        dims=("time",),
                        coords={"time": np.arange(nt)},
                    )

                    ts_heat_da, rt_heat_da = calc_atm_heating(
                        rad_tran_infile=in_path,
                        rad_tran_outfile=out_path,
                        in_time_index=in_time_index,
                        out_time_index=slice(0, nt),
                        y_index=slice(y0, y1),
                        zmax_index=ilev_max - 1,
                        detailed_calc=False,
                    )

                    ts_heat = ts_heat_da.astype("float32").load().values
                    rt_heat = rt_heat_da.astype("float32").load().values

                    if ts_heat.ndim != 4 or rt_heat.ndim != 4:
                        raise ValueError(
                            f"Expected atm_heating arrays with 4 dims [time, z, y, x]; "
                            f"got ts={ts_heat.shape}, rt={rt_heat.shape}"
                        )

                    nlay = ts_heat.shape[1]
                    ts_loc = ts_heat.reshape(nt, nlay, -1)[:, :, loc_a:loc_b].reshape(nt, -1).astype(np.float32, copy=False)
                    ray_loc = rt_heat.reshape(nt, nlay, -1)[:, :, loc_a:loc_b].reshape(nt, -1).astype(np.float32, copy=False)

                    day_loc_cell = np.repeat(day_loc_col[:, None, :], nlay, axis=1).reshape(nt, -1)
                    ray_loc = np.where(day_loc_cell, ray_loc, np.nan)
                    ts_loc = np.where(day_loc_cell, ts_loc, np.nan)

                else:
                    raise ValueError(f"Unhandled flux_tag={flux_tag}")
            t_flux1 = MPI.Wtime()

            diff_loc = np.full_like(ray_loc, np.nan, dtype=np.float32)
            valid = np.isfinite(ray_loc) & np.isfinite(ts_loc)
            diff_loc[valid] = ray_loc[valid] - ts_loc[valid]

            ray_min, ray_max = finite_minmax(ray_loc)
            ts_min, ts_max = finite_minmax(ts_loc)
            gmin_flux = comm.allreduce(min(ray_min, ts_min), op=MPI.MIN)
            gmax_flux = comm.allreduce(max(ray_max, ts_max), op=MPI.MAX)

            if not np.isfinite(gmin_flux) or not np.isfinite(gmax_flux):
                gmin_flux, gmax_flux = -1.0, 1.0
            if gmin_flux == gmax_flux:
                eps = 1e-12 if gmin_flux == 0.0 else 1e-12 * abs(gmin_flux)
                gmin_flux -= eps
                gmax_flux += eps

            diff_abs_local = finite_minmax(np.abs(diff_loc))[1]
            if not np.isfinite(diff_abs_local):
                diff_abs_local = 0.0
            gmax_abs = comm.allreduce(diff_abs_local, op=MPI.MAX)
            if not np.isfinite(gmax_abs) or gmax_abs == 0.0:
                gmax_abs = 1e-12

            diff_bins = np.linspace(-gmax_abs, gmax_abs, args.nbins + 1, dtype=np.float32)
            flux_bins = np.linspace(gmin_flux, gmax_flux, args.nbins + 1, dtype=np.float32)

            need_pcolormesh = "pcolormesh" in visualizations_to_make
            need_distribution_gather = any(v in visualizations_to_make for v in ("boxplot", "ribbons", "violin"))

            t_hist0 = MPI.Wtime()
            ray_H_plot = ts_H_plot = diff_H_plot = None
            norm_all = None
            if need_pcolormesh:
                log(f"{lr_tag}/{flux_tag}: building histograms...", comm)
                ray_H_plot = reduce_histogram_to_root(
                    comm, build_histograms_fast(ray_loc, flux_bins), bin_min=args.bin_min
                )
                ts_H_plot = reduce_histogram_to_root(
                    comm, build_histograms_fast(ts_loc, flux_bins), bin_min=args.bin_min
                )
                diff_H_plot = reduce_histogram_to_root(
                    comm, build_histograms_fast(diff_loc, diff_bins), bin_min=args.bin_min
                )
            t_hist1 = MPI.Wtime()

            t_dist0 = MPI.Wtime()
            gathered_ray = gathered_ts = gathered_diff = None
            if need_distribution_gather:
                gathered_ray = comm.gather(
                    gather_distribution_data_local(ray_loc, subsample=args.distribution_subsample, seed=12345 + rank),
                    root=0,
                )
                gathered_ts = comm.gather(
                    gather_distribution_data_local(ts_loc, subsample=args.distribution_subsample, seed=22345 + rank),
                    root=0,
                )
                gathered_diff = comm.gather(
                    gather_distribution_data_local(diff_loc, subsample=args.distribution_subsample, seed=32345 + rank),
                    root=0,
                )
            t_dist1 = MPI.Wtime()

            del ray_loc, ts_loc, diff_loc

            if rank != 0:
                if args.timings:
                    parts = [
                        f"meta={t_meta1 - t_meta0:.3f}s",
                        f"outmeta={t_out1 - t_out0:.3f}s",
                        f"fluxread={t_flux1 - t_flux0:.3f}s",
                    ]
                    if need_pcolormesh:
                        parts.append(f"hist={t_hist1 - t_hist0:.3f}s")
                    if need_distribution_gather:
                        parts.append(f"dist={t_dist1 - t_dist0:.3f}s")
                    log(f"{lr_tag}/{flux_tag}: timing breakdown: " + ", ".join(parts), comm, root_only=False)
                continue

            t_post0 = MPI.Wtime()
            time_plot = np.asarray(time_use, dtype=np.float64)
            t_edges = time_edges_from_centers(time_plot)

            if need_pcolormesh:
                norm_all = compute_shared_norm(
                    [ray_H_plot, ts_H_plot, diff_H_plot],
                    cbar_log=args.cbar_log,
                    bin_min=args.bin_min,
                )

            merged_ray = merged_ts = merged_diff = None
            ray_stats = ts_stats = diff_stats = None
            if need_distribution_gather:
                merged_ray = merge_gathered_timewise_arrays(gathered_ray, nt, size)
                merged_ts = merge_gathered_timewise_arrays(gathered_ts, nt, size)
                merged_diff = merge_gathered_timewise_arrays(gathered_diff, nt, size)

                if "ribbons" in visualizations_to_make:
                    ray_stats = make_stats(merged_ray)
                    ts_stats = make_stats(merged_ts)
                    diff_stats = make_stats(merged_diff)

            flux_ylim = (float(flux_bins[0]), float(flux_bins[-1]))
            diff_ylim = (float(diff_bins[0]), float(diff_bins[-1]))
            t_post1 = MPI.Wtime()

            t_plot0 = MPI.Wtime()
            file_prefix = f"{lr_tag}_{flux_tag}"
            out_paths = []
            for viz in visualizations_to_make:
                out_paths.append(
                    save_visualization(
                        visualization=viz,
                        out_dir=viz_dir,
                        file_prefix=file_prefix,
                        time_x=time_plot,
                        day_segments=day_segments,
                        flux_bins=flux_bins,
                        diff_bins=diff_bins,
                        t_edges=t_edges,
                        ray_H_plot=ray_H_plot,
                        ts_H_plot=ts_H_plot,
                        diff_H_plot=diff_H_plot,
                        norm_all=norm_all,
                        cbar_log=args.cbar_log,
                        count_kind=count_kind,
                        flux_ylabel=flux_ylabel,
                        ray_data=merged_ray,
                        ts_data=merged_ts,
                        diff_data=merged_diff,
                        ray_stats=ray_stats,
                        ts_stats=ts_stats,
                        diff_stats=diff_stats,
                        violin_max_per_day=args.violin_max_per_day,
                        flux_ylim=flux_ylim,
                        diff_ylim=diff_ylim,
                        figure_height=args.figure_height,
                        figure_width_per_day=args.figure_width_per_day,
                        figure_dpi=args.figure_dpi,
                    )
                )
            t_plot1 = MPI.Wtime()

            for out_png in out_paths:
                log(f"Wrote: {out_png}", comm)

            if args.timings:
                parts = [
                    f"meta={t_meta1 - t_meta0:.3f}s",
                    f"outmeta={t_out1 - t_out0:.3f}s",
                    f"fluxread={t_flux1 - t_flux0:.3f}s",
                    f"post={t_post1 - t_post0:.3f}s",
                    f"plot={t_plot1 - t_plot0:.3f}s",
                ]
                if need_pcolormesh:
                    parts.append(f"hist={t_hist1 - t_hist0:.3f}s")
                if need_distribution_gather:
                    parts.append(f"dist={t_dist1 - t_dist0:.3f}s")
                log(f"{lr_tag}/{flux_tag}: timing breakdown: " + ", ".join(parts), comm)

        if rank == 0:
            log(f"{lr_tag}: done (walltime {MPI.Wtime() - t0_wall:.2f} s).", comm)

    if rank == 0:
        log("All resolutions complete.", comm)


if __name__ == "__main__":
    main()