#!/usr/bin/env python3
import os
import sys
import argparse
import time as pytime

import numpy as np
from mpi4py import MPI
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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


def compute_time_edges(time_hours_since_start):
    nt = len(time_hours_since_start)
    t_edges = np.empty(nt + 1, dtype=np.float64)
    if nt == 1:
        t_edges[0] = time_hours_since_start[0] - 0.5
        t_edges[1] = time_hours_since_start[0] + 0.5
    else:
        t_edges[1:-1] = 0.5 * (time_hours_since_start[1:] + time_hours_since_start[:-1])
        t_edges[0] = time_hours_since_start[0] - (t_edges[1] - time_hours_since_start[0])
        t_edges[-1] = time_hours_since_start[-1] + (time_hours_since_start[-1] - t_edges[-2])
    return t_edges


def read_root_metadata(dpscream_file_path, spinup_end, var_mu0):
    with xr.open_dataset(dpscream_file_path, engine="netcdf4", decode_times=False) as ds:
        time = ds["time"].isel(time=slice(spinup_end, None)).astype("float64").load().values
        mu0_da = ds[var_mu0].isel(time=slice(spinup_end, None))
        if "ncol" in mu0_da.dims:
            mu0_1d = mu0_da.isel(ncol=0).astype("float32").load().values
        else:
            mu0_1d = mu0_da.astype("float32").load().values
    return time, mu0_1d


def reduce_global_minmax_from_local_arrays(lw, clr, mu0, eps_clr, night_eps, comm):
    valid = np.isfinite(lw) & np.isfinite(clr) & np.isfinite(mu0)
    valid &= (clr > eps_clr) & (mu0 > night_eps)

    if np.any(valid):
        vals = lw[valid] / clr[valid]
        local_min = float(np.min(vals))
        local_max = float(np.max(vals))
    else:
        local_min = np.inf
        local_max = -np.inf

    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return global_min, global_max


def build_histograms_fast_from_local_arrays(lw, clr, mu0, bins, eps_clr, night_eps):
    nt, _ = lw.shape
    nbins = bins.size - 1
    global_min = float(bins[0])
    global_max = float(bins[-1])

    valid = np.isfinite(lw) & np.isfinite(clr) & np.isfinite(mu0)
    valid &= (clr > eps_clr) & (mu0 > night_eps)

    if not np.any(valid):
        return np.zeros((nt, nbins), dtype=np.int64)

    t_idx, c_idx = np.nonzero(valid)
    vals = lw[t_idx, c_idx] / clr[t_idx, c_idx]
    vals = np.clip(vals, global_min, global_max)

    bin_width = (global_max - global_min) / nbins
    if bin_width <= 0.0 or not np.isfinite(bin_width):
        raise ValueError("Invalid histogram bin width.")

    b_idx = np.floor((vals - global_min) / bin_width).astype(np.int64)
    b_idx = np.clip(b_idx, 0, nbins - 1)

    flat_idx = t_idx * nbins + b_idx
    counts_flat = np.bincount(flat_idx, minlength=nt * nbins)
    return counts_flat.reshape(nt, nbins).astype(np.int64, copy=False)


def gather_distribution_data_local(lw, clr, mu0, eps_clr, night_eps, subsample=0, seed=0):
    nt, _ = lw.shape
    rng = np.random.default_rng(seed=seed)
    out = []

    for t in range(nt):
        valid = np.isfinite(lw[t]) & np.isfinite(clr[t]) & np.isfinite(mu0[t])
        valid &= (clr[t] > eps_clr) & (mu0[t] > night_eps)

        if np.any(valid):
            vals = (lw[t, valid] / clr[t, valid]).astype(np.float32, copy=False)
            if subsample > 0 and vals.size > subsample:
                idx = rng.choice(vals.size, size=subsample, replace=False)
                vals = vals[idx]
        else:
            vals = np.empty(0, dtype=np.float32)

        out.append(vals)

    return out


def merge_gathered_timewise_arrays(gathered, nt2, size):
    merged = []
    for t in range(nt2):
        pieces = [gathered[r][t] for r in range(size) if gathered[r][t].size > 0]
        if len(pieces) == 0:
            x = np.empty(0, dtype=np.float32)
        elif len(pieces) == 1:
            x = pieces[0]
        else:
            x = np.concatenate(pieces)
        merged.append(x)
    return merged


def get_violin_keep_indices(nt2, max_violin_times):
    if max_violin_times <= 0 or max_violin_times >= nt2:
        return np.arange(nt2, dtype=np.int64)
    return np.linspace(0, nt2 - 1, max_violin_times, dtype=np.int64)


def format_day_axis(ax, x0, x1, iseg):
    if x0 == x1:
        xticks = [x0]
    else:
        xmid = 0.5 * (x0 + x1)
        xticks = [x0, xmid, x1]

    ax.set_xlim(x0, x1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:g}" for v in xticks])
    ax.text(
        0.02, 0.98, f"Day {iseg}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=1.0, edgecolor="black"),
    )


def plot_pcolormesh(
    axs, day_segments, time_hours_since_start, t_edges, bins, H_plot,
    norm, cbar_log, global_min, global_max, fig
):
    pcm = None

    for iseg, (ax, (i_start, i_end)) in enumerate(zip(axs, day_segments), start=1):
        seg_times = time_hours_since_start[i_start:i_end]
        if seg_times.size == 0:
            continue

        seg_edges = t_edges[i_start:i_end + 1]
        seg_H = H_plot[:, i_start:i_end]

        if np.any(np.isfinite(seg_H)):
            pcm = ax.pcolormesh(
                seg_edges,
                bins,
                seg_H,
                shading="auto",
                cmap="magma",
                norm=norm,
                rasterized=True,
            )

        x0 = float(seg_times[0])
        x1 = float(seg_times[-1])
        format_day_axis(ax, x0, x1, iseg)
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.35, zorder=0)

    axs[0].set_ylim(global_min, global_max)

    if pcm is not None:
        cb = fig.colorbar(pcm, ax=axs, location="right")
        if cbar_log:
            cb.set_label(r"$\log$(Number of Columns)")
        else:
            cb.set_label("Number of Columns")


def plot_boxplot(axs, day_segments, time_hours_since_start, boxplot_data, global_min, global_max):
    for iseg, (ax, (i_start, i_end)) in enumerate(zip(axs, day_segments), start=1):
        seg_times = time_hours_since_start[i_start:i_end]
        if seg_times.size == 0:
            continue

        seg_data = boxplot_data[i_start:i_end]

        clean_data = []
        clean_pos = []
        for p, d in zip(seg_times, seg_data):
            if d.size > 0:
                clean_data.append(d)
                clean_pos.append(float(p))

        if len(clean_data) > 0:
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

        x0 = float(seg_times[0])
        x1 = float(seg_times[-1])
        format_day_axis(ax, x0, x1, iseg)
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.35, zorder=0)

    axs[0].set_ylim(global_min, global_max)


def plot_ribbons(axs, day_segments, time_hours_since_start, ribbon_stats, global_min, global_max, fig):
    c = "tab:blue"

    for iseg, (ax, (i_start, i_end)) in enumerate(zip(axs, day_segments), start=1):
        seg_times = time_hours_since_start[i_start:i_end]
        if seg_times.size == 0:
            continue

        pmin = ribbon_stats["min"][i_start:i_end]
        p05 = ribbon_stats["p05"][i_start:i_end]
        p20 = ribbon_stats["p20"][i_start:i_end]
        p40 = ribbon_stats["p40"][i_start:i_end]
        p50 = ribbon_stats["p50"][i_start:i_end]
        p60 = ribbon_stats["p60"][i_start:i_end]
        p80 = ribbon_stats["p80"][i_start:i_end]
        p95 = ribbon_stats["p95"][i_start:i_end]
        pmax = ribbon_stats["max"][i_start:i_end]

        ax.fill_between(seg_times, p05, p95, color=c, alpha=0.12, linewidth=0.0)
        ax.fill_between(seg_times, p20, p80, color=c, alpha=0.22, linewidth=0.0)
        ax.fill_between(seg_times, p40, p60, color=c, alpha=0.35, linewidth=0.0)
        ax.plot(seg_times, p50, color="black", linewidth=1.2)
        ax.plot(seg_times, pmin, color="tab:red", linewidth=0.8, alpha=0.75, linestyle="--")
        ax.plot(seg_times, pmax, color="tab:red", linewidth=0.8, alpha=0.75, linestyle="--")

        x0 = float(seg_times[0])
        x1 = float(seg_times[-1])
        format_day_axis(ax, x0, x1, iseg)
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.35, zorder=0)

    axs[0].set_ylim(global_min, global_max)
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.2),
        Patch(facecolor=c, edgecolor="none", alpha=0.35),
        Patch(facecolor=c, edgecolor="none", alpha=0.22),
        Patch(facecolor=c, edgecolor="none", alpha=0.12),
        Line2D([0], [0], color="tab:red", linewidth=0.8, linestyle="--"),
    ]
    legend_labels = ["Median", "40-60%", "20-80%", "5-95%", "Min / Max"]

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 6),
            bbox_to_anchor=(0.5, 1.07),
            frameon=True,
        )


def plot_violin(
    axs, day_segments, time_hours_since_start, violin_keep_indices,
    violin_data, global_min, global_max, violin_max_per_day
):
    index_to_data = {int(i): d for i, d in zip(violin_keep_indices, violin_data)}

    for iseg, (ax, (i_start, i_end)) in enumerate(zip(axs, day_segments), start=1):
        seg_all_idx = np.arange(i_start, i_end, dtype=np.int64)
        seg_idx = np.array([i for i in seg_all_idx if int(i) in index_to_data], dtype=np.int64)

        seg_times = time_hours_since_start[i_start:i_end]
        if seg_times.size == 0:
            continue

        if seg_idx.size > 0:
            if violin_max_per_day > 0 and seg_idx.size > violin_max_per_day:
                pick = np.linspace(0, seg_idx.size - 1, violin_max_per_day, dtype=np.int64)
                seg_idx = seg_idx[pick]

            clean_positions = []
            clean_data = []
            for i in seg_idx:
                d = index_to_data[int(i)]
                if d.size > 0:
                    clean_positions.append(float(time_hours_since_start[i]))
                    clean_data.append(d)

            if len(clean_data) > 0:
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

        x0 = float(seg_times[0])
        x1 = float(seg_times[-1])
        format_day_axis(ax, x0, x1, iseg)
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.35, zorder=0)

    axs[0].set_ylim(global_min, global_max)


def make_figure(day_segments, figure_width_per_day, figure_height):
    nseg = len(day_segments)
    fig_width = max(10.0, figure_width_per_day * nseg)
    fig, axs = plt.subplots(
        1,
        nseg,
        figsize=(fig_width, figure_height),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axs = axs[0]
    for ax in axs:
        ax.set_axisbelow(True)
    return fig, axs


def save_visualization(
    visualization,
    out_dir,
    time_hours_since_start,
    day_segments,
    global_min,
    global_max,
    bins,
    t_edges,
    H_plot,
    norm,
    cbar_log,
    boxplot_data,
    ribbon_stats,
    violin_keep_indices,
    violin_data,
    violin_max_per_day,
    figure_height,
    figure_width_per_day,
    figure_dpi,
):
    nseg = len(day_segments)
    fig_width = max(10.0, figure_width_per_day * nseg)

    fig, axs = make_figure(day_segments, figure_width_per_day, figure_height)

    if visualization == "pcolormesh":
        plot_pcolormesh(
            axs=axs,
            day_segments=day_segments,
            time_hours_since_start=time_hours_since_start,
            t_edges=t_edges,
            bins=bins,
            H_plot=H_plot,
            norm=norm,
            cbar_log=cbar_log,
            global_min=global_min,
            global_max=global_max,
            fig=fig,
        )
    elif visualization == "boxplot":
        plot_boxplot(
            axs=axs,
            day_segments=day_segments,
            time_hours_since_start=time_hours_since_start,
            boxplot_data=boxplot_data,
            global_min=global_min,
            global_max=global_max,
        )
    elif visualization == "ribbons":
        plot_ribbons(
            axs=axs,
            day_segments=day_segments,
            time_hours_since_start=time_hours_since_start,
            ribbon_stats=ribbon_stats,
            global_min=global_min,
            global_max=global_max,
            fig=fig,
        )
    elif visualization == "violin":
        plot_violin(
            axs=axs,
            day_segments=day_segments,
            time_hours_since_start=time_hours_since_start,
            violin_keep_indices=violin_keep_indices,
            violin_data=violin_data,
            global_min=global_min,
            global_max=global_max,
            violin_max_per_day=violin_max_per_day,
        )
    else:
        raise ValueError(f"Unsupported visualization: {visualization}")

    fig.supxlabel("Time Since Simulation Start [Hours]")
    fig.supylabel("Outgoing Longwave Radiative Flux (Actual / Clear-Sky)")

    out_png = os.path.join(out_dir, f"olr_{visualization}.png")
    fig.savefig(out_png, dpi=figure_dpi, bbox_inches="tight")

    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dpscream_file_path",
        action="store",
        nargs=1,
        type=str,
        required=True,
        help="Path to DP-SCREAM output (.nc).",
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_viz_dir_path",
        action="store",
        nargs=1,
        type=str,
        required=False,
        default=["comparison"],
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--spinup-end",
        action="store",
        type=int,
        required=False,
        default=0,
        help="Time index where spin-up has ended (default: 0). Uses time[spinup_end:].",
    )
    parser.add_argument(
        "--nbins",
        action="store",
        type=int,
        required=False,
        default=200,
        help="Number of histogram bins for pcolormesh (default: 200).",
    )
    parser.add_argument(
        "--eps-clr",
        action="store",
        type=float,
        required=False,
        default=1.0e-6,
        help="Treat clear-sky LW <= eps as invalid for normalization (default: 1e-6).",
    )
    parser.add_argument(
        "--cbar-log",
        action="store",
        type=str2bool,
        required=False,
        default=True,
        help="Use log scaling on the colorbar/counts for pcolormesh (default: true).",
    )
    parser.add_argument(
        "--night-eps",
        action="store",
        type=float,
        required=False,
        default=1.0e-3,
        help="Night when mu0 <= night_eps (default: 1.e-3).",
    )
    parser.add_argument(
        "--bin-min",
        action="store",
        type=int,
        required=False,
        default=1,
        help="For pcolormesh, suppress histogram bins with fewer than this many entries (default: 1).",
    )
    parser.add_argument(
        "--distribution-visualization",
        action="store",
        type=str,
        required=False,
        choices=("pcolormesh", "boxplot", "ribbons", "violin"),
        default=None,
        help=(
            "Visualization mode. If omitted, generate all modes in separate files: "
            "pcolormesh, boxplot, ribbons, violin."
        ),
    )
    parser.add_argument(
        "--figure-dpi",
        action="store",
        type=int,
        required=False,
        default=200,
        help="Output figure DPI (default: 200).",
    )
    parser.add_argument(
        "--figure-height",
        action="store",
        type=float,
        required=False,
        default=5.0,
        help="Figure height in inches (default: 5.0).",
    )
    parser.add_argument(
        "--figure-width-per-day",
        action="store",
        type=float,
        required=False,
        default=3.6,
        help="Figure width per daytime segment in inches (default: 3.6).",
    )
    parser.add_argument(
        "--boxplot-subsample",
        action="store",
        type=int,
        required=False,
        default=0,
        help=(
            "If > 0, randomly subsample at most this many local valid columns per time per rank "
            "before gather for boxplot/ribbons/violin. 0 means use all local valid columns."
        ),
    )
    parser.add_argument(
        "--violin-max-times",
        action="store",
        type=int,
        required=False,
        default=48,
        help=(
            "Maximum number of total time indices retained for violin plots across the full run. "
            "Default: 48."
        ),
    )
    parser.add_argument(
        "--violin-max-per-day",
        action="store",
        type=int,
        required=False,
        default=12,
        help="Maximum number of violin positions per day subplot (default: 12).",
    )
    parser.add_argument(
        "--timings",
        action="store",
        type=str2bool,
        required=False,
        default=True,
        help="Print timing breakdowns (default: true).",
    )
    args = parser.parse_args()

    dpscream_file_path = args.dpscream_file_path[0]
    out_dir = args.rte_rrtmgp_cpp_viz_dir_path[0]
    spinup_end = args.spinup_end
    nbins = args.nbins
    eps_clr = args.eps_clr
    cbar_log = args.cbar_log
    night_eps = args.night_eps
    bin_min = args.bin_min
    distribution_visualization = args.distribution_visualization
    figure_dpi = args.figure_dpi
    figure_height = args.figure_height
    figure_width_per_day = args.figure_width_per_day
    boxplot_subsample = args.boxplot_subsample
    violin_max_times = args.violin_max_times
    violin_max_per_day = args.violin_max_per_day
    timings = args.timings

    if bin_min < 1:
        raise ValueError("--bin-min must be >= 1")
    if nbins < 1:
        raise ValueError("--nbins must be >= 1")
    if boxplot_subsample < 0:
        raise ValueError("--boxplot-subsample must be >= 0")
    if violin_max_times < 1:
        raise ValueError("--violin-max-times must be >= 1")
    if violin_max_per_day < 1:
        raise ValueError("--violin-max-per-day must be >= 1")

    if distribution_visualization is None:
        visualizations_to_make = ["pcolormesh", "boxplot", "ribbons", "violin"]
    else:
        visualizations_to_make = [distribution_visualization]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    var_lw = "LW_flux_up_at_model_top"
    var_clr = "LW_clrsky_flux_up_at_model_top"
    var_mu0 = "cosine_solar_zenith_angle"

    if rank == 0:
        log(f"Starting job with {size} MPI ranks", comm)
        log(f"Input file: {dpscream_file_path}", comm)
        log(f"Output dir: {out_dir}", comm)
        log(
            f"spinup_end={spinup_end}, nbins={nbins}, cbar_log={cbar_log}, "
            f"bin_min={bin_min}, night_eps={night_eps}, "
            f"visualizations={visualizations_to_make}, "
            f"boxplot_subsample={boxplot_subsample}, "
            f"violin_max_times={violin_max_times}, violin_max_per_day={violin_max_per_day}",
            comm,
        )

    t_wall0 = MPI.Wtime()

    t_read0 = MPI.Wtime()
    with xr.open_dataset(dpscream_file_path, engine="netcdf4", decode_times=False) as ds:
        nt = int(ds.sizes["time"])
        ncol = int(ds.sizes["ncol"])

        if spinup_end < 0 or spinup_end >= nt:
            raise ValueError(f"--spinup-end must be in [0, {nt-1}], got {spinup_end}")

        i0, i1 = decompose_1d(ncol, size, rank)
        nt2 = nt - spinup_end
        nloc = i1 - i0
        log(f"Assigned ncol slab [{i0}:{i1}) (nloc={nloc})", comm)

        ds_s = ds.isel(time=slice(spinup_end, None), ncol=slice(i0, i1))
        lw = ds_s[var_lw].astype("float32").load().values
        clr = ds_s[var_clr].astype("float32").load().values
        mu0 = ds_s[var_mu0].astype("float32").load().values
    t_read1 = MPI.Wtime()

    t_minmax0 = MPI.Wtime()
    global_min, global_max = reduce_global_minmax_from_local_arrays(
        lw=lw, clr=clr, mu0=mu0,
        eps_clr=eps_clr, night_eps=night_eps,
        comm=comm,
    )
    t_minmax1 = MPI.Wtime()

    if rank == 0:
        log(f"Global normalized OLR range: min={global_min:.6g}, max={global_max:.6g}", comm)

    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
        if rank == 0:
            log("ERROR: Could not determine finite global min/max for normalized OLR.", comm)
            log("Hint: daytime masking may have removed all valid points.", comm)
        sys.exit(2)

    bins = np.linspace(global_min, global_max, nbins + 1, dtype=np.float32)

    H = None
    gathered_dist = None

    need_pcolormesh = "pcolormesh" in visualizations_to_make
    need_distribution_gather = any(v in visualizations_to_make for v in ("boxplot", "ribbons", "violin"))

    if need_pcolormesh:
        t_hist0 = MPI.Wtime()
        H_local = build_histograms_fast_from_local_arrays(
            lw=lw, clr=clr, mu0=mu0,
            bins=bins, eps_clr=eps_clr, night_eps=night_eps,
        )
        t_hist1 = MPI.Wtime()

        t_reduce0 = MPI.Wtime()
        if rank == 0:
            H = np.empty_like(H_local)
        else:
            H = None
        comm.Reduce(H_local, H, op=MPI.SUM, root=0)
        del H_local
        t_reduce1 = MPI.Wtime()
    else:
        t_hist0 = t_hist1 = t_reduce0 = t_reduce1 = MPI.Wtime()

    if need_distribution_gather:
        t_dist0 = MPI.Wtime()
        local_rows = gather_distribution_data_local(
            lw=lw, clr=clr, mu0=mu0,
            eps_clr=eps_clr, night_eps=night_eps,
            subsample=boxplot_subsample,
            seed=12345 + rank,
        )
        t_dist1 = MPI.Wtime()

        t_gather0 = MPI.Wtime()
        gathered_dist = comm.gather(local_rows, root=0)
        t_gather1 = MPI.Wtime()
    else:
        local_rows = None
        t_dist0 = t_dist1 = t_gather0 = t_gather1 = MPI.Wtime()

    del lw, clr, mu0

    if rank != 0:
        if timings:
            parts = [
                f"read={t_read1 - t_read0:.3f}s",
                f"minmax={t_minmax1 - t_minmax0:.3f}s",
            ]
            if need_pcolormesh:
                parts.extend([
                    f"hist={t_hist1 - t_hist0:.3f}s",
                    f"reduce={t_reduce1 - t_reduce0:.3f}s",
                ])
            if need_distribution_gather:
                parts.extend([
                    f"dist_local={t_dist1 - t_dist0:.3f}s",
                    f"dist_gather={t_gather1 - t_gather0:.3f}s",
                ])
            log("Timing breakdown: " + ", ".join(parts), comm, root_only=False)

        log(f"Rank complete (walltime {MPI.Wtime() - t_wall0:.2f} s).", comm, root_only=False)
        return

    t_meta0 = MPI.Wtime()
    time, mu0_1d = read_root_metadata(
        dpscream_file_path=dpscream_file_path,
        spinup_end=spinup_end,
        var_mu0=var_mu0,
    )
    t_meta1 = MPI.Wtime()

    day_segments = find_day_segments(mu0_1d=mu0_1d, night_eps=night_eps)
    if len(day_segments) == 0:
        log("ERROR: No daytime segments found.", comm)
        sys.exit(4)

    log(f"Found {len(day_segments)} daytime segments.", comm)

    time0 = float(time[0])
    time_hours_since_start = (time - time0) * 24.0
    t_edges = compute_time_edges(time_hours_since_start)

    H_plot = None
    norm = None
    merged_dist = None
    ribbon_stats = None
    violin_keep_indices = None
    violin_data = None

    t_post0 = MPI.Wtime()

    if need_pcolormesh:
        H_plot = H.T.astype(np.float32, copy=False)
        if bin_min > 1:
            H_plot[H_plot < float(bin_min)] = np.nan
        else:
            H_plot[H_plot < 1.0] = np.nan

        finite_vals = H_plot[np.isfinite(H_plot)]
        if finite_vals.size == 0:
            log("ERROR: No histogram bins left to plot after applying --bin-min.", comm)
            sys.exit(3)

        if cbar_log:
            vmin = max(1.0, float(bin_min))
            vmax = float(np.nanmax(H_plot))
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

    if need_distribution_gather:
        merged_dist = merge_gathered_timewise_arrays(gathered_dist, nt2, size)

        if "ribbons" in visualizations_to_make:
            qnames = ["min", "p05", "p20", "p40", "p50", "p60", "p80", "p95", "max"]
            ribbon_stats = {k: np.full(nt2, np.nan, dtype=np.float64) for k in qnames}
            for t, x in enumerate(merged_dist):
                if x.size == 0:
                    continue
                ribbon_stats["min"][t] = float(np.min(x))
                ribbon_stats["p05"][t] = float(np.percentile(x, 5.0))
                ribbon_stats["p20"][t] = float(np.percentile(x, 20.0))
                ribbon_stats["p40"][t] = float(np.percentile(x, 40.0))
                ribbon_stats["p50"][t] = float(np.percentile(x, 50.0))
                ribbon_stats["p60"][t] = float(np.percentile(x, 60.0))
                ribbon_stats["p80"][t] = float(np.percentile(x, 80.0))
                ribbon_stats["p95"][t] = float(np.percentile(x, 95.0))
                ribbon_stats["max"][t] = float(np.max(x))

        if "violin" in visualizations_to_make:
            violin_keep_indices = get_violin_keep_indices(nt2, violin_max_times)
            violin_data = [merged_dist[i] for i in violin_keep_indices]

    t_post1 = MPI.Wtime()

    t_plot0 = MPI.Wtime()
    os.makedirs(out_dir, exist_ok=True)

    out_paths = []
    for viz in visualizations_to_make:
        if viz == "boxplot" and merged_dist is None:
            log("Skipping boxplot because no gathered distribution data are available.", comm)
            continue

        out_png = save_visualization(
            visualization=viz,
            out_dir=out_dir,
            time_hours_since_start=time_hours_since_start,
            day_segments=day_segments,
            global_min=global_min,
            global_max=global_max,
            bins=bins,
            t_edges=t_edges,
            H_plot=H_plot,
            norm=norm,
            cbar_log=cbar_log,
            boxplot_data=merged_dist,
            ribbon_stats=ribbon_stats,
            violin_keep_indices=violin_keep_indices,
            violin_data=violin_data,
            violin_max_per_day=violin_max_per_day,
            figure_height=figure_height,
            figure_width_per_day=figure_width_per_day,
            figure_dpi=figure_dpi,
        )
        out_paths.append(out_png)

    t_plot1 = MPI.Wtime()

    for out_png in out_paths:
        log(f"Wrote: {out_png}", comm)

    if timings:
        parts = [
            f"read={t_read1 - t_read0:.3f}s",
            f"minmax={t_minmax1 - t_minmax0:.3f}s",
            f"metadata={t_meta1 - t_meta0:.3f}s",
            f"post={t_post1 - t_post0:.3f}s",
            f"plot={t_plot1 - t_plot0:.3f}s",
        ]
        if need_pcolormesh:
            parts.extend([
                f"hist={t_hist1 - t_hist0:.3f}s",
                f"reduce={t_reduce1 - t_reduce0:.3f}s",
            ])
        if need_distribution_gather:
            parts.extend([
                f"dist_local={t_dist1 - t_dist0:.3f}s",
                f"dist_gather={t_gather1 - t_gather0:.3f}s",
            ])
        log("Timing breakdown: " + ", ".join(parts), comm)

    log(f"All done. Total walltime {MPI.Wtime() - t_wall0:.2f} s.", comm)


if __name__ == "__main__":
    main()