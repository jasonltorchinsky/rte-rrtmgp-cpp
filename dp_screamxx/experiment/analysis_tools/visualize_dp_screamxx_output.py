#!/usr/bin/env python3
"""
MPI-parallel DP-SCREAM quicklook plotting.

Parallelization strategy:
- Rank 0 reads the NetCDF once (avoids hammering the filesystem).
- Rank 0 computes the 1D time series for each requested plot type.
- Each MPI rank is assigned a subset of plot types to render/save (Matplotlib is the slow part).
- Rank 0 prints Slurm-friendly progress messages; other ranks stay mostly quiet.

Run (example):
  srun -n 4 python visualize_dpscream_output_mpi.py --dpscream_file_path /path/out.nc \
       --rte_rrtmgp_cpp_viz_dir_path comparison
"""

import os
import sys
import argparse
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # important for batch/Slurm
import matplotlib.pyplot as plt

from mpi4py import MPI

# If you need local imports from your repo layout
exp_hres_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if exp_hres_dir not in sys.path:
    sys.path.append(exp_hres_dir)

# Local Library Imports (kept, though sort_mask isn't used below)
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET  # type helpers
from convert_utils import get_sort_mask  # noqa: F401


prog_name: str = "visualize_dpscream_output"
prog_desc: str = "Visualizes DP-SCREAM output (MPI-parallel plotting)."


def log(comm: MPI.Comm, msg: str, rank0_only: bool = True) -> None:
    """Print Slurm-friendly messages with rank prefix."""
    r = comm.Get_rank()
    if (not rank0_only) or (r == 0):
        print(f"[rank {r:04d}] {msg}", flush=True)


def partition_items(items: List[str], comm: MPI.Comm) -> List[str]:
    """Round-robin partition of items across ranks."""
    r = comm.Get_rank()
    n = comm.Get_size()
    return [it for i, it in enumerate(items) if (i % n) == r]


def compute_series_on_root(
    comm: MPI.Comm,
    dpscream_file_path: str,
    plot_keys: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[Dict[str, Dict]]]:
    """
    Rank 0 loads data and builds per-plot 1D series + metadata.
    Returns (t, time_hours, file_name_root, payload_dict) on root; Nones on others.
    payload_dict[field_str] = {
        "field_key": str,
        "ylabel": str,
        "title": str,
        "y": np.ndarray (1D float),
    }
    """
    rank = comm.Get_rank()
    if rank != 0:
        return None, None, None, None

    log(comm, f"Opening dataset: {dpscream_file_path}")
    xr_dpscream: XR_DATASET = xr.open_dataset(dpscream_file_path, engine="netcdf4")

    # (Optional) If you truly need sorting later, compute it here once
    # sort_mask = get_sort_mask(xr_dpscream)

    file_ext: re.Pattern = re.compile(r"\.nc$")
    file_name_root: str = file_ext.sub("", os.path.basename(dpscream_file_path))

    # Hours since simulation start (time in ns)
    time: np.ndarray = xr_dpscream["time"].values
    time_hours: np.ndarray = (time - time[0]).astype(np.float64) / 3.6e12

    # Time-step numbers
    t: np.ndarray = np.arange(time_hours.size, dtype=np.int64)

    payload: Dict[str, Dict] = {}

    for field_str in plot_keys:
        if field_str == "sza":
            field_key = "cosine_solar_zenith_angle"
            # take a representative column
            cos_sza = xr_dpscream[field_key].isel(ncol=0).values
            y = np.arccos(np.clip(cos_sza.astype(np.float64, copy=False), -1.0, 1.0))
            payload[field_str] = dict(
                field_key=field_key,
                ylabel="Solar Zenith Angle (rad)",
                title=f"{file_name_root} - SZA",
                y=y,
            )

        elif field_str in ["clt_max", "clt_ran", "clt_min"]:
            field_key = "cldfrac_tot"

            # mean over columns -> (time, lev)
            cf = xr_dpscream[field_key].mean(dim="ncol").values.astype(np.float64, copy=False)

            # Reduce over lev to a single time series
            # Assumes cf shape is (time, lev) after mean(ncol); if dims differ, adjust here.
            if field_str == "clt_max":
                y = np.max(cf, axis=-1)
                title_str = "Cloud Cover - Maximum Overlap"
            elif field_str == "clt_min":
                y = np.clip(np.sum(cf, axis=-1), None, 1.0)
                title_str = "Cloud Cover - Minimum Overlap"
            else:  # clt_ran
                y = 1.0 - np.prod(1.0 - cf, axis=-1)
                title_str = "Cloud Cover - Random Overlap"

            payload[field_str] = dict(
                field_key=field_key,
                ylabel="Total Cloud Fraction",
                title=f"{file_name_root} - {title_str}",
                y=y,
            )
        elif field_str in ["olr"]:
            field_key = "LW_flux_up_at_model_top"
            title_str = r"ToA Outgoing Longwave Radiation $\left[W\,m^{-2}\right]$"
    
            cmap = plt.colormaps["Blues"]
        
            y = xr_dpscream[field_key].sum(dim = "ncol").values.astype(np.float64, copy=False)

            payload[field_str] = dict(
                field_key=field_key,
                ylabel=title_str,
                title=None,
                y=y,
            )
        else:
            raise ValueError(f"Unknown plot key: {field_str}")

    log(comm, f"Prepared {len(payload)} time series on root.")
    return t, time_hours, file_name_root, payload


def render_and_save_plot(
    t: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: List[str]) -> int:
    comm: MPI.Comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(prog=prog_name, description=prog_desc)
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
        "--plots",
        action="store",
        nargs="*",
        type=str,
        default=["sza", "clt_max", "clt_ran", "clt_min", "olr"],
        help="Which plots to generate (subset of: sza clt_max clt_ran clt_min olr).",
    )
    parser.add_argument(
        "--log_every_rank",
        action="store_true",
        help="If set, all ranks print progress messages (can be noisy).",
    )

    args = parser.parse_args(argv[1:])

    dpscream_file_path: str = os.path.normpath(args.dpscream_file_path[0])
    plot_outdir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_viz_dir_path[0])
    plot_keys: List[str] = args.plots

    if rank == 0:
        os.makedirs(plot_outdir_path, exist_ok=True)

    comm.Barrier()
    log(comm, f"MPI size={size}. Output dir: {plot_outdir_path}")

    # Root loads + computes time series; then broadcast to all ranks
    t, time_hours, file_name_root, payload = compute_series_on_root(comm, dpscream_file_path, plot_keys)

    t = comm.bcast(t, root=0)
    file_name_root = comm.bcast(file_name_root, root=0)
    payload = comm.bcast(payload, root=0)

    # Each rank renders a subset of plots
    my_plots = partition_items(plot_keys, comm)
    log(comm, f"Assigned plots: {my_plots}", rank0_only=not args.log_every_rank)

    for field_str in my_plots:
        meta = payload[field_str]
        y = meta["y"]
        title = meta["title"]
        ylabel = meta["ylabel"]

        outname = f"{file_name_root}_{field_str}.png"
        outpath = os.path.join(plot_outdir_path, outname)

        log(comm, f"Rendering {field_str} -> {outname}", rank0_only=not args.log_every_rank)
        render_and_save_plot(
            t=t,
            y=y,
            title=title,
            xlabel="Time Step",
            ylabel=ylabel,
            outpath=outpath,
        )
        log(comm, f"Wrote: {outpath}", rank0_only=not args.log_every_rank)

    comm.Barrier()
    log(comm, "All plots complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))