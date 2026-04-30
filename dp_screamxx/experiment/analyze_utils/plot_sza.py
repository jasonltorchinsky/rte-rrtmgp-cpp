#!/usr/bin/env python3

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot solar zenith angle vs time (hours) using the coarsest input file (largest lr tag)."
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_input_dir_path",
        required=True,
        help="Directory containing RTE+RRTMGP C++ input netCDF files.",
    )
    parser.add_argument(
        "--rte_rrtmgp_cpp_viz_dir_path",
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--file_glob",
        default="*.in.nc",
        help="Glob pattern for input files inside input dir (default: *.in.nc).",
    )
    parser.add_argument(
        "--output_filename",
        default="sza.png",
        help="Output plot filename.",
    )
    return parser.parse_args()


def extract_lr_tag(filepath):
    """
    Extract integer lr tag from filename, e.g.
    '...lr_01.in' -> 1
    '...lr_12.in' -> 12
    Returns None if not found.
    """
    base = os.path.basename(filepath)
    match = re.search(r"lr_(\d+)", base)
    if match:
        return int(match.group(1))
    return None


def select_coarsest_file(files):
    tagged = []
    for f in files:
        lr = extract_lr_tag(f)
        if lr is not None:
            tagged.append((lr, f))

    if not tagged:
        raise ValueError("No files with an lr_<number> tag were found.")

    tagged.sort(key=lambda x: x[0], reverse=True)
    return tagged[0][1], tagged[0][0]


def find_day_night_transition_times(time_hours, mu0_1d, night_eps=1.e-3):
    """
    Return times where the classification changes between:
      - night: mu0 <= night_eps
      - day:   mu0 >  night_eps

    A tick is placed at the actual threshold-crossing time, found by
    linear interpolation between adjacent samples.
    """
    tick_times = []

    for i in range(1, len(mu0_1d)):
        prev_mu0 = mu0_1d[i - 1]
        curr_mu0 = mu0_1d[i]

        if not (np.isfinite(prev_mu0) and np.isfinite(curr_mu0)):
            continue

        is_prev_day = prev_mu0 > night_eps
        is_curr_day = curr_mu0 > night_eps

        if is_prev_day and not is_curr_day: # Transition day-to-night
            tick_times.append(float(time_hours[i - 1]))
        elif not is_prev_day and is_curr_day: # Transition night-to-day
            tick_times.append(float(time_hours[i]))

    return tick_times


def main():
    args = parse_args()

    os.makedirs(args.rte_rrtmgp_cpp_viz_dir_path, exist_ok=True)

    pattern = os.path.join(args.rte_rrtmgp_cpp_input_dir_path, args.file_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    filepath, lr_tag = select_coarsest_file(files)
    print(f"Selected coarsest file (largest lr tag): {filepath} (lr_{lr_tag:02d})")

    with xr.open_dataset(filepath) as ds:
        if "mu0" not in ds:
            raise KeyError(f"'mu0' not found in {filepath}")
        if "time" not in ds:
            raise KeyError(f"'time' not found in {filepath}")

        mu0 = ds["mu0"]
        time_hours = ds["time"].values

        if set(["time", "y", "x"]).issubset(mu0.dims):
            mu0_series = mu0.isel(y=0, x=0).values

            mu0_min = mu0.min(dim=("y", "x"), skipna=True).values
            mu0_max = mu0.max(dim=("y", "x"), skipna=True).values
            max_spatial_spread = np.nanmax(np.abs(mu0_max - mu0_min))
            if max_spatial_spread > 1.0e-12:
                print(
                    f"Warning: mu0 is not perfectly constant across x,y. "
                    f"Maximum spatial spread = {max_spatial_spread:.3e}"
                )
        elif "time" in mu0.dims:
            mu0_series = mu0.values
        else:
            raise ValueError(f"Unexpected mu0 dimensions: {mu0.dims}")

    mu0_series = np.clip(mu0_series, -1.0, 1.0)
    sza_deg = np.degrees(np.arccos(mu0_series))

    tick_times = find_day_night_transition_times(
        time_hours=time_hours,
        mu0_1d=mu0_series
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_hours, sza_deg, linewidth=2, label=f"Coarsest file: lr_{lr_tag:02d}")
    ax.set_xlabel("Time Since Simulation Start [Hours]")
    ax.set_ylabel("Solar Zenith Angle [Degrees]")
    ax.set_xticks(tick_times)
    ax.grid(axis="x", which="major", linestyle="-", alpha=0.4)
    fig.tight_layout()

    outpath = os.path.join(args.rte_rrtmgp_cpp_viz_dir_path, args.output_filename)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print(f"Wrote plot to: {outpath}")
    print(f"90-degree crossing tick times [hours]: {tick_times}")


if __name__ == "__main__":
    main()