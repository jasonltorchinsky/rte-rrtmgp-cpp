#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
import re
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    XR_DATAARRAY, \
    MPL_AXES, MPL_FIGURE, MPL_LINEAR_SEGMENTED_COLORMAP, MPL_LOGNORM, \
    MPL_COLORBAR
from consts.visual import diff_cmap, heating_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_z_max_info, find_grid, print_msg

# Script variables
prog_name: str = "animate-rte-rrtmgp-cpp-diorama-heating"
prog_desc: str = "Create an animation of atmospheric heating rates for RTE-RRTMGP-CPP."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    msg: str = "Parsing command-line input..."
    print_msg(msg)

    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory.")
    parser.add_argument("--rad-tran-outdir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT combined output directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate all quantities needed for plotting.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
    parser.add_argument("--frames", nargs = "?", default = 0, type = int,
        help = "Number of frames per day to include in the animation. If <= 0, use all frames.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None
    nframes_day_req: NP_INT = NP_INT(args.frames)

    coarse_factors: Optional[NP_ARRAY[NP_INT]] = None
    if args.coarse_factors is not None:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Find file pairs at requested resolutions
    #---------------------------------------------------------------------------
    rad_tran_infiles: list[str]
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(
        rad_tran_indir, rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain daytime time indices, times, SZAs, z_max_info
        #-----------------------------------------------------------------------
        msg: str = "Obtaining time index and z-max info..."
        print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(
            rad_tran_infile
        ) # [ndays, ndaytime]
        ndays: NP_INT = NP_INT(daytime_indices.shape[0])

        daytime_times: NP_ARRAY[NP_REAL] = find_times(
            rad_tran_infile,
            daytime_indices) # [h]; [ndays, ndaytime]
        daytime_szas: NP_ARRAY[NP_REAL] = find_szas(
            rad_tran_infile,
            daytime_indices) # [degrees]; [ndays, ndaytime]

        z_max_info: dict = calc_z_max_info(
            rad_tran_infile,
            z_max = z_max)

        time_indices_list: list[NP_ARRAY[NP_INT]] = []
        times_list: list[NP_ARRAY[NP_REAL]] = []
        szas_list: list[NP_ARRAY[NP_REAL]] = []
        day_frame_counts: list[int] = []

        jj: int
        for jj in range(0, ndays):
            day_indices_full: NP_ARRAY[NP_INT] = NP_INT(daytime_indices[jj,:])

            if (nframes_day_req is not None) and (nframes_day_req > 0) and (day_indices_full.size > nframes_day_req):
                sample_indices: NP_ARRAY[NP_INT] = NP_INT(
                    np.linspace(0, day_indices_full.size - 1, nframes_day_req, dtype = int)
                )
                day_indices: NP_ARRAY[NP_INT] = day_indices_full[sample_indices]
            else:
                day_indices = day_indices_full

            if day_indices.size < 1:
                day_indices = np.array([day_indices_full[0]], dtype = NP_INT)

            day_times_full: NP_ARRAY[NP_REAL] = NP_REAL(daytime_times[jj,:])
            day_szas_full: NP_ARRAY[NP_REAL] = NP_REAL(daytime_szas[jj,:])

            if (nframes_day_req is not None) and (nframes_day_req > 0) and (day_indices_full.size > nframes_day_req):
                sample_indices: NP_ARRAY[NP_INT] = NP_INT(
                    np.linspace(0, day_indices_full.size - 1, nframes_day_req, dtype = int)
                )
                day_indices: NP_ARRAY[NP_INT] = day_indices_full[sample_indices]
                day_times: NP_ARRAY[NP_REAL] = day_times_full[sample_indices]
                day_szas: NP_ARRAY[NP_REAL] = day_szas_full[sample_indices]
            else:
                day_indices = day_indices_full
                day_times = day_times_full
                day_szas = day_szas_full

            time_indices_list.append(day_indices.astype(NP_INT))
            times_list.append(day_times.astype(NP_REAL))
            szas_list.append(day_szas.astype(NP_REAL))
            day_frame_counts.append(int(day_indices.size))

        n_t_day_max: NP_INT = NP_INT(max(day_frame_counts))

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)
        grid: dict = find_grid(rad_tran_infile)

        # Rescale grids to have correct units
        xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
        yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]
        zh: XR_DATAARRAY = (grid["zh"]
            .sel(zh = z_max_info["sel_indexers"]["zh"])) * 1.e-3 # [m] => [km]

        # Get number of grid points
        n_xh: NP_INT = NP_INT(xh.size)
        n_yh: NP_INT = NP_INT(yh.size)
        n_zh: NP_INT = NP_INT(zh.size)

        n_x: NP_INT = n_xh - 1
        n_y: NP_INT = n_yh - 1
        n_z: NP_INT = n_zh - 1

        #-----------------------------------------------------------------------
        # Read cached plotting data or calculate and save them
        #-----------------------------------------------------------------------
        working_filename: str = "rte_rrtmgp_cpp_diorama_heating.animation.{}.nc".format(lr_str)
        working_filepath: str = os.path.join(working_dir, working_filename)

        need_recalculate: bool = recalculate or (not os.path.exists(working_filepath))

        if not need_recalculate:
            msg: str = "Checking cached plotting data..."
            print_msg(msg)

            ds_plot: Optional[xr.Dataset] = None
            try:
                ds_plot = xr.load_dataset(working_filepath)

                required_vars: list[str] = [
                    "filled",
                    "filled_diff",
                    "facecolors_ts",
                    "facecolors_rt",
                    "facecolors_diff",
                    "time_plot_values",
                    "sza_plot_values",
                    "day_frame_counts",
                    "xh",
                    "yh",
                    "zh"
                ]
                required_dims: list[str] = [
                    "day",
                    "time_plot",
                    "x_edge",
                    "y_edge",
                    "z_edge",
                    "x",
                    "y",
                    "z",
                    "rgba"
                ]

                has_required_vars: bool = all(var_name in ds_plot.variables for var_name in required_vars)
                has_required_dims: bool = all(dim_name in ds_plot.dims for dim_name in required_dims)
                has_required_attrs: bool = all(attr_name in ds_plot.attrs for attr_name in [
                    "heating_vmin",
                    "heating_vmax",
                    "heating_diff_linthresh",
                    "heating_diff_vmax"
                ])

                valid_shapes: bool = True
                if has_required_vars and has_required_dims:
                    valid_shapes = valid_shapes and (ds_plot["filled"].shape == (ndays, n_t_day_max, n_x, n_y, n_z))
                    valid_shapes = valid_shapes and (ds_plot["filled_diff"].shape == (ndays, n_t_day_max, n_x, n_y, n_z))
                    valid_shapes = valid_shapes and (ds_plot["facecolors_ts"].shape == (ndays, n_t_day_max, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["facecolors_rt"].shape == (ndays, n_t_day_max, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["facecolors_diff"].shape == (ndays, n_t_day_max, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["time_plot_values"].shape == (ndays, n_t_day_max))
                    valid_shapes = valid_shapes and (ds_plot["sza_plot_values"].shape == (ndays, n_t_day_max))
                    valid_shapes = valid_shapes and (ds_plot["day_frame_counts"].shape == (ndays,))
                    valid_shapes = valid_shapes and (ds_plot["xh"].size == n_xh)
                    valid_shapes = valid_shapes and (ds_plot["yh"].size == n_yh)
                    valid_shapes = valid_shapes and (ds_plot["zh"].size == n_zh)
                else:
                    valid_shapes = False

                valid_times: bool = True
                valid_szas: bool = True
                valid_counts: bool = True
                if has_required_vars and valid_shapes:
                    day_frame_counts_cache: NP_ARRAY[NP_INT] = NP_INT(ds_plot["day_frame_counts"].to_numpy())
                    valid_counts = np.array_equal(day_frame_counts_cache, NP_INT(np.array(day_frame_counts, dtype = NP_INT)))

                    for jj in range(0, ndays):
                        n_day_frames: int = day_frame_counts[jj]
                        valid_times = valid_times and np.array_equal(
                            NP_REAL(ds_plot["time_plot_values"].to_numpy()[jj,0:n_day_frames]),
                            NP_REAL(times_list[jj])
                        )
                        valid_szas = valid_szas and np.array_equal(
                            NP_REAL(ds_plot["sza_plot_values"].to_numpy()[jj,0:n_day_frames]),
                            NP_REAL(szas_list[jj])
                        )

                if has_required_vars and has_required_dims and valid_shapes and has_required_attrs and valid_times and valid_szas and valid_counts:
                    filled: NP_ARRAY[NP_BOOL] = NP_BOOL(ds_plot["filled"].to_numpy())
                    filled_diff: NP_ARRAY[NP_BOOL] = NP_BOOL(ds_plot["filled_diff"].to_numpy())
                    facecolors_ts: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors_ts"].to_numpy())
                    facecolors_rt: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors_rt"].to_numpy())
                    facecolors_diff: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors_diff"].to_numpy())
                    times_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["time_plot_values"].to_numpy())
                    szas_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["sza_plot_values"].to_numpy())
                    day_frame_counts_arr: NP_ARRAY[NP_INT] = NP_INT(ds_plot["day_frame_counts"].to_numpy())
                    xh_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["xh"].to_numpy())
                    yh_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["yh"].to_numpy())
                    zh_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["zh"].to_numpy())

                    min_heating: NP_REAL = NP_REAL(ds_plot.attrs["heating_vmin"])
                    max_heating: NP_REAL = NP_REAL(ds_plot.attrs["heating_vmax"])
                    heating_diff_linthresh: NP_REAL = NP_REAL(ds_plot.attrs["heating_diff_linthresh"])
                    max_heating_diff: NP_REAL = NP_REAL(ds_plot.attrs["heating_diff_vmax"])
                    heating_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_heating, vmax = max_heating)
                    heating_diff_colormap_norm = colors.SymLogNorm(
                        linthresh = heating_diff_linthresh,
                        vmin = -max_heating_diff,
                        vmax = max_heating_diff
                    )

                    use_cached: bool = True
                else:
                    use_cached = False

                ds_plot.close()

                if not use_cached:
                    need_recalculate = True

            except Exception:
                if ds_plot is not None:
                    ds_plot.close()
                need_recalculate = True

        if need_recalculate:
            msg: str = "Calculating plotting data..."
            print_msg(msg)

            #-------------------------------------------------------------------
            # Calculate CWC for voxel occupancy mask and heating rates
            #-------------------------------------------------------------------
            msg: str = "Obtaining cloud water content info..."
            print_msg(msg)

            cwc_tol: NP_REAL = NP_REAL(1.e-2)
            heating_diff_linthresh: NP_REAL = NP_REAL(1.0)

            filled: NP_ARRAY[NP_BOOL] = np.zeros(
                (ndays, n_t_day_max, n_x, n_y, n_z), dtype = NP_BOOL)
            heating_ts_vals: NP_ARRAY[NP_REAL] = np.zeros(
                (ndays, n_t_day_max, n_x, n_y, n_z), dtype = NP_REAL)
            heating_rt_vals: NP_ARRAY[NP_REAL] = np.zeros(
                (ndays, n_t_day_max, n_x, n_y, n_z), dtype = NP_REAL)
            heating_diff_vals: NP_ARRAY[NP_REAL] = np.zeros(
                (ndays, n_t_day_max, n_x, n_y, n_z), dtype = NP_REAL)
            times_plot: NP_ARRAY[NP_REAL] = np.full(
                (ndays, n_t_day_max), np.nan, dtype = NP_REAL)
            szas_plot: NP_ARRAY[NP_REAL] = np.full(
                (ndays, n_t_day_max), np.nan, dtype = NP_REAL)
            day_frame_counts_arr: NP_ARRAY[NP_INT] = NP_INT(np.array(day_frame_counts, dtype = NP_INT))

            jj: int
            for jj in range(0, ndays):
                n_day_frames: int = day_frame_counts[jj]

                cwc_day: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile,
                    time_indices = time_indices_list[jj],
                    z_max_info = z_max_info) # [time, lay, y, x]

                heating_ts_day: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = time_indices_list[jj],
                    z_max_info = z_max_info,
                    solver = "ts") # [time, lay, y, x]

                heating_rt_day: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = time_indices_list[jj],
                    z_max_info = z_max_info,
                    solver = "rt") # [time, lay, y, x]

                cwc_day = xr.where(cwc_day < cwc_tol, NP_REAL(0.), cwc_day)

                filled_day: NP_ARRAY[NP_BOOL] = NP_BOOL((cwc_day >= cwc_tol).to_numpy()) # [time, z, y, x]
                heating_ts_day_raw: NP_ARRAY[NP_REAL] = NP_REAL(heating_ts_day.to_numpy()) # [time, z, y, x]
                heating_rt_day_raw: NP_ARRAY[NP_REAL] = NP_REAL(heating_rt_day.to_numpy()) # [time, z, y, x]
                heating_ts_day_vals: NP_ARRAY[NP_REAL] = NP_REAL(np.abs(heating_ts_day_raw)) # [time, z, y, x]
                heating_rt_day_vals: NP_ARRAY[NP_REAL] = NP_REAL(np.abs(heating_rt_day_raw)) # [time, z, y, x]
                heating_diff_day_vals: NP_ARRAY[NP_REAL] = NP_REAL(heating_ts_day_raw - heating_rt_day_raw) # [time, z, y, x]

                filled[jj,0:n_day_frames,...] = np.transpose(filled_day, axes = (0, 3, 2, 1)) # [time, x, y, z]
                heating_ts_vals[jj,0:n_day_frames,...] = np.transpose(heating_ts_day_vals, axes = (0, 3, 2, 1)) # [time, x, y, z]
                heating_rt_vals[jj,0:n_day_frames,...] = np.transpose(heating_rt_day_vals, axes = (0, 3, 2, 1)) # [time, x, y, z]
                heating_diff_vals[jj,0:n_day_frames,...] = np.transpose(heating_diff_day_vals, axes = (0, 3, 2, 1)) # [time, x, y, z]
                times_plot[jj,0:n_day_frames] = NP_REAL(times_list[jj])
                szas_plot[jj,0:n_day_frames] = NP_REAL(szas_list[jj])

            filled_diff: NP_ARRAY[NP_BOOL] = NP_BOOL(
                filled & (np.abs(heating_diff_vals) >= heating_diff_linthresh)
            )

            max_heating: NP_REAL = NP_REAL(0.)
            for jj in range(0, ndays):
                n_day_frames: int = day_frame_counts[jj]
                max_heating = NP_REAL(max(
                    max_heating,
                    NP_REAL(np.nanmax(heating_ts_vals[jj,0:n_day_frames,...])),
                    NP_REAL(np.nanmax(heating_rt_vals[jj,0:n_day_frames,...]))
                ))

            min_heating: NP_REAL = NP_REAL(1.e-2)

            if max_heating <= min_heating:
                max_heating = NP_REAL(min_heating * NP_REAL(10.))

            if np.any(filled_diff):
                max_heating_diff: NP_REAL = NP_REAL(np.nanmax(np.abs(heating_diff_vals[filled_diff])))
            else:
                max_heating_diff = NP_REAL(0.)
                for jj in range(0, ndays):
                    n_day_frames: int = day_frame_counts[jj]
                    max_heating_diff = NP_REAL(max(
                        max_heating_diff,
                        NP_REAL(np.nanmax(np.abs(heating_diff_vals[jj,0:n_day_frames,...])))
                    ))

            if (not np.isfinite(max_heating_diff)) or (max_heating_diff <= heating_diff_linthresh):
                max_heating_diff = NP_REAL(heating_diff_linthresh * NP_REAL(10.))

            heating_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[heating_cmap]
            heating_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_heating, vmax = max_heating)

            diff_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[diff_cmap]
            heating_diff_colormap_norm = colors.SymLogNorm(
                linthresh = heating_diff_linthresh,
                vmin = -max_heating_diff,
                vmax = max_heating_diff
            )

            heating_ts_norm: NP_ARRAY[NP_REAL] = NP_REAL(
                heating_colormap_norm(
                    np.clip(
                        heating_ts_vals,
                        min_heating,
                        max_heating
                    ).flatten()
                ).reshape(heating_ts_vals.shape)
            )

            heating_rt_norm: NP_ARRAY[NP_REAL] = NP_REAL(
                heating_colormap_norm(
                    np.clip(
                        heating_rt_vals,
                        min_heating,
                        max_heating
                    ).flatten()
                ).reshape(heating_rt_vals.shape)
            )

            heating_diff_norm: NP_ARRAY[NP_REAL] = NP_REAL(
                heating_diff_colormap_norm(
                    np.clip(
                        heating_diff_vals,
                        -max_heating_diff,
                        max_heating_diff
                    ).flatten()
                ).reshape(heating_diff_vals.shape)
            )

            facecolors_ts: NP_ARRAY[NP_REAL] = NP_REAL(heating_colormap(heating_ts_norm))
            facecolors_rt: NP_ARRAY[NP_REAL] = NP_REAL(heating_colormap(heating_rt_norm))
            facecolors_diff: NP_ARRAY[NP_REAL] = NP_REAL(diff_colormap(heating_diff_norm))

            # Set alpha
            heating_all_vals_list: list[NP_ARRAY[NP_REAL]] = []
            for jj in range(0, ndays):
                n_day_frames: int = day_frame_counts[jj]
                heating_all_vals_list.append(heating_ts_vals[jj,0:n_day_frames,...].reshape(-1))
                heating_all_vals_list.append(heating_rt_vals[jj,0:n_day_frames,...].reshape(-1))
            heating_all_vals: NP_ARRAY[NP_REAL] = np.concatenate(heating_all_vals_list)

            heating_min_alpha: NP_REAL = NP_REAL(np.nanmin(heating_all_vals))
            heating_max_alpha: NP_REAL = NP_REAL(np.nanmax(heating_all_vals))
            alpha_min: NP_REAL = NP_REAL(0.1)
            alpha_max: NP_REAL = NP_REAL(0.5)

            if heating_max_alpha > heating_min_alpha:
                alpha_ts: NP_ARRAY[NP_REAL] = NP_REAL(
                    ((alpha_max - alpha_min)
                        * (heating_ts_vals - heating_min_alpha)
                        / (heating_max_alpha - heating_min_alpha)) + alpha_min
                )
                alpha_rt: NP_ARRAY[NP_REAL] = NP_REAL(
                    ((alpha_max - alpha_min)
                        * (heating_rt_vals - heating_min_alpha)
                        / (heating_max_alpha - heating_min_alpha)) + alpha_min
                )
                alpha_ts = NP_REAL(np.clip(alpha_ts, alpha_min, alpha_max))
                alpha_rt = NP_REAL(np.clip(alpha_rt, alpha_min, alpha_max))
            else:
                alpha_ts = NP_REAL(alpha_min * np.ones(heating_ts_vals.shape, dtype = NP_REAL))
                alpha_rt = NP_REAL(alpha_min * np.ones(heating_rt_vals.shape, dtype = NP_REAL))

            heating_diff_abs_vals: NP_ARRAY[NP_REAL] = NP_REAL(np.abs(heating_diff_vals))
            if max_heating_diff > heating_diff_linthresh:
                alpha_diff: NP_ARRAY[NP_REAL] = NP_REAL(
                    ((alpha_max - alpha_min)
                        * (heating_diff_abs_vals - heating_diff_linthresh)
                        / (max_heating_diff - heating_diff_linthresh)) + alpha_min
                )
                alpha_diff = NP_REAL(np.clip(alpha_diff, alpha_min, alpha_max))
            else:
                alpha_diff = NP_REAL(alpha_min * np.ones(heating_diff_vals.shape, dtype = NP_REAL))

            facecolors_ts[...,3] = alpha_ts
            facecolors_rt[...,3] = alpha_rt
            facecolors_diff[...,3] = alpha_diff

            xh_plot: NP_ARRAY[NP_REAL] = NP_REAL(xh.to_numpy())
            yh_plot: NP_ARRAY[NP_REAL] = NP_REAL(yh.to_numpy())
            zh_plot: NP_ARRAY[NP_REAL] = NP_REAL(zh.to_numpy())

            msg: str = "Saving plotting data to file..."
            print_msg(msg)

            ds_plot: xr.Dataset = xr.Dataset(
                data_vars = {
                    "filled" : (
                        ("day", "time_plot", "x", "y", "z"),
                        filled.astype(np.int8)
                    ),
                    "filled_diff" : (
                        ("day", "time_plot", "x", "y", "z"),
                        filled_diff.astype(np.int8)
                    ),
                    "facecolors_ts" : (
                        ("day", "time_plot", "x", "y", "z", "rgba"),
                        facecolors_ts.astype(NP_REAL)
                    ),
                    "facecolors_rt" : (
                        ("day", "time_plot", "x", "y", "z", "rgba"),
                        facecolors_rt.astype(NP_REAL)
                    ),
                    "facecolors_diff" : (
                        ("day", "time_plot", "x", "y", "z", "rgba"),
                        facecolors_diff.astype(NP_REAL)
                    ),
                    "time_plot_values" : (
                        ("day", "time_plot"),
                        times_plot.astype(NP_REAL)
                    ),
                    "sza_plot_values" : (
                        ("day", "time_plot"),
                        szas_plot.astype(NP_REAL)
                    ),
                    "day_frame_counts" : (
                        ("day",),
                        day_frame_counts_arr.astype(NP_INT)
                    )
                },
                coords = {
                    "day" : np.arange(ndays, dtype = NP_INT),
                    "time_plot" : np.arange(n_t_day_max, dtype = NP_INT),
                    "xh" : (("x_edge",), xh_plot),
                    "yh" : (("y_edge",), yh_plot),
                    "zh" : (("z_edge",), zh_plot),
                    "x" : np.arange(n_x, dtype = NP_INT),
                    "y" : np.arange(n_y, dtype = NP_INT),
                    "z" : np.arange(n_z, dtype = NP_INT),
                    "x_edge" : np.arange(n_xh, dtype = NP_INT),
                    "y_edge" : np.arange(n_yh, dtype = NP_INT),
                    "z_edge" : np.arange(n_zh, dtype = NP_INT),
                    "rgba" : np.arange(4, dtype = NP_INT)
                },
                attrs = {
                    "lr_str" : lr_str,
                    "heating_cmap" : heating_cmap,
                    "diff_cmap" : diff_cmap,
                    "z_max_km" : float(zh_plot.max()) if zh_plot.size > 0 else 0.0,
                    "heating_vmin" : float(min_heating),
                    "heating_vmax" : float(max_heating),
                    "heating_diff_linthresh" : float(heating_diff_linthresh),
                    "heating_diff_vmax" : float(max_heating_diff)
                }
            )
            ds_plot["filled"].attrs["description"] = "Voxel occupancy mask from cloud water content threshold"
            ds_plot["filled_diff"].attrs["description"] = "Voxel occupancy mask from cloud water content threshold and heating-rate difference threshold"
            ds_plot["facecolors_ts"].attrs["description"] = "RGBA voxel face colors for two-stream heating"
            ds_plot["facecolors_rt"].attrs["description"] = "RGBA voxel face colors for ray-tracer heating"
            ds_plot["facecolors_diff"].attrs["description"] = "RGBA voxel face colors for two-stream minus ray-tracer heating difference"
            ds_plot["time_plot_values"].attrs["units"] = "h"
            ds_plot["sza_plot_values"].attrs["units"] = "degrees"
            ds_plot["day_frame_counts"].attrs["description"] = "Number of valid animation frames for each day"
            ds_plot["xh"].attrs["units"] = "km"
            ds_plot["yh"].attrs["units"] = "km"
            ds_plot["zh"].attrs["units"] = "km"

            ds_plot.to_netcdf(working_filepath)
            ds_plot.close()

        if need_recalculate:
            filled = NP_BOOL(filled)
            filled_diff = NP_BOOL(filled_diff)
            facecolors_ts = NP_REAL(facecolors_ts)
            facecolors_rt = NP_REAL(facecolors_rt)
            facecolors_diff = NP_REAL(facecolors_diff)
            times_plot = NP_REAL(times_plot)
            szas_plot = NP_REAL(szas_plot)
            day_frame_counts_arr = NP_INT(day_frame_counts_arr)
            xh_plot = NP_REAL(xh_plot)
            yh_plot = NP_REAL(yh_plot)
            zh_plot = NP_REAL(zh_plot)

        heating_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[heating_cmap]
        diff_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[diff_cmap]
        heating_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_heating, vmax = max_heating)
        heating_diff_colormap_norm = colors.SymLogNorm(
            linthresh = heating_diff_linthresh,
            vmin = -max_heating_diff,
            vmax = max_heating_diff
        )

        #-----------------------------------------------------------------------
        # Build animation frame index with 1-second title cards before each day
        #-----------------------------------------------------------------------
        msg: str = "Building animation frame sequence..."
        print_msg(msg)

        fps: NP_INT = NP_INT(8)
        titlecard_frames: NP_INT = NP_INT(fps)

        animation_day_list: list[int] = []
        animation_time_list: list[int] = []
        animation_is_titlecard_list: list[bool] = []

        for jj in range(0, ndays):
            for _ in range(0, int(titlecard_frames)):
                animation_day_list.append(jj)
                animation_time_list.append(0)
                animation_is_titlecard_list.append(True)

            n_day_frames: int = int(day_frame_counts_arr[jj])
            kk: int
            for kk in range(0, n_day_frames):
                animation_day_list.append(jj)
                animation_time_list.append(kk)
                animation_is_titlecard_list.append(False)

        animation_day: NP_ARRAY[NP_INT] = np.array(animation_day_list, dtype = NP_INT)
        animation_time: NP_ARRAY[NP_INT] = np.array(animation_time_list, dtype = NP_INT)
        animation_is_titlecard: NP_ARRAY[NP_BOOL] = np.array(animation_is_titlecard_list, dtype = NP_BOOL)
        n_anim_frames: NP_INT = NP_INT(animation_day.size)

        #-----------------------------------------------------------------------
        # Set up the figure
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        nrows: NP_INT = NP_INT(3)
        ncols: NP_INT = NP_INT(1)
        fig_height: NP_REAL = NP_REAL(8.5)
        fig_width: NP_REAL = NP_REAL(6.8)
        fig: MPL_FIGURE
        axs: NP_ARRAY[MPL_AXES]
        fig, axs = plt.subplots(
            nrows = nrows, ncols = ncols,
            sharex = False, sharey = False,
            constrained_layout = False,
            figsize = (fig_width, fig_height),
            subplot_kw = {"projection" : "3d"})

        if ncols == 1:
            axs = axs[...,None]
        elif nrows == 1:
            axs = axs[None,...]

        fig.subplots_adjust(
            left = 0.00,
            right = 0.80,
            bottom = 0.02,
            top = 0.95,
            wspace = 0.02,
            hspace = -0.10
        )

        heating_cax = fig.add_axes([0.88, 0.35, 0.025, 0.55])
        heating_diff_cax = fig.add_axes([0.88, 0.08, 0.025, 0.18])

        #-----------------------------------------------------------------------
        # Set up colorbars
        #-----------------------------------------------------------------------
        msg: str = "Setting up colorbar..."
        print_msg(msg)

        heating_colorbar: MPL_COLORBAR = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm = heating_colormap_norm,
                cmap = heating_colormap
            ),
            cax = heating_cax
        )
        heating_colorbar.ax.set_yscale("log")

        heating_diff_colorbar: MPL_COLORBAR = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm = heating_diff_colormap_norm,
                cmap = diff_colormap
            ),
            cax = heating_diff_cax
        )
        
        heating_diff_colorbar.ax.axhline(
            heating_diff_linthresh,
            color = "k",
            linestyle = "solid",
            linewidth = 1.0
        )
        heating_diff_colorbar.ax.axhline(
            -heating_diff_linthresh,
            color = "k",
            linestyle = "dashed",
            linewidth = 1.0
        )

        #-----------------------------------------------------------------------
        # Set style elements
        #-----------------------------------------------------------------------
        msg: str = "Setting style elements..."
        print_msg(msg)

        pane_color: list[float] = [0.9, 0.9, 0.9, 0.0]
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.xaxis.set_pane_color(pane_color)
            ax.yaxis.set_pane_color(pane_color)
            ax.zaxis.set_pane_color(pane_color)

        for ax in axs.flatten():
            ax.xaxis._axinfo["grid"]["linewidth"] = 0
            ax.yaxis._axinfo["grid"]["linewidth"] = 0
            ax.zaxis._axinfo["grid"]["linewidth"] = 0

        xh_len: NP_REAL = NP_REAL(xh_plot.max() - xh_plot.min())
        yh_len: NP_REAL = NP_REAL(yh_plot.max() - yh_plot.min())
        zh_len: NP_REAL = NP_REAL(zh_plot.max() - zh_plot.min())
        for ax in axs.flatten():
            ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

        x_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(xh_plot.min(), xh_plot.max()))
        y_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(yh_plot.min(), yh_plot.max()))
        z_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 2).tick_values(0., np.floor(zh_plot.max())))
        for ax in axs.flatten():
            ax.xaxis.set_ticks(x_ticks)
            ax.yaxis.set_ticks(y_ticks)
            ax.zaxis.set_ticks(z_ticks)

        axs[0,0].xaxis.set_tick_params(labelcolor = "none")
        axs[0,0].yaxis.set_tick_params(labelcolor = "none")
        axs[1,0].xaxis.set_tick_params(labelcolor = "none")
        axs[1,0].yaxis.set_tick_params(labelcolor = "none")
        axs[2,0].set_ylabel(r"y $\left[ km \right]$")
        for ax in axs[:,0].flatten():
            ax.set_zlabel(r"z $\left[ km \right]$")
        axs[2,0].set_xlabel(r"x $\left[ km \right]$")

        axs[0,0].text2D(
            -0.12, 0.50,
            r"Two-Stream",
            transform = axs[0,0].transAxes,
            rotation = 90,
            ha = "center",
            va = "center"
        )
        axs[1,0].text2D(
            -0.12, 0.50,
            r"Ray-Tracer",
            transform = axs[1,0].transAxes,
            rotation = 90,
            ha = "center",
            va = "center"
        )
        axs[2,0].text2D(
            -0.12, 0.50,
            r"Two-Stream - Ray-Tracer",
            transform = axs[2,0].transAxes,
            rotation = 90,
            ha = "center",
            va = "center"
        )

        dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

        fig.suptitle(
            r"Heating Rate $\left[ K\,d^{-1} \right]$" + " - {}".format(dx_str)
        )
        fig.supxlabel("")

        frame_text = fig.text(
            0.5, 0.94,
            "",
            ha = "center",
            va = "top"
        )

        titlecard_text = fig.text(
            0.5, 0.5,
            "",
            ha = "center",
            va = "center"
        )

        x0: NP_ARRAY[NP_REAL] = xh_plot[:-1]
        x1: NP_ARRAY[NP_REAL] = xh_plot[1:]
        y0: NP_ARRAY[NP_REAL] = yh_plot[:-1]
        y1: NP_ARRAY[NP_REAL] = yh_plot[1:]
        z0: NP_ARRAY[NP_REAL] = zh_plot[:-1]
        z1: NP_ARRAY[NP_REAL] = zh_plot[1:]

        current_poly_ts: Optional[Poly3DCollection] = None
        current_poly_rt: Optional[Poly3DCollection] = None
        current_poly_diff: Optional[Poly3DCollection] = None

        #-----------------------------------------------------------------------
        # Create animation
        #-----------------------------------------------------------------------
        msg: str = "Creating animation..."
        print_msg(msg)

        def update(i_anim: int):
            nonlocal current_poly_ts, current_poly_rt, current_poly_diff

            jj: int = int(animation_day[i_anim])
            kk: int = int(animation_time[i_anim])

            if current_poly_ts is not None:
                current_poly_ts.remove()
                current_poly_ts = None
            if current_poly_rt is not None:
                current_poly_rt.remove()
                current_poly_rt = None
            if current_poly_diff is not None:
                current_poly_diff.remove()
                current_poly_diff = None

            if animation_is_titlecard[i_anim]:
                for ax in axs.flatten():
                    ax.set_axis_off()
                titlecard_text.set_text("Day {}".format(jj))
                frame_text.set_text("")
                return [titlecard_text, frame_text]

            for ax in axs.flatten():
                ax.set_axis_on()

            titlecard_text.set_text("")

            for ax in axs.flatten():
                ax.xaxis.set_pane_color(pane_color)
                ax.yaxis.set_pane_color(pane_color)
                ax.zaxis.set_pane_color(pane_color)

            for ax in axs.flatten():
                ax.xaxis._axinfo["grid"]["linewidth"] = 0
                ax.yaxis._axinfo["grid"]["linewidth"] = 0
                ax.zaxis._axinfo["grid"]["linewidth"] = 0

            for ax in axs.flatten():
                ax.xaxis.set_ticks(x_ticks)
                ax.yaxis.set_ticks(y_ticks)
                ax.zaxis.set_ticks(z_ticks)

            axs[0,0].xaxis.set_tick_params(labelcolor = "none")
            axs[0,0].yaxis.set_tick_params(labelcolor = "none")
            axs[1,0].xaxis.set_tick_params(labelcolor = "none")
            axs[1,0].yaxis.set_tick_params(labelcolor = "none")
            axs[2,0].set_ylabel(r"y $\left[ km \right]$")
            for ax in axs[:,0].flatten():
                ax.set_zlabel(r"z $\left[ km \right]$")
            axs[2,0].set_xlabel(r"x $\left[ km \right]$")

            filled_i: NP_ARRAY[NP_BOOL] = filled[jj,kk,...]
            if np.any(filled_i):
                i_x: NP_ARRAY[NP_INT]
                i_y: NP_ARRAY[NP_INT]
                i_z: NP_ARRAY[NP_INT]
                i_x, i_y, i_z = np.where(filled_i)

                n_vox: int = int(i_x.size)

                xv0: NP_ARRAY[NP_REAL] = x0[i_x]
                xv1: NP_ARRAY[NP_REAL] = x1[i_x]
                yv0: NP_ARRAY[NP_REAL] = y0[i_y]
                yv1: NP_ARRAY[NP_REAL] = y1[i_y]
                zv0: NP_ARRAY[NP_REAL] = z0[i_z]
                zv1: NP_ARRAY[NP_REAL] = z1[i_z]

                verts: NP_ARRAY[NP_REAL] = np.empty((6 * n_vox, 4, 3), dtype = NP_REAL)

                verts[0::6,0,:] = np.stack((xv0, yv0, zv0), axis = 1)
                verts[0::6,1,:] = np.stack((xv1, yv0, zv0), axis = 1)
                verts[0::6,2,:] = np.stack((xv1, yv1, zv0), axis = 1)
                verts[0::6,3,:] = np.stack((xv0, yv1, zv0), axis = 1)

                verts[1::6,0,:] = np.stack((xv0, yv0, zv1), axis = 1)
                verts[1::6,1,:] = np.stack((xv1, yv0, zv1), axis = 1)
                verts[1::6,2,:] = np.stack((xv1, yv1, zv1), axis = 1)
                verts[1::6,3,:] = np.stack((xv0, yv1, zv1), axis = 1)

                verts[2::6,0,:] = np.stack((xv0, yv0, zv0), axis = 1)
                verts[2::6,1,:] = np.stack((xv1, yv0, zv0), axis = 1)
                verts[2::6,2,:] = np.stack((xv1, yv0, zv1), axis = 1)
                verts[2::6,3,:] = np.stack((xv0, yv0, zv1), axis = 1)

                verts[3::6,0,:] = np.stack((xv0, yv1, zv0), axis = 1)
                verts[3::6,1,:] = np.stack((xv1, yv1, zv0), axis = 1)
                verts[3::6,2,:] = np.stack((xv1, yv1, zv1), axis = 1)
                verts[3::6,3,:] = np.stack((xv0, yv1, zv1), axis = 1)

                verts[4::6,0,:] = np.stack((xv0, yv0, zv0), axis = 1)
                verts[4::6,1,:] = np.stack((xv0, yv1, zv0), axis = 1)
                verts[4::6,2,:] = np.stack((xv0, yv1, zv1), axis = 1)
                verts[4::6,3,:] = np.stack((xv0, yv0, zv1), axis = 1)

                verts[5::6,0,:] = np.stack((xv1, yv0, zv0), axis = 1)
                verts[5::6,1,:] = np.stack((xv1, yv1, zv0), axis = 1)
                verts[5::6,2,:] = np.stack((xv1, yv1, zv1), axis = 1)
                verts[5::6,3,:] = np.stack((xv1, yv0, zv1), axis = 1)

                voxel_facecolors_ts: NP_ARRAY[NP_REAL] = facecolors_ts[jj, kk, i_x, i_y, i_z, :]
                poly_facecolors_ts: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors_ts, 6, axis = 0)

                current_poly_ts = Poly3DCollection(
                    verts,
                    facecolors = poly_facecolors_ts,
                    edgecolors = poly_facecolors_ts,
                    linewidths = 0
                )
                axs[0,0].add_collection3d(current_poly_ts)

                voxel_facecolors_rt: NP_ARRAY[NP_REAL] = facecolors_rt[jj, kk, i_x, i_y, i_z, :]
                poly_facecolors_rt: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors_rt, 6, axis = 0)

                current_poly_rt = Poly3DCollection(
                    verts,
                    facecolors = poly_facecolors_rt,
                    edgecolors = poly_facecolors_rt,
                    linewidths = 0
                )
                axs[1,0].add_collection3d(current_poly_rt)

                filled_diff_i: NP_ARRAY[NP_BOOL] = filled_diff[jj,kk,i_x,i_y,i_z]
                if np.any(filled_diff_i):
                    verts_diff: NP_ARRAY[NP_REAL] = verts.reshape(n_vox, 6, 4, 3)[filled_diff_i,...].reshape(-1, 4, 3)

                    voxel_facecolors_diff: NP_ARRAY[NP_REAL] = facecolors_diff[
                        jj, kk, i_x[filled_diff_i], i_y[filled_diff_i], i_z[filled_diff_i], :]
                    poly_facecolors_diff: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors_diff, 6, axis = 0)

                    current_poly_diff = Poly3DCollection(
                        verts_diff,
                        facecolors = poly_facecolors_diff,
                        edgecolors = poly_facecolors_diff,
                        linewidths = 0
                    )
                    axs[2,0].add_collection3d(current_poly_diff)

            for ax in axs.flatten():
                ax.set_xlim(xh_plot.min(), xh_plot.max())
                ax.set_ylim(yh_plot.min(), yh_plot.max())
                ax.set_zlim(zh_plot.min(), zh_plot.max())

            frame_text.set_text(
                r"{:.2f} $h$   -   SZA {:.1f}$^{{\circ}}$".format(
                    times_plot[jj,kk],
                    szas_plot[jj,kk]
                )
            )

            artists: list = [titlecard_text, frame_text]
            if current_poly_ts is not None:
                artists.append(current_poly_ts)
            if current_poly_rt is not None:
                artists.append(current_poly_rt)
            if current_poly_diff is not None:
                artists.append(current_poly_diff)

            return artists

        anim = animation.FuncAnimation(
            fig,
            update,
            frames = int(n_anim_frames),
            interval = float(1000.0 / fps),
            blit = False,
            repeat = False
        )

        #-----------------------------------------------------------------------
        # Save the animation to file
        #-----------------------------------------------------------------------
        msg: str = "Saving animation to file..."
        print_msg(msg)

        plt_filename = "rte_rrtmgp_cpp_diorama_heating.{}.mp4".format(lr_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        anim.save(
            plt_filepath,
            writer = animation.FFMpegWriter(fps = int(fps))
        )
        plt.close(fig)

if __name__ == "__main__":
    main()