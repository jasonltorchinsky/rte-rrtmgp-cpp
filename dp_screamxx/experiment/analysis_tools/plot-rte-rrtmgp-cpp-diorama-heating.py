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
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_z_max_info, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-diorama-heating"
prog_desc: str = "Create a diorama of atmospheric heating rates for RTE-RRTMGP-CPP."

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
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

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
        # Obtain Morning-Noon-Night time indices, times, SZAs, z_max_info
        #-----------------------------------------------------------------------
        msg: str = "Obtaining time index and z-max info..."
        print_msg(msg)

        mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(
            rad_tran_infile
        ) # [ndays, 3]
        mnn_times: NP_ARRAY[NP_REAL] = find_times(
            rad_tran_infile,
            mnn_indices) # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = find_szas(
            rad_tran_infile,
            mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(mnn_indices.shape[0])
        z_max_info: dict = calc_z_max_info(
            rad_tran_infile,
            z_max = z_max)

        time_indices: NP_ARRAY[NP_INT] = mnn_indices[:,0:2]
        times: NP_ARRAY[NP_REAL] = mnn_times[:,0:2]
        szas: NP_ARRAY[NP_REAL] = mnn_szas[:,0:2]

        n_t_day: NP_INT = NP_INT(time_indices.shape[1])

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
        working_filename: str = "rte_rrtmgp_cpp_diorama_heating.{}.nc".format(lr_str)
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
                    valid_shapes = valid_shapes and (ds_plot["filled"].shape == (ndays, n_t_day, n_x, n_y, n_z))
                    valid_shapes = valid_shapes and (ds_plot["filled_diff"].shape == (ndays, n_t_day, n_x, n_y, n_z))
                    valid_shapes = valid_shapes and (ds_plot["facecolors_ts"].shape == (ndays, n_t_day, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["facecolors_rt"].shape == (ndays, n_t_day, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["facecolors_diff"].shape == (ndays, n_t_day, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["xh"].size == n_xh)
                    valid_shapes = valid_shapes and (ds_plot["yh"].size == n_yh)
                    valid_shapes = valid_shapes and (ds_plot["zh"].size == n_zh)
                    valid_shapes = valid_shapes and (ds_plot["time_plot_values"].shape == (ndays, n_t_day))
                    valid_shapes = valid_shapes and (ds_plot["sza_plot_values"].shape == (ndays, n_t_day))
                else:
                    valid_shapes = False

                if has_required_vars and has_required_dims and valid_shapes and has_required_attrs:
                    filled: NP_ARRAY[NP_BOOL] = NP_BOOL(ds_plot["filled"].to_numpy())
                    filled_diff: NP_ARRAY[NP_BOOL] = NP_BOOL(ds_plot["filled_diff"].to_numpy())
                    facecolors_ts: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors_ts"].to_numpy())
                    facecolors_rt: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors_rt"].to_numpy())
                    facecolors_diff: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors_diff"].to_numpy())
                    times_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["time_plot_values"].to_numpy())
                    szas_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["sza_plot_values"].to_numpy())
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
                    use_cached: bool = False

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
                (ndays, n_t_day, n_x, n_y, n_z), dtype = NP_BOOL)
            heating_ts_vals: NP_ARRAY[NP_REAL] = np.zeros(
                (ndays, n_t_day, n_x, n_y, n_z), dtype = NP_REAL)
            heating_rt_vals: NP_ARRAY[NP_REAL] = np.zeros(
                (ndays, n_t_day, n_x, n_y, n_z), dtype = NP_REAL)
            heating_diff_vals: NP_ARRAY[NP_REAL] = np.zeros(
                (ndays, n_t_day, n_x, n_y, n_z), dtype = NP_REAL)

            jj: int
            for jj in range(0, ndays):
                cwc_day: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile,
                    time_indices = time_indices[jj,:],
                    z_max_info = z_max_info) # [time, lay, y, x]

                heating_ts_day: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = time_indices[jj,:],
                    z_max_info = z_max_info,
                    solver = "ts") # [time, lay, y, x]

                heating_rt_day: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = time_indices[jj,:],
                    z_max_info = z_max_info,
                    solver = "rt") # [time, lay, y, x]

                cwc_day = xr.where(cwc_day < cwc_tol, NP_REAL(0.), cwc_day)

                filled_day: NP_ARRAY[NP_BOOL] = NP_BOOL((cwc_day >= cwc_tol).to_numpy()) # [time, z, y, x]
                heating_ts_day_raw: NP_ARRAY[NP_REAL] = NP_REAL(heating_ts_day.to_numpy()) # [time, z, y, x]
                heating_rt_day_raw: NP_ARRAY[NP_REAL] = NP_REAL(heating_rt_day.to_numpy()) # [time, z, y, x]
                heating_ts_day_vals: NP_ARRAY[NP_REAL] = NP_REAL(np.abs(heating_ts_day_raw)) # [time, z, y, x]
                heating_rt_day_vals: NP_ARRAY[NP_REAL] = NP_REAL(np.abs(heating_rt_day_raw)) # [time, z, y, x]
                heating_diff_day_vals: NP_ARRAY[NP_REAL] = NP_REAL(heating_ts_day_raw - heating_rt_day_raw) # [time, z, y, x]

                filled[jj,...] = np.transpose(filled_day, axes = (0, 3, 2, 1)) # [time, x, y, z]
                heating_ts_vals[jj,...] = np.transpose(heating_ts_day_vals, axes = (0, 3, 2, 1)) # [time, x, y, z]
                heating_rt_vals[jj,...] = np.transpose(heating_rt_day_vals, axes = (0, 3, 2, 1)) # [time, x, y, z]
                heating_diff_vals[jj,...] = np.transpose(heating_diff_day_vals, axes = (0, 3, 2, 1)) # [time, x, y, z]

            filled_diff: NP_ARRAY[NP_BOOL] = NP_BOOL(
                filled & (np.abs(heating_diff_vals) >= heating_diff_linthresh)
            )

            max_heating: NP_REAL = NP_REAL(max(
                NP_REAL(np.nanmax(heating_ts_vals)),
                NP_REAL(np.nanmax(heating_rt_vals))
            ))
            min_heating: NP_REAL = NP_REAL(1.e-2)

            if max_heating <= min_heating:
                max_heating = NP_REAL(min_heating * NP_REAL(10.))

            if np.any(filled):
                max_heating_diff: NP_REAL = NP_REAL(np.nanmax(np.abs(heating_diff_vals[filled])))
            else:
                max_heating_diff = NP_REAL(np.nanmax(np.abs(heating_diff_vals)))

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
            heating_all_vals: NP_ARRAY[NP_REAL] = np.concatenate(
                [heating_ts_vals.reshape(-1), heating_rt_vals.reshape(-1)]
            )
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
            times_plot: NP_ARRAY[NP_REAL] = NP_REAL(times)
            szas_plot: NP_ARRAY[NP_REAL] = NP_REAL(szas)

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
                    )
                },
                coords = {
                    "day" : np.arange(ndays, dtype = NP_INT),
                    "time_plot" : np.arange(n_t_day, dtype = NP_INT),
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
            ds_plot["xh"].attrs["units"] = "km"
            ds_plot["yh"].attrs["units"] = "km"
            ds_plot["zh"].attrs["units"] = "km"
            ds_plot["time_plot_values"].attrs["units"] = "h"
            ds_plot["sza_plot_values"].attrs["units"] = "degrees"

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
            xh_plot = NP_REAL(xh_plot)
            yh_plot = NP_REAL(yh_plot)
            zh_plot = NP_REAL(zh_plot)

        heating_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_heating, vmax = max_heating)
        heating_diff_colormap_norm = colors.SymLogNorm(
            linthresh = heating_diff_linthresh,
            vmin = -max_heating_diff,
            vmax = max_heating_diff
        )

        #-----------------------------------------------------------------------
        # Plot the data for each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            #-------------------------------------------------------------------
            # Set up the figure
            #-------------------------------------------------------------------
            msg: str = "Setting up figure for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            nrows: NP_INT = NP_INT(3)
            ncols: NP_INT = NP_INT(n_t_day)
            fig_height: NP_REAL = NP_REAL(6.0)
            fig_width: NP_REAL = NP_REAL(6.5)
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
                left = 0.04,
                right = 0.78,
                bottom = 0.00,
                top = 0.94,
                wspace = 0.04,
                hspace = -0.16
            )

            heating_cax = fig.add_axes([0.90, 0.35, 0.02, 0.55])
            heating_diff_cax = fig.add_axes([0.90, 0.06, 0.02, 0.20])

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting the data for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            heating_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[heating_cmap]
            diff_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[diff_cmap]

            x0: NP_ARRAY[NP_REAL] = xh_plot[:-1]
            x1: NP_ARRAY[NP_REAL] = xh_plot[1:]
            y0: NP_ARRAY[NP_REAL] = yh_plot[:-1]
            y1: NP_ARRAY[NP_REAL] = yh_plot[1:]
            z0: NP_ARRAY[NP_REAL] = zh_plot[:-1]
            z1: NP_ARRAY[NP_REAL] = zh_plot[1:]

            kk: int
            for kk in range(0, ncols):
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

                    poly_ts: Poly3DCollection = Poly3DCollection(
                        verts,
                        facecolors = poly_facecolors_ts,
                        edgecolors = poly_facecolors_ts,
                        linewidths = 0
                    )
                    axs[0, kk].add_collection3d(poly_ts)

                    voxel_facecolors_rt: NP_ARRAY[NP_REAL] = facecolors_rt[jj, kk, i_x, i_y, i_z, :]
                    poly_facecolors_rt: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors_rt, 6, axis = 0)

                    poly_rt: Poly3DCollection = Poly3DCollection(
                        verts,
                        facecolors = poly_facecolors_rt,
                        edgecolors = poly_facecolors_rt,
                        linewidths = 0
                    )
                    axs[1, kk].add_collection3d(poly_rt)

                    filled_diff_i: NP_ARRAY[NP_BOOL] = filled_diff[jj,kk,i_x,i_y,i_z]
                    if np.any(filled_diff_i):
                        verts_diff: NP_ARRAY[NP_REAL] = verts.reshape(n_vox, 6, 4, 3)[filled_diff_i,...].reshape(-1, 4, 3)

                        voxel_facecolors_diff: NP_ARRAY[NP_REAL] = facecolors_diff[
                            jj, kk, i_x[filled_diff_i], i_y[filled_diff_i], i_z[filled_diff_i], :]
                        poly_facecolors_diff: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors_diff, 6, axis = 0)

                        poly_diff: Poly3DCollection = Poly3DCollection(
                            verts_diff,
                            facecolors = poly_facecolors_diff,
                            edgecolors = poly_facecolors_diff,
                            linewidths = 0
                        )
                        axs[2, kk].add_collection3d(poly_diff)

                axs[0, kk].set_xlim(xh_plot.min(), xh_plot.max())
                axs[0, kk].set_ylim(yh_plot.min(), yh_plot.max())
                axs[0, kk].set_zlim(zh_plot.min(), zh_plot.max())

                axs[1, kk].set_xlim(xh_plot.min(), xh_plot.max())
                axs[1, kk].set_ylim(yh_plot.min(), yh_plot.max())
                axs[1, kk].set_zlim(zh_plot.min(), zh_plot.max())

                axs[2, kk].set_xlim(xh_plot.min(), xh_plot.max())
                axs[2, kk].set_ylim(yh_plot.min(), yh_plot.max())
                axs[2, kk].set_zlim(zh_plot.min(), zh_plot.max())

            #-------------------------------------------------------------------
            # Set up colorbars
            #-------------------------------------------------------------------
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
            heating_diff_colorbar.ax.set_ylabel(r"Difference")
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

            #-------------------------------------------------------------------
            # Set style elements
            #-------------------------------------------------------------------
            msg: str = "Setting style elements..."
            print_msg(msg)

            # Background Panes
            pane_color: list[float] = [0.9, 0.9, 0.9, 0.1]
            ax: MPL_AXES
            for ax in axs.flatten():
                ax.xaxis.set_pane_color(pane_color)
                ax.yaxis.set_pane_color(pane_color)
                ax.zaxis.set_pane_color(pane_color)

            # Set grid linewidth
            for ax in axs.flatten():
                ax.xaxis._axinfo["grid"]["linewidth"] = 0
                ax.yaxis._axinfo["grid"]["linewidth"] = 0
                ax.zaxis._axinfo["grid"]["linewidth"] = 0

            # Aspect Ratio
            xh_len: NP_REAL = NP_REAL(xh_plot.max() - xh_plot.min())
            yh_len: NP_REAL = NP_REAL(yh_plot.max() - yh_plot.min())
            zh_len: NP_REAL = NP_REAL(zh_plot.max() - zh_plot.min())
            for ax in axs.flatten():
                ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

            # Column Labels
            kk: int
            for kk in range(0, ncols):
                axs[0,kk].text2D(
                    0.5, 0.86,
                    r"SZA {:.1f}$^{{\circ}}$".format(szas_plot[jj,kk]),
                    transform = axs[0,kk].transAxes,
                    ha = "center",
                    va = "top"
                )

            # Tick Labels - Get rid of unnecessary ones and keep them uniform
            # across all plots
            x_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(xh_plot.min(), xh_plot.max()))
            y_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(yh_plot.min(), yh_plot.max()))
            z_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 2).tick_values(0., np.floor(zh_plot.max())))
            for ax in axs.flatten():
                ax.xaxis.set_ticks(x_ticks)
                ax.yaxis.set_ticks(y_ticks)
                ax.zaxis.set_ticks(z_ticks)
            for ax in (axs[:-1,-1]).flatten():
                ax.xaxis.set_tick_params(labelcolor = "none")
                ax.yaxis.set_tick_params(labelcolor = "none")
            for ax in (axs[-1,:-1]).flatten():
                ax.yaxis.set_tick_params(labelcolor = "none")
                ax.zaxis.set_tick_params(labelcolor = "none")
            for ax in (axs[:-1,:-1]).flatten():
                ax.xaxis.set_tick_params(labelcolor = "none")
                ax.yaxis.set_tick_params(labelcolor = "none")
                ax.zaxis.set_tick_params(labelcolor = "none")

            # Axis labels
            axs[-1,-1].set_ylabel(r"y $\left[ km \right]$")
            for ax in (axs[:,-1]).flatten():
                ax.set_zlabel(r"z $\left[ km \right]$")
            for ax in (axs[-1,:]).flatten():
                ax.set_xlabel(r"x $\left[ km \right]$")

            # Row labels
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

            # Suplabels
            dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
            dx_str: str
            if dx < 1.e3:
                dx_str = r"{:.0f} $m$".format(dx)
            else:
                dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

            fig.suptitle(
                r"Heating Rate $\left[ K\,d^{-1} \right]$" + " - {}".format(dx_str)
            )
            fig.supxlabel("") # Add for padding at bottom

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            msg: str = "Saving plot to file..."
            print_msg(msg)

            plt_filename = "rte_rrtmgp_cpp_diorama_heating.{}.{}.png".format(lr_str, day_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 256)
            plt.close(fig)

if __name__ == "__main__":
    main()