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
from matplotlib import ticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, MPL_PCOLORMESH
from consts.visual import diff_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_reflectance, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-snapshot-reflectance"
prog_desc: str = "Visualize top-of-atmosphere state for RTE-RRTMGP-CPP."

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
    parser.add_argument("--recalculate", action = "store_true", default = False,
        help = "Re-calculate quantities required for plotting.")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate

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
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(rad_tran_indir,
        rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    if nfiles == 0:
        return

    #---------------------------------------------------------------------------
    # Define required saved-data structure
    #---------------------------------------------------------------------------
    required_data_vars: list[str] = [
        "vwp",
        "reflectance_rt",
        "reflectance_ts",
        "reflectance_diff",
        "mnn_times",
        "mnn_szas"
    ]
    required_coords: list[str] = [
        "day",
        "time",
        "x",
        "y",
        "xh",
        "yh"
    ]
    required_dims: dict[str, tuple[str, ...]] = {
        "vwp": ("day", "time", "x", "y"),
        "reflectance_rt": ("day", "time", "x", "y"),
        "reflectance_ts": ("day", "time", "x", "y"),
        "reflectance_diff": ("day", "time", "x", "y"),
        "mnn_times": ("day", "time"),
        "mnn_szas": ("day", "time")
    }

    lr_re: re.Pattern = re.compile("lr_..")

    plot_ds_list: list[xr.Dataset] = []
    lr_str_list: list[str] = []

    #---------------------------------------------------------------------------
    # Calculate or read plotting data for each horizontal resolution
    #---------------------------------------------------------------------------
    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        msg = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Find saved data file
        #-----------------------------------------------------------------------
        plot_data_filename: str = "rte_rrtmgp_cpp_rad_tran_tod_snapshot_reflectance.{}.nc".format(lr_str)
        plot_data_filepath: str = os.path.join(working_dir, plot_data_filename)

        #-----------------------------------------------------------------------
        # Determine whether plotting data must be recalculated
        #-----------------------------------------------------------------------
        plot_ds: Optional[xr.Dataset] = None
        recalculate_plot_data: bool = False

        if recalculate:
            msg = "Recalculating plotted quantities..."
            print_msg(msg)
            recalculate_plot_data = True
        elif not os.path.exists(plot_data_filepath):
            msg = "Saved plotted quantities not found; calculating..."
            print_msg(msg)
            recalculate_plot_data = True
        else:
            msg = "Reading plotted quantities from {}...".format(plot_data_filepath)
            print_msg(msg)

            try:
                with xr.open_dataset(plot_data_filepath) as tmp_ds:
                    plot_ds = tmp_ds.load()

                missing_data_vars: list[str] = [
                    var_name for var_name in required_data_vars
                    if var_name not in plot_ds.data_vars
                ]
                missing_coords: list[str] = [
                    coord_name for coord_name in required_coords
                    if coord_name not in plot_ds.coords
                ]
                incorrect_dims: list[str] = [
                    var_name for var_name in required_dims
                    if var_name in plot_ds and tuple(plot_ds[var_name].dims) != required_dims[var_name]
                ]

                if len(missing_data_vars) > 0 or len(missing_coords) > 0 or len(incorrect_dims) > 0:
                    msg = "Saved plotted quantities do not contain all required information; recalculating..."
                    print_msg(msg)

                    if len(missing_data_vars) > 0:
                        msg = "Missing data variables: {}".format(", ".join(missing_data_vars))
                        print_msg(msg)
                    if len(missing_coords) > 0:
                        msg = "Missing coordinates: {}".format(", ".join(missing_coords))
                        print_msg(msg)
                    if len(incorrect_dims) > 0:
                        msg = "Incorrect dimensions: {}".format(", ".join(incorrect_dims))
                        print_msg(msg)

                    recalculate_plot_data = True
                    plot_ds.close()
                    plot_ds = None

            except Exception as err:
                msg = "Unable to read saved plotted quantities; recalculating..."
                print_msg(msg)
                msg = "Read error: {}".format(err)
                print_msg(msg)

                recalculate_plot_data = True
                plot_ds = None

        #-----------------------------------------------------------------------
        # Calculate plotting data
        #-----------------------------------------------------------------------
        if recalculate_plot_data:
            #-------------------------------------------------------------------
            # Obtain grid information
            #-------------------------------------------------------------------
            msg = "Obtaining grid information..."
            print_msg(msg)
            grid: dict = find_grid(rad_tran_infile)

            dz: NP_REAL = NP_REAL(grid["zh"][1] - grid["zh"][0])

            x: XR_DATAARRAY = grid["x"] * 1.e-3 # [m] => [km]
            y: XR_DATAARRAY = grid["y"] * 1.e-3 # [m] => [km]
            xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
            yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]

            #-------------------------------------------------------------------
            # Obtain Morning-Noon-Night time indices, times, SZAs
            #-------------------------------------------------------------------
            msg = "Obtaining morning-noon-night information..."
            print_msg(msg)

            mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
            mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
            mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
            ndays: NP_INT = NP_INT(mnn_indices.shape[0])

            day: NP_ARRAY[NP_INT] = np.arange(0, ndays, dtype = NP_INT)
            time: NP_ARRAY[NP_INT] = np.arange(0, 3, dtype = NP_INT)

            vwp_list: list[XR_DATAARRAY] = []
            reflectance_rt_list: list[XR_DATAARRAY] = []
            reflectance_ts_list: list[XR_DATAARRAY] = []

            #-------------------------------------------------------------------
            # Calculate fields for each MNN of each day
            #-------------------------------------------------------------------
            jj: int
            for jj in range(0, ndays):
                #---------------------------------------------------------------
                # Calculate vertical water path
                #---------------------------------------------------------------
                msg = "Calculating vertical water path for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj]) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
                vwp: XR_DATAARRAY = dz * cloud_wc.sum(dim = "lay") # [g m^{-2}], [time, y, x]

                #---------------------------------------------------------------
                # Calculate reflectance
                #---------------------------------------------------------------
                msg = "Calculating reflectance for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                reflectance_rt: XR_DATAARRAY = calc_sw_reflectance(rad_tran_infile,
                    rad_tran_outfile,
                    mnn_indices[jj],
                    solver = "rt") # Reflectance - ray-tracer; [N/A]; [time, y, x]
                reflectance_ts: XR_DATAARRAY = calc_sw_reflectance(rad_tran_infile,
                    rad_tran_outfile,
                    mnn_indices[jj],
                    solver = "ts") # Reflectance - two-stream; [N/A]; [time, y, x]

                #---------------------------------------------------------------
                # Transpose fields before saving and plotting
                #---------------------------------------------------------------
                vwp = (vwp
                    .transpose("time", "x", "y")
                    .assign_coords(time = time, x = x, y = y)
                    .load()) # [g m^{-2}], [time, x, y]

                reflectance_rt = (reflectance_rt
                    .transpose("time", "x", "y")
                    .assign_coords(time = time, x = x, y = y)
                    .load()) # [N/A], [time, x, y]
                reflectance_ts = (reflectance_ts
                    .transpose("time", "x", "y")
                    .assign_coords(time = time, x = x, y = y)
                    .load()) # [N/A], [time, x, y]

                vwp_list.append(vwp)
                reflectance_rt_list.append(reflectance_rt)
                reflectance_ts_list.append(reflectance_ts)

            #-------------------------------------------------------------------
            # Assemble plotting dataset
            #-------------------------------------------------------------------
            vwp = (xr.concat(vwp_list, dim = "day")
                .assign_coords(day = day, time = time, x = x, y = y)) # [g m^{-2}], [day, time, x, y]
            reflectance_rt = (xr.concat(reflectance_rt_list, dim = "day")
                .assign_coords(day = day, time = time, x = x, y = y)) # [N/A], [day, time, x, y]
            reflectance_ts = (xr.concat(reflectance_ts_list, dim = "day")
                .assign_coords(day = day, time = time, x = x, y = y)) # [N/A], [day, time, x, y]
            reflectance_diff: XR_DATAARRAY = reflectance_ts - reflectance_rt # [N/A], [day, time, x, y]

            plot_ds = xr.Dataset(
                data_vars = {
                    "vwp": vwp,
                    "reflectance_rt": reflectance_rt,
                    "reflectance_ts": reflectance_ts,
                    "reflectance_diff": reflectance_diff,
                    "mnn_times": (["day", "time"], mnn_times),
                    "mnn_szas": (["day", "time"], mnn_szas)
                },
                coords = {
                    "day": day,
                    "time": time,
                    "x": x,
                    "y": y,
                    "xh": xh,
                    "yh": yh
                }
            )

            #-------------------------------------------------------------------
            # Add metadata
            #-------------------------------------------------------------------
            plot_ds["vwp"].attrs["long_name"] = "Vertical cloud water path"
            plot_ds["vwp"].attrs["units"] = "g m^{-2}"
            plot_ds["reflectance_rt"].attrs["long_name"] = "Shortwave reflectance, ray-tracer"
            plot_ds["reflectance_rt"].attrs["units"] = "1"
            plot_ds["reflectance_ts"].attrs["long_name"] = "Shortwave reflectance, two-stream"
            plot_ds["reflectance_ts"].attrs["units"] = "1"
            plot_ds["reflectance_diff"].attrs["long_name"] = "Shortwave reflectance difference, two-stream minus ray-tracer"
            plot_ds["reflectance_diff"].attrs["units"] = "1"
            plot_ds["mnn_times"].attrs["long_name"] = "Morning-noon-night time since simulation start"
            plot_ds["mnn_times"].attrs["units"] = "h"
            plot_ds["mnn_szas"].attrs["long_name"] = "Morning-noon-night solar zenith angle"
            plot_ds["mnn_szas"].attrs["units"] = "degrees"
            plot_ds["x"].attrs["units"] = "km"
            plot_ds["y"].attrs["units"] = "km"
            plot_ds["xh"].attrs["units"] = "km"
            plot_ds["yh"].attrs["units"] = "km"

            #-------------------------------------------------------------------
            # Save plotting dataset
            #-------------------------------------------------------------------
            msg = "Saving plotted quantities to {}...".format(plot_data_filepath)
            print_msg(msg)
            plot_ds.to_netcdf(plot_data_filepath)

        plot_ds_list.append(plot_ds)
        lr_str_list.append(lr_str)

    #---------------------------------------------------------------------------
    # Plot the data for each horizontal resolution and each day
    #---------------------------------------------------------------------------
    for ii in range(0, nfiles):
        plot_ds = plot_ds_list[ii]
        lr_str: str = lr_str_list[ii]

        msg = "Plotting {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain saved data
        #-----------------------------------------------------------------------
        vwp_all: XR_DATAARRAY = plot_ds["vwp"] # [g m^{-2}], [day, time, x, y]
        reflectance_rt_all: XR_DATAARRAY = plot_ds["reflectance_rt"] # [N/A], [day, time, x, y]
        reflectance_ts_all: XR_DATAARRAY = plot_ds["reflectance_ts"] # [N/A], [day, time, x, y]
        reflectance_diff_all: XR_DATAARRAY = plot_ds["reflectance_diff"] # [N/A], [day, time, x, y]

        mnn_times: NP_ARRAY[NP_REAL] = plot_ds["mnn_times"].values # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = plot_ds["mnn_szas"].values # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(plot_ds.sizes["day"])

        x: XR_DATAARRAY = plot_ds["x"] # [km]
        y: XR_DATAARRAY = plot_ds["y"] # [km]
        xh: XR_DATAARRAY = plot_ds["xh"] # [km]
        yh: XR_DATAARRAY = plot_ds["yh"] # [km]

        #-----------------------------------------------------------------------
        # Plot the data for each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            msg = "Preparing data for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            vwp: XR_DATAARRAY = vwp_all.isel(day = jj) # [g m^{-2}], [time, x, y]
            reflectance_rt: XR_DATAARRAY = reflectance_rt_all.isel(day = jj) # [N/A], [time, x, y]
            reflectance_ts: XR_DATAARRAY = reflectance_ts_all.isel(day = jj) # [N/A], [time, x, y]
            reflectance_diff: XR_DATAARRAY = reflectance_diff_all.isel(day = jj) # [N/A], [time, x, y]

            #-------------------------------------------------------------------
            # Obtain colorbar limits for this day and horizontal resolution
            #-------------------------------------------------------------------
            vwp_positive_min: NP_REAL = NP_REAL(vwp.where(vwp > 0.).min(skipna = True))
            vwp_max: NP_REAL = NP_REAL(vwp.max(skipna = True))

            if np.isfinite(vwp_positive_min):
                vwp_min: NP_REAL = NP_REAL(vwp_positive_min)
            else:
                vwp_min = NP_REAL(1.e1)

            if np.isfinite(vwp_max):
                vwp_max = NP_REAL(vwp_max)
            else:
                vwp_max = NP_REAL(1.e2)

            vwp_vmin: NP_REAL = NP_REAL(max(1.e1, vwp_min))
            vwp_vmax: NP_REAL = NP_REAL(vwp_max)
            if vwp_vmax <= vwp_vmin:
                vwp_vmax = NP_REAL(10. * vwp_vmin)

            reflectance_min: NP_REAL = NP_REAL(min(
                NP_REAL(reflectance_rt.min(skipna = True)),
                NP_REAL(reflectance_ts.min(skipna = True))
            ))
            reflectance_max: NP_REAL = NP_REAL(max(
                NP_REAL(reflectance_rt.max(skipna = True)),
                NP_REAL(reflectance_ts.max(skipna = True))
            ))

            if not np.isfinite(reflectance_min):
                reflectance_min = NP_REAL(0.)
            if not np.isfinite(reflectance_max):
                reflectance_max = NP_REAL(1.)
            if reflectance_max <= reflectance_min:
                reflectance_max = NP_REAL(reflectance_min + 1.e-6)

            reflectance_levels: NP_ARRAY[NP_REAL] = np.linspace(
                reflectance_min,
                reflectance_max,
                5,
                dtype = NP_REAL)

            reflectance_diff_abs_max: NP_REAL = NP_REAL(np.abs(reflectance_diff).max(skipna = True))

            if not np.isfinite(reflectance_diff_abs_max):
                reflectance_diff_abs_max = NP_REAL(1.e-6)
            if reflectance_diff_abs_max <= 0.:
                reflectance_diff_abs_max = NP_REAL(1.e-6)

            reflectance_diff_locator = ticker.MaxNLocator(
                nbins = 5,
                symmetric = True)

            reflectance_diff_levels: NP_ARRAY[NP_REAL] = np.array(
                reflectance_diff_locator.tick_values(
                    -reflectance_diff_abs_max,
                    reflectance_diff_abs_max),
                dtype = NP_REAL)

            if 0. not in reflectance_diff_levels:
                reflectance_diff_levels = np.sort(np.append(reflectance_diff_levels, NP_REAL(0.)))

            reflectance_diff_abs_max = NP_REAL(max(np.abs(reflectance_diff_levels)))

            if reflectance_diff_abs_max <= 0.:
                reflectance_diff_abs_max = NP_REAL(1.e-6)
                reflectance_diff_levels = np.array(
                    [-reflectance_diff_abs_max, 0., reflectance_diff_abs_max],
                    dtype = NP_REAL)

            #-------------------------------------------------------------------
            # Set up figure
            #-------------------------------------------------------------------
            msg = "Plotting data..."
            print_msg(msg)

            nrows: NP_INT = NP_INT(4)
            ncols: NP_INT = NP_INT(2)
            fig_width: NP_REAL = NP_REAL(6.0)
            fig_height: NP_REAL = NP_REAL(8.)
            fig_size: list[NP_REAL] = [fig_width, fig_height]

            fig = plt.figure(
                constrained_layout = True,
                figsize = fig_size)

            gs = fig.add_gridspec(
                nrows = nrows,
                ncols = ncols + 1,
                width_ratios = [1. for _ in range(0, ncols)] + [0.06])

            axs = np.empty((nrows, ncols), dtype = object)

            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    if ll == 0 and mm == 0:
                        axs[ll,mm] = fig.add_subplot(gs[ll,mm])
                    elif ll == 0:
                        axs[ll,mm] = fig.add_subplot(
                            gs[ll,mm],
                            sharey = axs[0,0])
                    elif mm == 0:
                        axs[ll,mm] = fig.add_subplot(
                            gs[ll,mm],
                            sharex = axs[0,mm],
                            sharey = axs[0,0])
                    else:
                        axs[ll,mm] = fig.add_subplot(
                            gs[ll,mm],
                            sharex = axs[0,mm],
                            sharey = axs[0,0])

            vwp_cax = fig.add_subplot(gs[0,ncols])
            reflectance_cax = fig.add_subplot(gs[1:3,ncols])
            reflectance_diff_cax = fig.add_subplot(gs[3,ncols])

            #-------------------------------------------------------------------
            # Plot filled fields
            #-------------------------------------------------------------------
            # Row 0: Vertical Water Path
            vwp_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            for ll in range(0, ncols):
                vwp_pcm[ll] = axs[0,ll].pcolormesh(
                    xh,
                    yh,
                    vwp.isel(time = ll),
                    norm = colors.LogNorm(
                        vmin = vwp_vmin,
                        vmax = vwp_vmax),
                    cmap = cw_cmap,
                    shading = "flat")

            # Row 1: Reflectance, Two-Stream
            reflectance_cmap: str = "Oranges_r"
            reflectance_ts_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            for ll in range(0, ncols):
                reflectance_ts_pcm[ll] = axs[1,ll].pcolormesh(
                    xh,
                    yh,
                    reflectance_ts.isel(time = ll),
                    vmin = reflectance_min,
                    vmax = reflectance_max,
                    cmap = reflectance_cmap,
                    shading = "flat")

            # Row 2: Reflectance, Ray-Tracer
            reflectance_rt_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            for ll in range(0, ncols):
                reflectance_rt_pcm[ll] = axs[2,ll].pcolormesh(
                    xh,
                    yh,
                    reflectance_rt.isel(time = ll),
                    vmin = reflectance_min,
                    vmax = reflectance_max,
                    cmap = reflectance_cmap,
                    shading = "flat")

            # Row 3: Reflectance Difference
            reflectance_diff_norm = colors.Normalize(
                vmin = -reflectance_diff_abs_max,
                vmax = reflectance_diff_abs_max)

            reflectance_diff_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            for ll in range(0, ncols):
                reflectance_diff_pcm[ll] = axs[3,ll].pcolormesh(
                    xh,
                    yh,
                    reflectance_diff.isel(time = ll),
                    norm = reflectance_diff_norm,
                    cmap = diff_cmap,
                    shading = "flat")

            #-------------------------------------------------------------------
            # Add colorbars
            #-------------------------------------------------------------------
            vwp_cbar = fig.colorbar(
                vwp_pcm[0],
                cax = vwp_cax,
                extend = "min")
            reflectance_cbar = fig.colorbar(
                reflectance_ts_pcm[0],
                cax = reflectance_cax)
            reflectance_diff_cbar = fig.colorbar(
                reflectance_diff_pcm[0],
                cax = reflectance_diff_cax,
                ticks = reflectance_diff_levels)

            #-------------------------------------------------------------------
            # Plot contours at major ticks
            #-------------------------------------------------------------------
            # VWP
            vwp_levels: NP_ARRAY[NP_REAL] = np.array(vwp_cbar.ax.get_yticks(), dtype = NP_REAL)

            row: NP_INT = NP_INT(0)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    x,
                    y,
                    vwp.isel(time = ll),
                    levels = vwp_levels,
                    colors = "k",
                    linewidths = 1.0
                )

            level: NP_REAL
            for level in vwp_levels:
                vwp_cbar.ax.axhline(
                    level,
                    color = "k",
                    linewidth = 1.0,
                    linestyle = "solid"
                )

            # Reflectance
            reflectance_levels: NP_ARRAY[NP_REAL] = np.array(reflectance_cbar.ax.get_yticks(), dtype = NP_REAL)

            # Row 1: Two-Stream
            row = NP_INT(1)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    x,
                    y,
                    reflectance_ts.isel(time = ll),
                    levels = reflectance_levels,
                    colors = "k",
                    linewidths = 1.0
                )

            # Row 2: Ray-Tracer
            row = NP_INT(2)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    x,
                    y,
                    reflectance_rt.isel(time = ll),
                    levels = reflectance_levels,
                    colors = "k",
                    linewidths = 1.0
                )

            for level in reflectance_levels:
                reflectance_cbar.ax.axhline(
                    level,
                    color = "k",
                    linewidth = 1.0,
                    linestyle = "solid"
                )

            # Reflectance Difference
            reflectance_diff_contour_levels: NP_ARRAY[NP_REAL] = np.array(
                [np.min(reflectance_diff_levels[reflectance_diff_levels > 0.]),
                np.max(reflectance_diff_levels[reflectance_diff_levels < 0.])],
                dtype = NP_REAL)

            reflectance_diff_negative_levels: NP_ARRAY[NP_REAL] = reflectance_diff_contour_levels[
                reflectance_diff_contour_levels < 0.]
            reflectance_diff_positive_levels: NP_ARRAY[NP_REAL] = reflectance_diff_contour_levels[
                reflectance_diff_contour_levels > 0.]

            row = NP_INT(3)
            for ll in range(0, ncols):
                if len(reflectance_diff_negative_levels) > 0:
                    axs[row,ll].contour(
                        x,
                        y,
                        reflectance_diff.isel(time = ll),
                        levels = reflectance_diff_negative_levels,
                        colors = "k",
                        linewidths = 1.0,
                        linestyles = "dashed"
                    )

                if len(reflectance_diff_positive_levels) > 0:
                    axs[row,ll].contour(
                        x,
                        y,
                        reflectance_diff.isel(time = ll),
                        levels = reflectance_diff_positive_levels,
                        colors = "k",
                        linewidths = 1.0,
                        linestyles = "solid"
                    )

            level: NP_REAL
            for level in reflectance_diff_negative_levels:
                reflectance_diff_cbar.ax.axhline(
                    level,
                    color = "k",
                    linewidth = 1.0,
                    linestyle = "dashed"
                )

            for level in reflectance_diff_positive_levels:
                reflectance_diff_cbar.ax.axhline(
                    level,
                    color = "k",
                    linewidth = 1.0,
                    linestyle = "solid"
                )

            #-------------------------------------------------------------------
            # Labels
            #-------------------------------------------------------------------
            dx: NP_REAL = NP_REAL(1.e3 * (xh[1] - xh[0])) # [km] => [m]
            dx_str: str
            if dx < 1.e3:
                dx_str = r"{:.0f} $m$".format(dx)
            else:
                dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

            fig.suptitle(r"Reflectance" + " - {}".format(dx_str))
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"y $\left[ km \right]$")

            for ll in range(0, ncols):
                # col_title: str = (r"{:.2f} Hours - ".format(mnn_times[jj,ll])
                #     + r"Solar Zenith Angle {:.1f}$^{{\circ}}$".format(mnn_szas[jj,ll]))
                col_title: str = (r"SZA {:.1f}$^{{\circ}}$".format(mnn_szas[jj,ll]))
                axs[0,ll].set_title(col_title)

            axs[1,0].set_ylabel(r"Two-Stream")
            axs[2,0].set_ylabel(r"Ray-Tracer")
            axs[3,0].set_ylabel(r"Two-Stream - Ray-Tracer")

            vwp_cbar.ax.set_ylabel(r"Vertical CWP $\left[ g\,m^{-2} \right]$")
            reflectance_diff_cbar.ax.set_ylabel(r"Difference")

            #-------------------------------------------------------------------
            # Aspect ratio, limits, and ticks
            #-------------------------------------------------------------------
            x_min: NP_REAL = NP_REAL(xh.min())
            x_max: NP_REAL = NP_REAL(xh.max())
            y_min: NP_REAL = NP_REAL(yh.min())
            y_max: NP_REAL = NP_REAL(yh.max())

            tick_spacing: NP_REAL = NP_REAL(50.) # [km]
            tick_tol: NP_REAL = NP_REAL(1.e-6 * tick_spacing)

            x_lim_min: NP_REAL = NP_REAL(np.floor((x_min + tick_tol) / tick_spacing) * tick_spacing)
            x_lim_max: NP_REAL = NP_REAL(np.ceil((x_max - tick_tol) / tick_spacing) * tick_spacing)
            y_lim_min: NP_REAL = NP_REAL(np.floor((y_min + tick_tol) / tick_spacing) * tick_spacing)
            y_lim_max: NP_REAL = NP_REAL(np.ceil((y_max - tick_tol) / tick_spacing) * tick_spacing)

            x_ticks: NP_ARRAY[NP_REAL] = np.arange(
                x_lim_min,
                x_lim_max + 0.5 * tick_spacing,
                tick_spacing,
                dtype = NP_REAL)
            y_ticks: NP_ARRAY[NP_REAL] = np.arange(
                y_lim_min,
                y_lim_max + 0.5 * tick_spacing,
                tick_spacing,
                dtype = NP_REAL)

            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_aspect("equal")
                    axs[ll,mm].set_xlim(x_lim_min, x_lim_max)
                    axs[ll,mm].set_ylim(y_lim_min, y_lim_max)
                    axs[ll,mm].set_xticks(x_ticks)
                    axs[ll,mm].set_yticks(y_ticks)

            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].label_outer()

            #-------------------------------------------------------------------
            # Colorbar axes are managed explicitly by GridSpec
            #-------------------------------------------------------------------
            fig.canvas.draw()

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_snapshot_reflectance.{}.{}.png".format(lr_str, day_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()