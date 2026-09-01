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
from ml3drt import calc_sw_reflectance as ml3drt_calc_sw_reflectance
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_reflectance as rte_rrtmgp_cpp_calc_sw_reflectance, \
    find_grid, print_msg

# Script variables
prog_name: str = "plot-ml3drt-reflectance-snapshot"
prog_desc: str = "Visualize top-of-atmosphere ML3DRT reflectance state."

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
    parser.add_argument("--ml3drt-outfile", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for ML3DRT output file.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true", default = False,
        help = "Re-calculate necessary quantities for plotting.")
    
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    ml3drt_outfile: str = os.path.normpath(args.ml3drt_outfile)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate

    #---------------------------------------------------------------------------
    # Extract coarse factor from ML3DRT output file name
    #---------------------------------------------------------------------------
    msg: str = "Extracting coarse factor from ML3DRT output file name..."
    print_msg(msg)

    lr_re: re.Pattern = re.compile(r"lr_\d+")
    lr_match: Optional[re.Match] = lr_re.search(os.path.basename(ml3drt_outfile))
    if lr_match is None:
        raise RuntimeError("Unable to extract lr string from ML3DRT output file name.")

    lr_str: str = lr_match.group()
    coarse_factor: NP_INT = NP_INT(lr_str.split("_")[1])
    coarse_factors: NP_ARRAY[NP_INT] = np.array([coarse_factor], dtype = NP_INT)

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
    if nfiles < 1:
        raise RuntimeError("No RTE-RRTMGP-CPP input/output file pairs were found.")

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)
        grid: dict = find_grid(rad_tran_infile)

        dz: NP_REAL = NP_REAL(grid["zh"][1] - grid["zh"][0])

        #-----------------------------------------------------------------------
        # Obtain Morning-Noon-Night time indices, times, SZAs
        #-----------------------------------------------------------------------
        msg: str = "Obtaining morning-noon-night information..."
        print_msg(msg)

        mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
        mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(mnn_indices.shape[0])

        #-----------------------------------------------------------------------
        # Calculate fields for each MNN of each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            nplots: NP_INT = NP_INT(2)
            working_filename: str = "ml3drt_reflectance_snapshot.{}.{}.nc".format(
                lr_str, day_str)
            working_filepath: str = os.path.join(working_dir, working_filename)

            calculate: bool = recalculate or not os.path.exists(working_filepath)

            if calculate:
                #---------------------------------------------------------------
                # Select times to plot
                #---------------------------------------------------------------
                mnn_indices_plot: NP_ARRAY[NP_INT] = mnn_indices[jj,0:nplots]
                mnn_times_plot: NP_ARRAY[NP_REAL] = mnn_times[jj,0:nplots]
                mnn_szas_plot: NP_ARRAY[NP_REAL] = mnn_szas[jj,0:nplots]

                #---------------------------------------------------------------
                # Calculate vertical water path
                #---------------------------------------------------------------
                msg: str = "Calculating vertical water path for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices_plot) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
                vwp: XR_DATAARRAY = dz * cloud_wc.sum(dim = "lay") # [g m^{-2}], [time, y, x]

                #---------------------------------------------------------------
                # Calculate reflectance
                #---------------------------------------------------------------
                msg: str = "Calculating reflectance for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                reflectance_rt: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_reflectance(
                    rad_tran_infile,
                    rad_tran_outfile,
                    mnn_indices_plot,
                    solver = "rt") # Reflectance - ray-tracer; [N/A]; [time, y, x]
                reflectance_ml3drt: XR_DATAARRAY = ml3drt_calc_sw_reflectance(
                    rad_tran_infile,
                    ml3drt_outfile,
                    mnn_indices_plot) # Reflectance - ML3DRT; [N/A]; [time, y, x]
                reflectance_ts: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_reflectance(
                    rad_tran_infile,
                    rad_tran_outfile,
                    mnn_indices_plot,
                    solver = "ts") # Reflectance - two-stream; [N/A]; [time, y, x]

                #---------------------------------------------------------------
                # Transpose fields before plotting
                #---------------------------------------------------------------
                vwp: XR_DATAARRAY = (vwp
                    .transpose("time", "x", "y")
                    .load()) # [g m^{-2}], [time, x, y]

                reflectance_rt: XR_DATAARRAY = (reflectance_rt
                    .transpose("time", "x", "y")
                    .load()) # [N/A], [time, x, y]
                reflectance_ml3drt: XR_DATAARRAY = (reflectance_ml3drt
                    .transpose("time", "x", "y")
                    .load()) # [N/A], [time, x, y]
                reflectance_ts: XR_DATAARRAY = (reflectance_ts
                    .transpose("time", "x", "y")
                    .load()) # [N/A], [time, x, y]

                #---------------------------------------------------------------
                # Calculate differences
                #---------------------------------------------------------------
                reflectance_diff_ml3drt: XR_DATAARRAY = reflectance_ml3drt - reflectance_rt
                reflectance_diff_ts: XR_DATAARRAY = reflectance_ts - reflectance_rt

                #---------------------------------------------------------------
                # Rescale horizontal grids to have correct units
                #---------------------------------------------------------------
                x: XR_DATAARRAY = grid["x"] * 1.e-3 # [m] => [km]
                y: XR_DATAARRAY = grid["y"] * 1.e-3 # [m] => [km]
                xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
                yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]

                #---------------------------------------------------------------
                # Save calculated values to file
                #---------------------------------------------------------------
                msg: str = "Saving calculated values to {}...".format(working_filepath)
                print_msg(msg)

                ds_plot: xr.Dataset = xr.Dataset()
                ds_plot["vwp"] = vwp
                ds_plot["reflectance_rt"] = reflectance_rt
                ds_plot["reflectance_ml3drt"] = reflectance_ml3drt
                ds_plot["reflectance_ts"] = reflectance_ts
                ds_plot["reflectance_diff_ml3drt"] = reflectance_diff_ml3drt
                ds_plot["reflectance_diff_ts"] = reflectance_diff_ts
                ds_plot["mnn_times"] = xr.DataArray(mnn_times_plot, dims = ["time"])
                ds_plot["mnn_szas"] = xr.DataArray(mnn_szas_plot, dims = ["time"])
                ds_plot["x_plot"] = x
                ds_plot["y_plot"] = y
                ds_plot["xh_plot"] = xh
                ds_plot["yh_plot"] = yh

                ds_plot.attrs["rad_tran_infile"] = rad_tran_infile
                ds_plot.attrs["rad_tran_outfile"] = rad_tran_outfile
                ds_plot.attrs["ml3drt_outfile"] = ml3drt_outfile

                ds_plot.to_netcdf(working_filepath)
            else:
                #---------------------------------------------------------------
                # Read calculated values from file
                #---------------------------------------------------------------
                msg: str = "Reading calculated values from {}...".format(working_filepath)
                print_msg(msg)

                ds_plot: xr.Dataset = xr.open_dataset(working_filepath)
                ds_plot.load()
                ds_plot.close()

            #-------------------------------------------------------------------
            # Obtain data to plot
            #-------------------------------------------------------------------
            vwp: XR_DATAARRAY = ds_plot["vwp"]
            reflectance_rt: XR_DATAARRAY = ds_plot["reflectance_rt"]
            reflectance_ml3drt: XR_DATAARRAY = ds_plot["reflectance_ml3drt"]
            reflectance_ts: XR_DATAARRAY = ds_plot["reflectance_ts"]
            reflectance_diff_ml3drt: XR_DATAARRAY = ds_plot["reflectance_diff_ml3drt"]
            reflectance_diff_ts: XR_DATAARRAY = ds_plot["reflectance_diff_ts"]

            mnn_szas_plot: NP_ARRAY[NP_REAL] = np.asarray(ds_plot["mnn_szas"].values,
                dtype = NP_REAL)
            mnn_times_plot: NP_ARRAY[NP_REAL] = np.asarray(ds_plot["mnn_times"].values,
                dtype = NP_REAL)

            x: XR_DATAARRAY = ds_plot["x_plot"]
            y: XR_DATAARRAY = ds_plot["y_plot"]
            xh: XR_DATAARRAY = ds_plot["xh_plot"]
            yh: XR_DATAARRAY = ds_plot["yh_plot"]

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max: list[NP_REAL] = [NP_REAL(vwp.isel(time = ll).max())
                for ll in range(0, nplots)]
            vwp_min: list[NP_REAL] = [NP_REAL(vwp.isel(time = ll).min())
                for ll in range(0, nplots)]

            reflectance_max: list[NP_REAL] = [max(
                NP_REAL(reflectance_rt.isel(time = ll).max()), 
                NP_REAL(reflectance_ml3drt.isel(time = ll).max()), 
                NP_REAL(reflectance_ts.isel(time = ll).max())) for ll in range(0, nplots)]
            reflectance_min: list[NP_REAL] = [min(
                NP_REAL(reflectance_rt.isel(time = ll).min()), 
                NP_REAL(reflectance_ml3drt.isel(time = ll).min()), 
                NP_REAL(reflectance_ts.isel(time = ll).min())) for ll in range(0, nplots)]

            reflectance_diff_max: list[NP_REAL] = [max(
                NP_REAL(np.abs(reflectance_diff_ml3drt).isel(time = ll).max()),
                NP_REAL(np.abs(reflectance_diff_ts).isel(time = ll).max()))
                for ll in range(0, nplots)]

            reflectance_diff_abs_max: NP_REAL = NP_REAL(max(reflectance_diff_max))
            if reflectance_diff_abs_max == 0.:
                reflectance_diff_abs_max = NP_REAL(1.)

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting data..."
            print_msg(msg)

            ll: int
            for ll in range(0, nplots):
                nrows: NP_INT = NP_INT(3)
                ncols: NP_INT = NP_INT(2)
                fig_width: NP_REAL = NP_REAL(6.5)
                fig_height: NP_REAL = NP_REAL(7.5)
                fig_size: list[NP_REAL] = [fig_width, fig_height]
                cbar_width: NP_REAL = NP_REAL(0.06)

                fig = plt.figure(constrained_layout = True,
                    figsize = fig_size)
                gs = fig.add_gridspec(
                    nrows = nrows,
                    ncols = 4,
                    width_ratios = [cbar_width, 1., 1., cbar_width],
                    height_ratios = [1., 1., 1.])

                axs = np.empty((nrows, ncols), dtype = object)

                axs[0,0] = fig.add_subplot(gs[0,1])
                axs[0,1] = fig.add_subplot(gs[0,2], sharex = axs[0,0],
                    sharey = axs[0,0])
                axs[1,0] = fig.add_subplot(gs[1,1], sharex = axs[0,0],
                    sharey = axs[0,0])
                axs[1,1] = fig.add_subplot(gs[1,2], sharex = axs[0,0],
                    sharey = axs[0,0])
                axs[2,0] = fig.add_subplot(gs[2,1], sharex = axs[0,0],
                    sharey = axs[0,0])
                axs[2,1] = fig.add_subplot(gs[2,2], sharex = axs[0,0],
                    sharey = axs[0,0])

                reflectance_cax = fig.add_subplot(gs[:,0])
                vwp_cax = fig.add_subplot(gs[0,3])
                reflectance_diff_cax = fig.add_subplot(gs[1:3,3])

                #---------------------------------------------------------------
                # First column: Reflectance
                #---------------------------------------------------------------
                reflectance_cmap: str = "Oranges_r"

                reflectance_rt_pcm: MPL_PCOLORMESH = axs[0,0].pcolormesh(
                    xh, 
                    yh, 
                    reflectance_rt.isel(time = ll),
                    vmin = min(reflectance_min),
                    vmax = max(reflectance_max),
                    cmap = reflectance_cmap,
                    shading = "flat")

                reflectance_ts_pcm: MPL_PCOLORMESH = axs[1,0].pcolormesh(
                    xh, 
                    yh, 
                    reflectance_ts.isel(time = ll),
                    vmin = min(reflectance_min),
                    vmax = max(reflectance_max),
                    cmap = reflectance_cmap,
                    shading = "flat")

                reflectance_ml3drt_pcm: MPL_PCOLORMESH = axs[2,0].pcolormesh(
                    xh, 
                    yh, 
                    reflectance_ml3drt.isel(time = ll),
                    vmin = min(reflectance_min),
                    vmax = max(reflectance_max),
                    cmap = reflectance_cmap,
                    shading = "flat")

                #---------------------------------------------------------------
                # Second column, top: Vertical Water Path
                #---------------------------------------------------------------
                vwp_pcm: MPL_PCOLORMESH = axs[0,1].pcolormesh(xh, yh,
                    vwp.isel(time = ll),
                    norm = colors.LogNorm(
                        vmin = max(1.e1, min(vwp_min)),
                        vmax = max(vwp_max)),
                    cmap = cw_cmap, shading = "flat")

                #---------------------------------------------------------------
                # Second column, middle and bottom: Reflectance differences
                #---------------------------------------------------------------
                reflectance_diff_norm: colors.CenteredNorm = colors.CenteredNorm(
                    vcenter = 0.,
                    halfrange = reflectance_diff_abs_max)

                reflectance_diff_ts_pcm: MPL_PCOLORMESH = axs[1,1].pcolormesh(
                    xh, 
                    yh, 
                    reflectance_diff_ts.isel(time = ll),
                    norm = reflectance_diff_norm,
                    cmap = diff_cmap, shading = "flat")

                reflectance_diff_ml3drt_pcm: MPL_PCOLORMESH = axs[2,1].pcolormesh(
                    xh, 
                    yh, 
                    reflectance_diff_ml3drt.isel(time = ll),
                    norm = reflectance_diff_norm,
                    cmap = diff_cmap, shading = "flat")

                #---------------------------------------------------------------
                # Axis limits and endpoint ticks
                #---------------------------------------------------------------
                x_min: NP_REAL = NP_REAL(np.nanmin(np.asarray(xh)))
                x_max: NP_REAL = NP_REAL(np.nanmax(np.asarray(xh)))
                y_min: NP_REAL = NP_REAL(np.nanmin(np.asarray(yh)))
                y_max: NP_REAL = NP_REAL(np.nanmax(np.asarray(yh)))

                for mm in range(0, nrows):
                    for kk in range(0, ncols):
                        axs[mm,kk].set_xlim(x_min, x_max)
                        axs[mm,kk].set_ylim(y_min, y_max)

                x_ticks: NP_ARRAY[NP_REAL] = np.asarray(axs[0,0].get_xticks(),
                    dtype = NP_REAL)
                y_ticks: NP_ARRAY[NP_REAL] = np.asarray(axs[0,0].get_yticks(),
                    dtype = NP_REAL)

                x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
                y_ticks = y_ticks[(y_ticks >= y_min) & (y_ticks <= y_max)]

                x_ticks = np.unique(np.concatenate((
                    np.array([x_min], dtype = NP_REAL),
                    x_ticks,
                    np.array([x_max], dtype = NP_REAL))))
                y_ticks = np.unique(np.concatenate((
                    np.array([y_min], dtype = NP_REAL),
                    y_ticks,
                    np.array([y_max], dtype = NP_REAL))))

                for mm in range(0, nrows):
                    for kk in range(0, ncols):
                        axs[mm,kk].set_xticks(x_ticks)
                        axs[mm,kk].set_yticks(y_ticks)
                        axs[mm,kk].tick_params(axis = "x",
                            labelbottom = mm == nrows - 1)
                        axs[mm,kk].tick_params(axis = "y",
                            labelleft = kk == 0)

                #---------------------------------------------------------------
                # Colorbars
                #---------------------------------------------------------------
                reflectance_cbar = fig.colorbar(reflectance_rt_pcm,
                    cax = reflectance_cax)
                vwp_cbar = fig.colorbar(vwp_pcm, cax = vwp_cax,
                    extend = "min")
                reflectance_diff_cbar = fig.colorbar(reflectance_diff_ts_pcm,
                    cax = reflectance_diff_cax)

                reflectance_cbar.ax.yaxis.set_ticks_position("left")
                reflectance_cbar.ax.yaxis.set_label_position("left")

                reflectance_cbar.locator = ticker.MaxNLocator(nbins = 5)
                reflectance_cbar.update_ticks()

                reflectance_diff_cbar.locator = ticker.MaxNLocator(nbins = 5)
                reflectance_diff_cbar.update_ticks()

                #---------------------------------------------------------------
                # Plot contours at major ticks
                #---------------------------------------------------------------
                # Reflectance
                reflectance_levels: NP_ARRAY[NP_REAL] = NP_REAL(reflectance_cbar.ax.get_yticks())

                axs[0,0].contour(
                    x,
                    y,
                    reflectance_rt.isel(time = ll),
                    levels = reflectance_levels,
                    colors = "k",
                    linewidths = 1.0
                )
                axs[1,0].contour(
                    x,
                    y,
                    reflectance_ts.isel(time = ll),
                    levels = reflectance_levels,
                    colors = "k",
                    linewidths = 1.0
                )
                axs[2,0].contour(
                    x,
                    y,
                    reflectance_ml3drt.isel(time = ll),
                    levels = reflectance_levels,
                    colors = "k",
                    linewidths = 1.0
                )

                level: NP_REAL
                for level in reflectance_levels:
                    reflectance_cbar.ax.axhline(
                        level,
                        color = "k",
                        linewidth = 1.0,
                        linestyle = "solid"
                    )

                # VWP
                vwp_levels: NP_ARRAY[NP_REAL] = NP_REAL(vwp_cbar.ax.get_yticks())

                axs[0,1].contour(
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

                # Reflectance Difference
                reflectance_diff_levels: NP_ARRAY[NP_REAL] = NP_REAL(
                    reflectance_diff_cbar.ax.get_yticks())

                axs[1,1].contour(
                    x,
                    y,
                    reflectance_diff_ts.isel(time = ll),
                    levels = reflectance_diff_levels,
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )
                axs[2,1].contour(
                    x,
                    y,
                    reflectance_diff_ml3drt.isel(time = ll),
                    levels = reflectance_diff_levels,
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )
                
                level: NP_REAL
                for level in reflectance_diff_levels:
                    if level > 0.:
                        reflectance_diff_cbar.ax.axhline(
                            level,
                            color = "k",
                            linewidth = 1.0,
                            linestyle = "solid"
                        )
                    elif level < 0.:
                        reflectance_diff_cbar.ax.axhline(
                            level,
                            color = "k",
                            linewidth = 1.0,
                            linestyle = "dashed"
                        )

                #---------------------------------------------------------------
                # Labels
                #---------------------------------------------------------------
                fig.suptitle(r"Reflectance"
                    + r" - SZA {:.1f}$^{{\circ}}$".format(mnn_szas_plot[ll]))
                fig.supxlabel(r"x $\left[ km \right]$")
                fig.supylabel(r"y $\left[ km \right]$")

                axs[0,0].set_title(r"Ray-Tracer")
                axs[1,0].set_title(r"Two-Stream")
                axs[2,0].set_title(r"Emulator")
                axs[1,1].set_title(r"Two-Stream - Ray-Tracer")
                axs[2,1].set_title(r"Emulator - Ray-Tracer")

                vwp_cbar.ax.set_ylabel(r"Vertical CWP $\left[ g\,m^{-2} \right]$")
                reflectance_diff_cbar.ax.set_ylabel(r"Difference")

                # Aspect ratio
                mm: int
                for mm in range(0, nrows):
                    for kk in range(0, ncols):
                        axs[mm,kk].set_aspect("equal")

                #---------------------------------------------------------------
                # Save the plot to file
                #---------------------------------------------------------------
                sza_str: str = "sza_{:02d}".format(NP_INT(np.round(mnn_szas_plot[ll])))
                plt_filename: str = "ml3drt_reflectance_snapshot.{}.{}.{}.png".format(
                    lr_str, day_str, sza_str)
                plt_filepath: str = os.path.join(rad_tran_vizdir, plt_filename)
                fig.savefig(plt_filepath, dpi = 200)
                plt.close(fig)

if __name__ == "__main__":
    main()