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
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, MPL_PCOLORMESH
from consts.visual import diff_cmap, flux_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_flux_sfc_dn as rte_rrtmgp_cpp_calc_sw_flux_sfc_dn, \
    find_grid, print_msg
from ml3drt import calc_sw_flux_sfc_dn as ml3drt_calc_sw_flux_sfc_dn

# Script variables
prog_name: str = "plot-ml3drt-flux_sfc_dn-snapshot"
prog_desc: str = "Visualize surface downwelling flux snapshots for ML3DRT."

def find_plot_ticks(coord: XR_DATAARRAY, nticks: NP_INT = NP_INT(5)) -> NP_ARRAY[NP_REAL]:
    coord_min: NP_REAL = NP_REAL(np.nanmin(np.array(coord, dtype = NP_REAL)))
    coord_max: NP_REAL = NP_REAL(np.nanmax(np.array(coord, dtype = NP_REAL)))

    if coord_min == coord_max:
        ticks: NP_ARRAY[NP_REAL] = np.array([coord_min], dtype = NP_REAL)
    else:
        ticks: NP_ARRAY[NP_REAL] = np.array(
            np.linspace(coord_min, coord_max, int(nticks)), dtype = NP_REAL)

    return ticks

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
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate all necessary quantities for plotting.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    ml3drt_outfile: str = os.path.normpath(args.ml3drt_outfile)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate

    #---------------------------------------------------------------------------
    # Extract coarse factor from ML3DRT output filename
    #---------------------------------------------------------------------------
    lr_re: re.Pattern = re.compile(r"lr_\d+")
    lr_match: Optional[re.Match] = lr_re.search(os.path.basename(ml3drt_outfile))

    if lr_match is None:
        raise RuntimeError("Unable to extract lr string from {}.".format(ml3drt_outfile))

    lr_str: str = lr_match.group()
    coarse_factor: NP_INT = NP_INT(lr_str.split("_")[-1])
    coarse_factors: Optional[NP_ARRAY[NP_INT]] = np.array([coarse_factor], dtype = NP_INT)

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Find file pairs at requested resolution
    #---------------------------------------------------------------------------
    rad_tran_infiles: list[str]
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(rad_tran_indir,
        rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

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

            plot_data_filename: str = "ml3drt_flux_sfc_dn_snapshot.{}.{}.nc".format(
                lr_str, day_str)
            plot_data_filepath: str = os.path.join(working_dir, plot_data_filename)

            if recalculate or not os.path.exists(plot_data_filepath):
                #---------------------------------------------------------------
                # Calculate vertical water path
                #---------------------------------------------------------------
                msg: str = "Calculating vertical water path for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj]) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
                vwp: XR_DATAARRAY = dz * cloud_wc.sum(dim = "lay") # [g m^{-2}], [time, y, x]

                #---------------------------------------------------------------
                # Calculate downwelling surface flux
                #---------------------------------------------------------------
                msg: str = "Calculating downwelling surface flux for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                flux_sfc_dn_rt: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_flux_sfc_dn(
                    rad_tran_outfile,
                    mnn_indices[jj],
                    solver = "rt") # Downwelling surface flux - ray-tracer; [W m^{-2}]; [time, y, x]
                flux_sfc_dn_ts: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_flux_sfc_dn(
                    rad_tran_outfile,
                    mnn_indices[jj],
                    solver = "ts") # Downwelling surface flux - two-stream; [W m^{-2}]; [time, y, x]
                flux_sfc_dn_ml3drt: XR_DATAARRAY = ml3drt_calc_sw_flux_sfc_dn(
                    ml3drt_outfile,
                    mnn_indices[jj]) # Downwelling surface flux - ML3DRT emulator; [W m^{-2}]; [time, y, x]

                #---------------------------------------------------------------
                # Transpose fields before plotting
                #---------------------------------------------------------------
                vwp: XR_DATAARRAY = (vwp
                    .transpose("time", "x", "y")
                    .load()) # [g m^{-2}], [time, x, y]
                flux_sfc_dn_rt: XR_DATAARRAY = (flux_sfc_dn_rt
                    .transpose("time", "x", "y")
                    .load()) # [W m^{-2}], [time, x, y]
                flux_sfc_dn_ts: XR_DATAARRAY = (flux_sfc_dn_ts
                    .transpose("time", "x", "y")
                    .load()) # [W m^{-2}], [time, x, y]
                flux_sfc_dn_ml3drt: XR_DATAARRAY = (flux_sfc_dn_ml3drt
                    .transpose("time", "x", "y")
                    .load()) # [W m^{-2}], [time, x, y]

                #---------------------------------------------------------------
                # Rescale horizontal grids to have correct units
                #---------------------------------------------------------------
                x: XR_DATAARRAY = grid["x"] * 1.e-3 # [m] => [km]
                y: XR_DATAARRAY = grid["y"] * 1.e-3 # [m] => [km]
                xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
                yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]

                #---------------------------------------------------------------
                # Save plot data
                #---------------------------------------------------------------
                msg: str = "Saving plot data to {}...".format(plot_data_filepath)
                print_msg(msg)

                ncols: NP_INT = NP_INT(2)
                sza_indices: NP_ARRAY[NP_INT] = np.arange(0, ncols, dtype = NP_INT)

                flux_sfc_dn_ml3drt_diff: XR_DATAARRAY = (
                    flux_sfc_dn_ml3drt - flux_sfc_dn_rt)
                flux_sfc_dn_ts_diff: XR_DATAARRAY = (
                    flux_sfc_dn_ts - flux_sfc_dn_rt)

                plot_data: xr.Dataset = xr.Dataset(
                    data_vars = {
                        "vwp": (["sza", "x", "y"],
                            np.array(vwp.isel(time = slice(0, ncols)), dtype = NP_REAL)),
                        "flux_sfc_dn_rt": (["sza", "x", "y"],
                            np.array(flux_sfc_dn_rt.isel(time = slice(0, ncols)), dtype = NP_REAL)),
                        "flux_sfc_dn_ml3drt": (["sza", "x", "y"],
                            np.array(flux_sfc_dn_ml3drt.isel(time = slice(0, ncols)), dtype = NP_REAL)),
                        "flux_sfc_dn_ts": (["sza", "x", "y"],
                            np.array(flux_sfc_dn_ts.isel(time = slice(0, ncols)), dtype = NP_REAL)),
                        "flux_sfc_dn_ml3drt_diff": (["sza", "x", "y"],
                            np.array(flux_sfc_dn_ml3drt_diff.isel(time = slice(0, ncols)), dtype = NP_REAL)),
                        "flux_sfc_dn_ts_diff": (["sza", "x", "y"],
                            np.array(flux_sfc_dn_ts_diff.isel(time = slice(0, ncols)), dtype = NP_REAL)),
                        "mnn_times": (["sza"],
                            np.array(mnn_times[jj,0:ncols], dtype = NP_REAL)),
                        "mnn_szas": (["sza"],
                            np.array(mnn_szas[jj,0:ncols], dtype = NP_REAL))
                    },
                    coords = {
                        "sza": sza_indices,
                        "x": np.array(x, dtype = NP_REAL),
                        "y": np.array(y, dtype = NP_REAL),
                        "xh": np.array(xh, dtype = NP_REAL),
                        "yh": np.array(yh, dtype = NP_REAL)
                    },
                    attrs = {
                        "description": "ML3DRT downwelling surface flux snapshot plot data",
                        "rad_tran_infile": rad_tran_infile,
                        "rad_tran_outfile": rad_tran_outfile,
                        "ml3drt_outfile": ml3drt_outfile
                    })

                plot_data["vwp"].attrs["long_name"] = "Vertical cloud water path"
                plot_data["vwp"].attrs["units"] = "g m-2"
                plot_data["flux_sfc_dn_rt"].attrs["long_name"] = "Downwelling shortwave surface flux, ray-tracer"
                plot_data["flux_sfc_dn_rt"].attrs["units"] = "W m-2"
                plot_data["flux_sfc_dn_ml3drt"].attrs["long_name"] = "Downwelling shortwave surface flux, ML3DRT emulator"
                plot_data["flux_sfc_dn_ml3drt"].attrs["units"] = "W m-2"
                plot_data["flux_sfc_dn_ts"].attrs["long_name"] = "Downwelling shortwave surface flux, two-stream"
                plot_data["flux_sfc_dn_ts"].attrs["units"] = "W m-2"
                plot_data["flux_sfc_dn_ml3drt_diff"].attrs["long_name"] = "ML3DRT emulator minus ray-tracer downwelling shortwave surface flux"
                plot_data["flux_sfc_dn_ml3drt_diff"].attrs["units"] = "W m-2"
                plot_data["flux_sfc_dn_ts_diff"].attrs["long_name"] = "Two-stream minus ray-tracer downwelling shortwave surface flux"
                plot_data["flux_sfc_dn_ts_diff"].attrs["units"] = "W m-2"
                plot_data["mnn_times"].attrs["long_name"] = "Time since simulation start"
                plot_data["mnn_times"].attrs["units"] = "h"
                plot_data["mnn_szas"].attrs["long_name"] = "Solar zenith angle"
                plot_data["mnn_szas"].attrs["units"] = "degrees"
                plot_data["x"].attrs["units"] = "km"
                plot_data["y"].attrs["units"] = "km"
                plot_data["xh"].attrs["units"] = "km"
                plot_data["yh"].attrs["units"] = "km"

                plot_data.to_netcdf(plot_data_filepath)
            else:
                msg: str = "Reading plot data from {}...".format(plot_data_filepath)
                print_msg(msg)

                with xr.open_dataset(plot_data_filepath) as plot_data_in:
                    plot_data: xr.Dataset = plot_data_in.load()

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max: NP_REAL = NP_REAL(plot_data["vwp"].max())
            vwp_min: NP_REAL = NP_REAL(plot_data["vwp"].min())

            flux_sfc_dn_max: NP_REAL = NP_REAL(max(
                NP_REAL(plot_data["flux_sfc_dn_rt"].max()),
                NP_REAL(plot_data["flux_sfc_dn_ml3drt"].max()),
                NP_REAL(plot_data["flux_sfc_dn_ts"].max())))
            flux_sfc_dn_min: NP_REAL = NP_REAL(min(
                NP_REAL(plot_data["flux_sfc_dn_rt"].min()),
                NP_REAL(plot_data["flux_sfc_dn_ml3drt"].min()),
                NP_REAL(plot_data["flux_sfc_dn_ts"].min())))

            flux_sfc_dn_diff_max: NP_REAL = NP_REAL(max(
                NP_REAL(np.abs(plot_data["flux_sfc_dn_ml3drt_diff"]).max()),
                NP_REAL(np.abs(plot_data["flux_sfc_dn_ts_diff"]).max())))

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting data..."
            print_msg(msg)

            nrows: NP_INT = NP_INT(3)
            ncols: NP_INT = NP_INT(2)
            fig_width: NP_REAL = NP_REAL(6.5)
            fig_height: NP_REAL = NP_REAL(7.5)
            fig_size: list[NP_REAL] = [fig_width, fig_height]

            linthresh: NP_REAL = NP_REAL(100.)

            if flux_sfc_dn_diff_max <= 0.:
                flux_sfc_dn_diff_max = linthresh

            mnn_szas_plot: NP_ARRAY[NP_REAL] = np.array(plot_data["mnn_szas"],
                dtype = NP_REAL)

            x_ticks: NP_ARRAY[NP_REAL] = find_plot_ticks(plot_data["xh"])
            y_ticks: NP_ARRAY[NP_REAL] = find_plot_ticks(plot_data["yh"])
            x_lim: list[NP_REAL] = [NP_REAL(x_ticks[0]), NP_REAL(x_ticks[-1])]
            y_lim: list[NP_REAL] = [NP_REAL(y_ticks[0]), NP_REAL(y_ticks[-1])]

            ll: int
            for ll in range(0, ncols):
                fig = plt.figure(
                    constrained_layout = True,
                    figsize = fig_size)

                gs = fig.add_gridspec(
                    nrows = nrows,
                    ncols = 4,
                    width_ratios = [0.06, 1.0, 1.0, 0.06])

                axs = np.empty((nrows, ncols), dtype = object)

                axs[0,0] = fig.add_subplot(gs[0,1])
                axs[0,1] = fig.add_subplot(gs[0,2],
                    sharex = axs[0,0], sharey = axs[0,0])
                axs[1,0] = fig.add_subplot(gs[1,1],
                    sharex = axs[0,0], sharey = axs[0,0])
                axs[1,1] = fig.add_subplot(gs[1,2],
                    sharex = axs[0,0], sharey = axs[0,0])
                axs[2,0] = fig.add_subplot(gs[2,1],
                    sharex = axs[0,0], sharey = axs[0,0])
                axs[2,1] = fig.add_subplot(gs[2,2],
                    sharex = axs[0,0], sharey = axs[0,0])

                flux_sfc_dn_cax = fig.add_subplot(gs[:,0])
                vwp_cax = fig.add_subplot(gs[0,3])
                flux_sfc_dn_diff_cax = fig.add_subplot(gs[1:3,3])

                #---------------------------------------------------------------
                # Column 0: Fluxes
                #---------------------------------------------------------------
                flux_sfc_dn_rt_pcm: MPL_PCOLORMESH = axs[0,0].pcolormesh(
                    plot_data["xh"], 
                    plot_data["yh"], 
                    plot_data["flux_sfc_dn_rt"].isel(sza = ll),
                    norm = colors.LogNorm(
                        vmin = flux_sfc_dn_min,
                        vmax = flux_sfc_dn_max),
                    cmap = flux_cmap, shading = "flat")

                flux_sfc_dn_ts_pcm: MPL_PCOLORMESH = axs[1,0].pcolormesh(
                    plot_data["xh"], 
                    plot_data["yh"], 
                    plot_data["flux_sfc_dn_ts"].isel(sza = ll),
                    norm = colors.LogNorm(
                        vmin = flux_sfc_dn_min,
                        vmax = flux_sfc_dn_max),
                    cmap = flux_cmap, shading = "flat")

                flux_sfc_dn_ml3drt_pcm: MPL_PCOLORMESH = axs[2,0].pcolormesh(
                    plot_data["xh"], 
                    plot_data["yh"], 
                    plot_data["flux_sfc_dn_ml3drt"].isel(sza = ll),
                    norm = colors.LogNorm(
                        vmin = flux_sfc_dn_min,
                        vmax = flux_sfc_dn_max),
                    cmap = flux_cmap, shading = "flat")

                #---------------------------------------------------------------
                # Column 1, Row 0: Vertical Water Path
                #---------------------------------------------------------------
                vwp_pcm: MPL_PCOLORMESH = axs[0,1].pcolormesh(
                    plot_data["xh"], 
                    plot_data["yh"], 
                    plot_data["vwp"].isel(sza = ll),
                    norm = colors.LogNorm(
                        vmin = max(1.e1, vwp_min),
                        vmax = vwp_max),
                    cmap = cw_cmap, shading = "flat")

                #---------------------------------------------------------------
                # Column 1, Rows 1-2: Differences
                #---------------------------------------------------------------
                flux_sfc_dn_ts_diff_pcm: MPL_PCOLORMESH = axs[1,1].pcolormesh(
                    plot_data["xh"], 
                    plot_data["yh"], 
                    plot_data["flux_sfc_dn_ts_diff"].isel(sza = ll),
                    norm = colors.SymLogNorm(
                        linthresh = linthresh,
                        vmin = -flux_sfc_dn_diff_max,
                        vmax = flux_sfc_dn_diff_max),
                    cmap = diff_cmap, shading = "flat")
                axs[1,1].contour(
                    plot_data["x"],
                    plot_data["y"],
                    plot_data["flux_sfc_dn_ts_diff"].isel(sza = ll),
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

                flux_sfc_dn_ml3drt_diff_pcm: MPL_PCOLORMESH = axs[2,1].pcolormesh(
                    plot_data["xh"], 
                    plot_data["yh"], 
                    plot_data["flux_sfc_dn_ml3drt_diff"].isel(sza = ll),
                    norm = colors.SymLogNorm(
                        linthresh = linthresh,
                        vmin = -flux_sfc_dn_diff_max,
                        vmax = flux_sfc_dn_diff_max),
                    cmap = diff_cmap, shading = "flat")
                axs[2,1].contour(
                    plot_data["x"],
                    plot_data["y"],
                    plot_data["flux_sfc_dn_ml3drt_diff"].isel(sza = ll),
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

                flux_sfc_dn_cbar = fig.colorbar(
                    flux_sfc_dn_rt_pcm, cax = flux_sfc_dn_cax)
                vwp_cbar = fig.colorbar(
                    vwp_pcm, cax = vwp_cax, extend = "min")
                flux_sfc_dn_diff_cbar = fig.colorbar(
                    flux_sfc_dn_ts_diff_pcm, cax = flux_sfc_dn_diff_cax)

                flux_sfc_dn_cbar.ax.yaxis.set_ticks_position("left")
                flux_sfc_dn_cbar.ax.yaxis.set_label_position("left")

                #---------------------------------------------------------------
                # Plot contours at major ticks
                #---------------------------------------------------------------
                # Downwelling surface flux
                flux_sfc_dn_levels: NP_ARRAY[NP_REAL] = NP_REAL(
                    flux_sfc_dn_cbar.ax.get_yticks())

                axs[0,0].contour(
                    plot_data["x"],
                    plot_data["y"],
                    plot_data["flux_sfc_dn_rt"].isel(sza = ll),
                    levels = flux_sfc_dn_levels,
                    colors = "k",
                    linewidths = 1.0
                )
                axs[1,0].contour(
                    plot_data["x"],
                    plot_data["y"],
                    plot_data["flux_sfc_dn_ts"].isel(sza = ll),
                    levels = flux_sfc_dn_levels,
                    colors = "k",
                    linewidths = 1.0
                )
                axs[2,0].contour(
                    plot_data["x"],
                    plot_data["y"],
                    plot_data["flux_sfc_dn_ml3drt"].isel(sza = ll),
                    levels = flux_sfc_dn_levels,
                    colors = "k",
                    linewidths = 1.0
                )
                
                level: NP_REAL
                for level in flux_sfc_dn_levels:
                    flux_sfc_dn_cbar.ax.axhline(
                        level,
                        color = "k",
                        linewidth = 1.0,
                        linestyle = "solid"
                    )

                # VWP
                vwp_levels: NP_ARRAY[NP_REAL] = NP_REAL(vwp_cbar.ax.get_yticks())
                axs[0,1].contour(
                    plot_data["x"],
                    plot_data["y"],
                    plot_data["vwp"].isel(sza = ll),
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

                #---------------------------------------------------------------
                # Labels
                #---------------------------------------------------------------
                fig.suptitle(
                    r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$"
                    + r" - SZA {:.1f}$^{{\circ}}$".format(mnn_szas_plot[ll]))
                fig.supxlabel(r"x $\left[ km \right]$")
                fig.supylabel(r"y $\left[ km \right]$")

                axs[0,0].set_title(r"Ray-Tracer")
                axs[1,0].set_title(r"Two-Stream")
                axs[2,0].set_title(r"Emulator")

                axs[1,1].set_title(r"Two-Stream - Ray-Tracer")
                axs[2,1].set_title(r"Emulator - Ray-Tracer")

                vwp_cbar.ax.set_ylabel(
                    r"Vertical CWP $\left[ g\,m^{-2} \right]$")
                flux_sfc_dn_diff_cbar.ax.set_ylabel(r"Difference")

                #---------------------------------------------------------------
                # Additional Colorbar Elements
                #---------------------------------------------------------------
                flux_sfc_dn_diff_cbar.ax.axhline(
                    linthresh,
                    color = "k",
                    linestyle = "solid",
                    linewidth = 1.0
                )
                flux_sfc_dn_diff_cbar.ax.axhline(
                    -linthresh,
                    color = "k",
                    linestyle = "dashed",
                    linewidth = 1.0
                )

                #---------------------------------------------------------------
                # Additional figure styling
                #---------------------------------------------------------------
                # Aspect ratio and axis ticks
                mm: int
                nn: int
                for mm in range(0, nrows):
                    for nn in range(0, ncols):
                        axs[mm,nn].set_aspect("equal")
                        axs[mm,nn].set_xlim(x_lim)
                        axs[mm,nn].set_ylim(y_lim)
                        axs[mm,nn].set_xticks(x_ticks)
                        axs[mm,nn].set_yticks(y_ticks)
                        axs[mm,nn].tick_params(
                            labelbottom = (mm == nrows - 1),
                            labelleft = (nn == 0))

                #---------------------------------------------------------------
                # Save the plot to file
                #---------------------------------------------------------------
                sza_str: str = "sza_{:02d}".format(NP_INT(np.round(mnn_szas_plot[ll])))
                plt_filename: str = "ml3drt_flux_sfc_dn_snapshot.{}.{}.{}.png".format(
                    lr_str, day_str, sza_str)
                plt_filepath: str = os.path.join(rad_tran_vizdir, plt_filename)
                fig.savefig(plt_filepath, dpi = 200)
                plt.close(fig)

if __name__ == "__main__":
    main()