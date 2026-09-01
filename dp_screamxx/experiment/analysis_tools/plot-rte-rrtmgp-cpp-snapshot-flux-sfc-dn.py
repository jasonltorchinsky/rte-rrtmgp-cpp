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
    calc_cloud_wc, calc_sw_flux_sfc_dn, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-snapshot-flux-sfc-dn"
prog_desc: str = "Visualize surface state for RTE-RRTMGP-CPP."

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
        help = "Re-calculate quantities for plotting and save them to file.")
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

    lr_re: re.Pattern = re.compile("lr_..")

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Calculate or read data for plotting
        #-----------------------------------------------------------------------
        nc_filename: str = "rte_rrtmgp_cpp_rad_tran_sfc_snapshot.{}.nc".format(lr_str)
        nc_filepath: str = os.path.join(working_dir, nc_filename)

        required_data_vars: list[str] = [
            "vwp",
            "flux_sfc_dn_rt",
            "flux_sfc_dn_ts",
            "flux_sfc_dn_diff",
            "mnn_indices",
            "mnn_times",
            "mnn_szas"
        ]
        required_coords: list[str] = [
            "day",
            "mnn",
            "x",
            "y",
            "xh",
            "yh"
        ]
        required_dims: list[str] = [
            "day",
            "mnn",
            "x",
            "y",
            "xh",
            "yh"
        ]

        plot_data: xr.Dataset
        recalculate_data: bool = recalculate
        if (not recalculate_data) and os.path.exists(nc_filepath):
            msg: str = "Reading calculated quantities from {}...".format(nc_filepath)
            print_msg(msg)

            try:
                plot_data = xr.open_dataset(nc_filepath)
                plot_data.load()
                plot_data.close()

                missing_data_vars: list[str] = [
                    var_name for var_name in required_data_vars
                    if var_name not in plot_data.data_vars
                ]
                missing_coords: list[str] = [
                    coord_name for coord_name in required_coords
                    if coord_name not in plot_data.coords
                ]
                missing_dims: list[str] = [
                    dim_name for dim_name in required_dims
                    if dim_name not in plot_data.sizes
                ]

                if (len(missing_data_vars) > 0) or (len(missing_coords) > 0) or (len(missing_dims) > 0):
                    msg = "Calculated quantities file does not contain all necessary information. Recalculating quantities for plotting..."
                    print_msg(msg)
                    recalculate_data = True
            except Exception:
                msg = "Unable to read calculated quantities file. Recalculating quantities for plotting..."
                print_msg(msg)
                recalculate_data = True
        elif not recalculate_data:
            msg = "Calculated quantities not found. Calculating quantities for plotting..."
            print_msg(msg)
            recalculate_data = True

        if recalculate_data:
            if recalculate:
                msg: str = "Recalculating quantities for plotting..."
                print_msg(msg)

            #-------------------------------------------------------------------
            # Obtain grid information
            #-------------------------------------------------------------------
            msg: str = "Obtaining grid information..."
            print_msg(msg)
            grid: dict = find_grid(rad_tran_infile)

            dz: NP_REAL = NP_REAL(grid["zh"][1] - grid["zh"][0])

            #-------------------------------------------------------------------
            # Obtain Morning-Noon-Night time indices, times, SZAs
            #-------------------------------------------------------------------
            msg: str = "Obtaining morning-noon-night information..."
            print_msg(msg)

            mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
            mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
            mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
            ndays: NP_INT = NP_INT(mnn_indices.shape[0])

            day: NP_ARRAY[NP_INT] = np.arange(0, ndays, dtype = NP_INT)
            mnn: NP_ARRAY[NP_INT] = np.arange(0, 3, dtype = NP_INT)

            vwp_list: list[XR_DATAARRAY] = []
            flux_sfc_dn_rt_list: list[XR_DATAARRAY] = []
            flux_sfc_dn_ts_list: list[XR_DATAARRAY] = []
            flux_sfc_dn_diff_list: list[XR_DATAARRAY] = []

            #-------------------------------------------------------------------
            # Calculate fields for each MNN of each day
            #-------------------------------------------------------------------
            jj: int
            for jj in range(0, ndays):
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

                flux_sfc_dn_rt: XR_DATAARRAY = calc_sw_flux_sfc_dn(
                    rad_tran_outfile,
                    mnn_indices[jj],
                    solver = "rt") # Downwelling surface flux - ray-tracer; [W m^{-2}]; [time, y, x]
                flux_sfc_dn_ts: XR_DATAARRAY = calc_sw_flux_sfc_dn(
                    rad_tran_outfile,
                    mnn_indices[jj],
                    solver = "ts") # Downwelling surface flux - two-stream; [W m^{-2}]; [time, y, x]

                #---------------------------------------------------------------
                # Transpose fields before saving and plotting
                #---------------------------------------------------------------
                vwp = (vwp
                    .transpose("time", "x", "y")
                    .rename({"time": "mnn"})
                    .assign_coords(mnn = mnn)
                    .load()) # [g m^{-2}], [mnn, x, y]
                flux_sfc_dn_rt = (flux_sfc_dn_rt
                    .transpose("time", "x", "y")
                    .rename({"time": "mnn"})
                    .assign_coords(mnn = mnn)
                    .load()) # [W m^{-2}], [mnn, x, y]
                flux_sfc_dn_ts = (flux_sfc_dn_ts
                    .transpose("time", "x", "y")
                    .rename({"time": "mnn"})
                    .assign_coords(mnn = mnn)
                    .load()) # [W m^{-2}], [mnn, x, y]

                #---------------------------------------------------------------
                # Calculate differences
                #---------------------------------------------------------------
                flux_sfc_dn_diff: XR_DATAARRAY = flux_sfc_dn_ts - flux_sfc_dn_rt

                vwp_list.append(vwp)
                flux_sfc_dn_rt_list.append(flux_sfc_dn_rt)
                flux_sfc_dn_ts_list.append(flux_sfc_dn_ts)
                flux_sfc_dn_diff_list.append(flux_sfc_dn_diff)

            #-------------------------------------------------------------------
            # Combine calculated quantities and save to netCDF
            #-------------------------------------------------------------------
            msg: str = "Saving calculated quantities to {}...".format(nc_filepath)
            print_msg(msg)

            vwp_all: XR_DATAARRAY = (xr.concat(vwp_list, dim = "day")
                .assign_coords(day = day)
                .transpose("day", "mnn", "x", "y")
                .load())
            flux_sfc_dn_rt_all: XR_DATAARRAY = (xr.concat(flux_sfc_dn_rt_list, dim = "day")
                .assign_coords(day = day)
                .transpose("day", "mnn", "x", "y")
                .load())
            flux_sfc_dn_ts_all: XR_DATAARRAY = (xr.concat(flux_sfc_dn_ts_list, dim = "day")
                .assign_coords(day = day)
                .transpose("day", "mnn", "x", "y")
                .load())
            flux_sfc_dn_diff_all: XR_DATAARRAY = (xr.concat(flux_sfc_dn_diff_list, dim = "day")
                .assign_coords(day = day)
                .transpose("day", "mnn", "x", "y")
                .load())

            plot_data = xr.Dataset(
                data_vars = {
                    "vwp": (
                        ["day", "mnn", "x", "y"],
                        vwp_all.data,
                        {"long_name": "vertical cloud water path", "units": "g m^{-2}"}
                    ),
                    "flux_sfc_dn_rt": (
                        ["day", "mnn", "x", "y"],
                        flux_sfc_dn_rt_all.data,
                        {"long_name": "downwelling surface flux - ray-tracer", "units": "W m^{-2}"}
                    ),
                    "flux_sfc_dn_ts": (
                        ["day", "mnn", "x", "y"],
                        flux_sfc_dn_ts_all.data,
                        {"long_name": "downwelling surface flux - two-stream", "units": "W m^{-2}"}
                    ),
                    "flux_sfc_dn_diff": (
                        ["day", "mnn", "x", "y"],
                        flux_sfc_dn_diff_all.data,
                        {"long_name": "downwelling surface flux difference - two-stream minus ray-tracer", "units": "W m^{-2}"}
                    ),
                    "mnn_indices": (
                        ["day", "mnn"],
                        np.array(mnn_indices, dtype = NP_INT),
                        {"long_name": "morning-noon-night time indices"}
                    ),
                    "mnn_times": (
                        ["day", "mnn"],
                        np.array(mnn_times, dtype = NP_REAL),
                        {"long_name": "morning-noon-night times since simulation start", "units": "h"}
                    ),
                    "mnn_szas": (
                        ["day", "mnn"],
                        np.array(mnn_szas, dtype = NP_REAL),
                        {"long_name": "morning-noon-night solar zenith angles", "units": "degrees"}
                    )
                },
                coords = {
                    "day": day,
                    "mnn": mnn,
                    "x": (
                        ["x"],
                        np.array(grid["x"] * 1.e-3, dtype = NP_REAL),
                        {"long_name": "x-coordinate", "units": "km"}
                    ),
                    "y": (
                        ["y"],
                        np.array(grid["y"] * 1.e-3, dtype = NP_REAL),
                        {"long_name": "y-coordinate", "units": "km"}
                    ),
                    "xh": (
                        ["xh"],
                        np.array(grid["xh"] * 1.e-3, dtype = NP_REAL),
                        {"long_name": "x-coordinate cell edges", "units": "km"}
                    ),
                    "yh": (
                        ["yh"],
                        np.array(grid["yh"] * 1.e-3, dtype = NP_REAL),
                        {"long_name": "y-coordinate cell edges", "units": "km"}
                    )
                },
                attrs = {
                    "source_input_file": rad_tran_infile,
                    "source_output_file": rad_tran_outfile,
                    "horizontal_resolution": lr_str
                }
            )

            plot_data.to_netcdf(nc_filepath)

        ndays: NP_INT = NP_INT(plot_data.sizes["day"])

        #-----------------------------------------------------------------------
        # Plot fields for each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            vwp: XR_DATAARRAY = plot_data["vwp"].isel(day = jj)
            flux_sfc_dn_rt: XR_DATAARRAY = plot_data["flux_sfc_dn_rt"].isel(day = jj)
            flux_sfc_dn_ts: XR_DATAARRAY = plot_data["flux_sfc_dn_ts"].isel(day = jj)
            flux_sfc_dn_diff: XR_DATAARRAY = plot_data["flux_sfc_dn_diff"].isel(day = jj)

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max: list[NP_REAL] = [NP_REAL(vwp.isel(mnn = ll).max()) for ll in range(0, 3)]
            vwp_min: list[NP_REAL] = [NP_REAL(vwp.isel(mnn = ll).min()) for ll in range(0, 3)]

            flux_sfc_dn_max: list[NP_REAL] = [max(
                NP_REAL(flux_sfc_dn_rt.isel(mnn = ll).max()), 
                NP_REAL(flux_sfc_dn_ts.isel(mnn = ll).max())) for ll in range(0, 3)]
            flux_sfc_dn_min: list[NP_REAL] = [min(
                NP_REAL(flux_sfc_dn_rt.isel(mnn = ll).min()), 
                NP_REAL(flux_sfc_dn_ts.isel(mnn = ll).min())) for ll in range(0, 3)]

            flux_sfc_dn_diff_max: list[NP_REAL] = [
                NP_REAL(np.abs(flux_sfc_dn_diff).isel(mnn = ll).max())
                for ll in range(0, 3)]

            #-------------------------------------------------------------------
            # Obtain horizontal grids
            #-------------------------------------------------------------------
            x: XR_DATAARRAY = plot_data["x"] # [km]
            y: XR_DATAARRAY = plot_data["y"] # [km]
            xh: XR_DATAARRAY = plot_data["xh"] # [km]
            yh: XR_DATAARRAY = plot_data["yh"] # [km]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting data..."
            print_msg(msg)

            nrows: NP_INT = NP_INT(4)
            ncols: NP_INT = NP_INT(2)
            fig_width: NP_REAL = NP_REAL(6.0)
            fig_height: NP_REAL = NP_REAL(8.)
            fig_size: list[NP_REAL] = [fig_width, fig_height]

            fig = plt.figure(figsize = fig_size)

            gs = fig.add_gridspec(
                nrows = nrows,
                ncols = ncols + 1,
                left = 0.13,
                right = 0.80,
                bottom = 0.08,
                top = 0.92,
                wspace = 0.08,
                hspace = 0.10,
                width_ratios = [1., 1., 0.06]
            )

            axs = np.empty((nrows, ncols), dtype = object)
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    if (ll == 0) and (mm == 0):
                        axs[ll,mm] = fig.add_subplot(gs[ll,mm])
                    else:
                        axs[ll,mm] = fig.add_subplot(
                            gs[ll,mm],
                            sharex = axs[0,0],
                            sharey = axs[0,0]
                        )

            vwp_cax = fig.add_subplot(gs[0, ncols])
            flux_sfc_dn_cax = fig.add_subplot(gs[1:3, ncols])
            flux_sfc_dn_diff_cax = fig.add_subplot(gs[3, ncols])

            linthresh: NP_REAL = NP_REAL(100.)

            # Row 0: Vertical Water Path
            vwp_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                vwp_pcm[ll] = axs[0,ll].pcolormesh(
                    xh, 
                    yh, 
                    vwp.isel(mnn = ll),
                    norm = colors.LogNorm(
                        vmin = max(1.e1, min(vwp_min)),
                        vmax = max(vwp_max)),
                    cmap = cw_cmap, shading = "flat")

            # Row 1: Downwelling surface flux, Two-Stream
            flux_sfc_dn_ts_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                flux_sfc_dn_ts_pcm[ll] = axs[1,ll].pcolormesh(
                    xh, 
                    yh, 
                    flux_sfc_dn_ts.isel(mnn = ll),
                    norm = colors.LogNorm(
                        vmin = min(flux_sfc_dn_min),
                        vmax = max(flux_sfc_dn_max)),
                    cmap = flux_cmap, shading = "flat")

            # Row 2: Downwelling surface flux, Ray-Tracer
            flux_sfc_dn_rt_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                flux_sfc_dn_rt_pcm[ll] = axs[2,ll].pcolormesh(
                    xh, 
                    yh, 
                    flux_sfc_dn_rt.isel(mnn = ll),
                    norm = colors.LogNorm(
                        vmin = min(flux_sfc_dn_min),
                        vmax = max(flux_sfc_dn_max)),
                    cmap = flux_cmap, shading = "flat")

            # Row 3: Downwelling surface flux, Difference
            flux_sfc_dn_diff_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                flux_sfc_dn_diff_pcm[ll] = axs[3,ll].pcolormesh(
                    xh, 
                    yh, 
                    flux_sfc_dn_diff.isel(mnn = ll),
                    norm = colors.SymLogNorm(
                        linthresh = linthresh,
                        vmin = -max(flux_sfc_dn_diff_max),
                        vmax = max(flux_sfc_dn_diff_max)),
                    cmap = diff_cmap, shading = "flat")
                axs[3,ll].contour(
                    x,
                    y,
                    flux_sfc_dn_diff.isel(mnn = ll),
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

            vwp_cbar = fig.colorbar(vwp_pcm[0], cax = vwp_cax, extend = "min")
            flux_sfc_dn_cbar = fig.colorbar(flux_sfc_dn_ts_pcm[0], cax = flux_sfc_dn_cax)
            flux_sfc_dn_diff_cbar = fig.colorbar(flux_sfc_dn_diff_pcm[0], cax = flux_sfc_dn_diff_cax)

            #-------------------------------------------------------------------
            # Plot contours at major ticks
            #-------------------------------------------------------------------
            # VWP
            vwp_levels: NP_ARRAY[NP_REAL] = np.array(vwp_cbar.ax.get_yticks(), dtype = NP_REAL)
            # Row 0: Cloud Water Content
            row: NP_INT = NP_INT(0)
            ll: int
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    x,
                    y,
                    vwp.isel(mnn = ll),
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

            # Downwelling surface flux
            flux_sfc_dn_levels: NP_ARRAY[NP_REAL] = np.array(flux_sfc_dn_cbar.ax.get_yticks(), dtype = NP_REAL)
            # Row 0: Cloud Water Content
            ll: int
            row: NP_INT = NP_INT(1)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    x,
                    y,
                    flux_sfc_dn_ts.isel(mnn = ll),
                    levels = flux_sfc_dn_levels,
                    colors = "k",
                    linewidths = 1.0
                )

            row: NP_INT = NP_INT(2)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    x,
                    y,
                    flux_sfc_dn_rt.isel(mnn = ll),
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

            #-------------------------------------------------------------------
            # Labels
            #-------------------------------------------------------------------            
            dx: NP_REAL = NP_REAL((plot_data["xh"].values[1] - plot_data["xh"].values[0]) * 1.e3) # [km] => [m]
            dx_str: str
            if dx < 1.e3:
                dx_str = r"{:.0f} $m$".format(dx)
            else:
                dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

            fig.suptitle(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$" + " - {}".format(dx_str))
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"y $\left[ km \right]$")

            for ll in range(0, ncols):
                col_title: str = (r"SZA {:.1f}$^{{\circ}}$".format(plot_data["mnn_szas"].isel(day = jj, mnn = ll)))
                axs[0,ll].set_title(col_title)
            axs[1,0].set_ylabel(r"Two-Stream")
            axs[2,0].set_ylabel(r"Ray-Tracer")
            axs[3,0].set_ylabel(r"Two-Stream - Ray-Tracer")

            vwp_cbar.ax.set_ylabel(r"Vertical CWP $\left[ g\,m^{-2} \right]$")
            flux_sfc_dn_diff_cbar.ax.set_ylabel(r"Difference")

            #-------------------------------------------------------------------
            # Additional Colorbar Elements
            #-------------------------------------------------------------------
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

            #-------------------------------------------------------------------
            # Additional figure styling
            #-------------------------------------------------------------------
            # Aspect ratio
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_aspect("equal")

            # Axis limits and ticks
            tick_spacing: NP_REAL = NP_REAL(50.)

            x_min_data: NP_REAL = NP_REAL(np.min(xh.values))
            x_max_data: NP_REAL = NP_REAL(np.max(xh.values))
            y_min_data: NP_REAL = NP_REAL(np.min(yh.values))
            y_max_data: NP_REAL = NP_REAL(np.max(yh.values))

            x_min: NP_REAL = NP_REAL(np.round(x_min_data / tick_spacing) * tick_spacing)
            x_max: NP_REAL = NP_REAL(np.round(x_max_data / tick_spacing) * tick_spacing)
            y_min: NP_REAL = NP_REAL(np.round(y_min_data / tick_spacing) * tick_spacing)
            y_max: NP_REAL = NP_REAL(np.round(y_max_data / tick_spacing) * tick_spacing)

            x_ticks: NP_ARRAY[NP_REAL] = np.arange(
                x_min,
                x_max + 0.5 * tick_spacing,
                tick_spacing,
                dtype = NP_REAL
            )
            y_ticks: NP_ARRAY[NP_REAL] = np.arange(
                y_min,
                y_max + 0.5 * tick_spacing,
                tick_spacing,
                dtype = NP_REAL
            )

            x_tick_labels: list[str] = ["{:.0f}".format(tick) for tick in x_ticks]
            y_tick_labels: list[str] = ["{:.0f}".format(tick) for tick in y_ticks]

            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_xlim(x_min, x_max)
                    axs[ll,mm].set_ylim(y_min, y_max)
                    axs[ll,mm].set_xticks(x_ticks)
                    axs[ll,mm].set_yticks(y_ticks)
                    axs[ll,mm].set_xticklabels(x_tick_labels)
                    axs[ll,mm].set_yticklabels(y_tick_labels)
                    axs[ll,mm].tick_params(
                        labelbottom = (ll == nrows - 1),
                        labelleft = (mm == 0)
                    )

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_snapshot_flux_sfc_dn.{}.{}.png".format(lr_str, day_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200, bbox_inches = "tight")
            plt.close(fig)

if __name__ == "__main__":
    main()