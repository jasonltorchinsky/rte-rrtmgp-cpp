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

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_PCOLORMESH
from consts.visual import diff_cmap, flux_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_flux_sfc_dn, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-rad-tran-sfc-snapshot"
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
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
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

    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

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
        int: jj
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            #-------------------------------------------------------------------
            # Calculate vertical water path
            #-------------------------------------------------------------------
            msg: str = "Calculating vertical water path for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj]) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
            vwp: XR_DATAARRAY = dz * cloud_wc.sum(dim = "lay") # [g m^{-2}], [time, y, x]

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
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

            #-------------------------------------------------------------------
            # Transpose fields before plotting
            #-------------------------------------------------------------------
            vwp: XR_DATAARRAY = (vwp
                .transpose("time", "x", "y")
                .load()) # [g m^{-2}], [time, y, x]
            flux_sfc_dn_rt: XR_DATAARRAY = (flux_sfc_dn_rt
                .transpose("time", "x", "y")
                .load()) # [W m^{-2}], [time, y, x]
            flux_sfc_dn_ts: XR_DATAARRAY = (flux_sfc_dn_ts
                .transpose("time", "x", "y")
                .load()) # [W m^{-2}], [time, y, x]

            #-------------------------------------------------------------------
            # Calculate differences
            #-------------------------------------------------------------------
            flux_sfc_dn_diff: XR_DATAARRAY = flux_sfc_dn_rt - flux_sfc_dn_ts

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max: list[NP_REAL] = [NP_REAL(vwp.isel(time = ll).max()) for ll in range(0, 3)]
            vwp_min: list[NP_REAL] = [NP_REAL(vwp.isel(time = ll).min()) for ll in range(0, 3)]

            flux_sfc_dn_max: list[NP_REAL] = [max(
                NP_REAL(flux_sfc_dn_rt.isel(time = ll).max()), 
                NP_REAL(flux_sfc_dn_ts.isel(time = ll).max())) for ll in range(0, 3)]
            flux_sfc_dn_min: list[NP_REAL] = [min(
                NP_REAL(flux_sfc_dn_rt.isel(time = ll).min()), 
                NP_REAL(flux_sfc_dn_ts.isel(time = ll).min())) for ll in range(0, 3)]

            flux_sfc_dn_diff_max: list[NP_REAL] = [
                NP_REAL(np.abs(flux_sfc_dn_diff).isel(time = ll).max())
                for ll in range(0, 3)]

            #-------------------------------------------------------------------
            # Rescale horizontal grids to have correct units
            #-------------------------------------------------------------------
            x: XR_DATAARRAY = grid["x"] * 1.e-3 # [m] => [km]
            y: XR_DATAARRAY = grid["y"] * 1.e-3 # [m] => [km]
            xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
            yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting data..."
            print_msg(msg)

            nrows: NP_INT = NP_INT(4)
            ncols: NP_INT = NP_INT(2)
            fig_width: NP_REAL = NP_REAL(5.5)
            fig_height: NP_REAL = NP_REAL(8.)
            fig_size: list[NP_REAL] = [fig_width, fig_height]
            fig, axs = plt.subplots(
                nrows = nrows, ncols = ncols,
                sharex = "col", sharey = True,
                constrained_layout = True,
                figsize = fig_size)

            if ncols == 1:
                axs = axs[...,None]
            elif nrows == 1:
                axs = axs[None,...]

            linthresh: NP_REAL = NP_REAL(100.)

            # Row 0: Vertical Water Path
            vwp_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                vwp_pcm[ll] = axs[0, ll].pcolormesh(xh, yh, vwp.isel(time = ll),
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
                    flux_sfc_dn_ts.isel(time = ll),
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
                    flux_sfc_dn_rt.isel(time = ll),
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
                    flux_sfc_dn_diff.isel(time = ll),
                    norm = colors.SymLogNorm(
                        linthresh = linthresh,
                        vmin = -max(flux_sfc_dn_diff_max),
                        vmax = max(flux_sfc_dn_diff_max)),
                    cmap = diff_cmap, shading = "flat")
                axs[3,ll].contour(
                    x,
                    y,
                    flux_sfc_dn_diff.isel(time = ll),
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

            vwp_cbar = fig.colorbar(vwp_pcm[0], ax = axs[0,:], extend = "min")
            flux_sfc_dn_cbar = fig.colorbar(flux_sfc_dn_ts_pcm[0], ax = axs[1:3,:])
            flux_sfc_dn_diff_cbar = fig.colorbar(flux_sfc_dn_diff_pcm[0], ax = axs[3,:])

            # Labels
            dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
            dx_str: str
            if dx < 1.e3:
                dx_str = r"{:.0f} $m$".format(dx)
            else:
                dx_str = r"{:.1f} $km$".format(dx * 1.e-3)

            fig.suptitle(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$" + " - {}".format(dx_str))
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"y $\left[ km \right]$")

            for ll in range(0, ncols):
                col_title: str = (r"SZA {:.1f}$^{{\circ}}$".format(mnn_szas[jj,ll]))
                axs[0,ll].set_title(col_title)
            axs[1,0].set_ylabel(r"Two-Stream")
            axs[2,0].set_ylabel(r"Ray-Tracer")
            axs[3,0].set_ylabel(r"Ray-Tracer - Two-Stream")

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

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_rad_tran_sfc_snapshot.{}.{}.png".format(lr_str, day_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()