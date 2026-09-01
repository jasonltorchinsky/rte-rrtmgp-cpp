#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
from datetime import datetime
import re
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, MPL_FIGURE, MPL_AXES
from consts.visual import plot_colors
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_sw_flux_abs, calc_sw_flux_sfc_dn, \
    calc_sw_reflectance, calc_z_max_info, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-error-timeseries"
prog_desc: str = "Visualize error for radiative transfer for RTE-RRTMGP-CPP."

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
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
    parser.add_argument("--error-types", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Error types to calculate, e.g., mae,mbe,rmse.")

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

    error_types: list[str]
    if args.error_types is not None:
        error_types = args.error_types.split(",")
    else:
        error_types = ["mae", "mbe", "rmse"]

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

    #---------------------------------------------------------------------------
    # Calculate relevant quantities at each resolution for each day
    #---------------------------------------------------------------------------
    reflectance_rt: dict = {}
    reflectance_ts: dict = {}
    reflectance_corr: dict = {}
    flux_sfc_dn_rt: dict = {}
    flux_sfc_dn_ts: dict = {}
    flux_sfc_dn_corr: dict = {}
    heating_rt: dict = {}
    heating_ts: dict = {}
    heating_corr: dict = {}

    reflectance_anticorr: bool = False
    flux_sfc_dn_anticorr: bool = False
    flux_abs_anticorr: bool = False
    heating_anticorr: bool = False

    #---------------------------------------------------------------------------
    # Loop through resolutions
    #---------------------------------------------------------------------------
    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(coarse_factor_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Set up dicts for this resolution
        #-----------------------------------------------------------------------
        reflectance_rt[coarse_factor_str] = {}
        reflectance_ts[coarse_factor_str] = {}
        reflectance_corr[coarse_factor_str] = {}
        flux_sfc_dn_rt[coarse_factor_str] = {}
        flux_sfc_dn_ts[coarse_factor_str] = {}
        flux_sfc_dn_corr[coarse_factor_str] = {}
        heating_rt[coarse_factor_str] = {}
        heating_ts[coarse_factor_str] = {}
        heating_corr[coarse_factor_str] = {}

        #-----------------------------------------------------------------------
        # Obtain daytime indices, times, SZAs
        #-----------------------------------------------------------------------
        msg: str = "Obtaining daytime information..."
        print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
        daytime_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, daytime_indices) # Time since simulation start; [h]; [ndays, 3]
        daytime_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(daytime_indices.shape[0])
        z_max_info: dict = calc_z_max_info(rad_tran_infile, z_max = z_max)

        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate upwelling top-of-domain flux
            #-------------------------------------------------------------------
            msg: str = "Calculating reflectance for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            reflectance_rt[coarse_factor_str][jj] = calc_sw_reflectance(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "rt") # Shortwave upwelling top-of-domain flux, ray-tracer; [W m^{-2}]; [time, y, x]

            reflectance_ts[coarse_factor_str][jj] = calc_sw_reflectance(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "ts") # Shortwave upwelling top-of-domain flux, two-stream; [W m^{-2}]; [time, y, x]

            reflectance_corr[coarse_factor_str][jj] = xr.corr(
                reflectance_rt[coarse_factor_str][jj],
                reflectance_ts[coarse_factor_str][jj],
                dim = ["y", "x"]
            )

            if reflectance_corr[coarse_factor_str][jj].min() < 0.0:
                reflectance_anticorr = True

            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
            msg: str = "Calculating heating rates for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            heating_rt[coarse_factor_str][jj] = calc_sw_heating(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max_info = z_max_info,
                solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

            heating_ts[coarse_factor_str][jj] = calc_sw_heating(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max_info = z_max_info,
                solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

            heating_corr[coarse_factor_str][jj] = xr.corr(
                heating_rt[coarse_factor_str][jj],
                heating_ts[coarse_factor_str][jj],
                dim = ["lay", "y", "x"]
            )

            if heating_corr[coarse_factor_str][jj].min() < 0.0:
                heating_anticorr = True

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
            msg: str = "Calculating downwelling surface fluxes for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            flux_sfc_dn_rt[coarse_factor_str][jj] = calc_sw_flux_sfc_dn(
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

            flux_sfc_dn_ts[coarse_factor_str][jj] = calc_sw_flux_sfc_dn(
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

            flux_sfc_dn_corr[coarse_factor_str][jj] = xr.corr(
                flux_sfc_dn_rt[coarse_factor_str][jj],
                flux_sfc_dn_ts[coarse_factor_str][jj],
                dim = ["y", "x"]
            )

            if flux_sfc_dn_corr[coarse_factor_str][jj].min() < 0.0:
                flux_sfc_dn_anticorr = True
        
    #-----------------------------------------------------------------------
    # Set up figure for plotting
    #-----------------------------------------------------------------------
    msg: str = "Setting up figure..."
    print_msg(msg)

    nrows: NP_INT = NP_INT(3)
    ncols: NP_INT = NP_INT(ndays)
    fig_height: NP_REAL = NP_REAL(8.)
    fig_width: NP_REAL = (NP_REAL(ncols) / NP_REAL(nrows)) * fig_height
    fig_size: list[NP_REAL] = [fig_width, fig_height]
    fig: MPL_FIGURE
    axs: MPL_AXES
    fig, axs = plt.subplots(
        nrows = nrows, ncols = ncols,
        sharex = "col", sharey = "row",
        constrained_layout = True,
        figsize = fig_size)

    if (ncols == 1) and (nrows == 1):
        axs = np.array([[axs]])
    elif ncols == 1:
        axs = axs[...,None]
    elif nrows == 1:
        axs = axs[None,...]

    #---------------------------------------------------------------------------
    # Loop through days
    #---------------------------------------------------------------------------
    jj: int
    for jj in range(0, ndays):
        #-----------------------------------------------------------------------
        # Obtain quantities common across plots
        #-----------------------------------------------------------------------
        hres_str_list: list[str] = []
        coarse_factor_str: str
        for coarse_factor_str in flux_sfc_dn_rt.keys():
            x: XR_DATAARRAY = flux_sfc_dn_rt[coarse_factor_str][jj]["x"] # [n_x]; [m]
            dx: NP_REAL = NP_REAL(x[1] - x[0]) # [m]
            if dx < 1.0e3:
                hres_str_list += [r"{:.0f} $m$".format(dx)]
            else:
                hres_str_list += [r"{:.2f} $km$".format(dx * 1.e-3)]
        time: XR_DATAARRAY = flux_sfc_dn_rt[coarse_factor_str][jj]["time"] # [time]; [h]

        #-----------------------------------------------------------------------
        # Plot errors at each resolution
        #-----------------------------------------------------------------------
        msg: str = "Plotting correlation for day {} of {}...".format(jj, ndays - 1)
        print_msg(msg)

        nplot_colors: NP_INT = NP_INT(len(plot_colors))

        # Row 0 - Reflectance
        ii: int
        for ii in range(0, coarse_factors.size):
            coarse_factor: NP_INT = coarse_factors[ii]
            coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

            axs[0,jj].plot(time, reflectance_corr[coarse_factor_str][jj],
                linewidth = 1.0,
                color = plot_colors[ii % nplot_colors],
                label = hres_str_list[ii])

        # Row 1 - Heating Rate
        ii: int
        for ii in range(0, coarse_factors.size):
            coarse_factor: NP_INT = coarse_factors[ii]
            coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

            axs[1,jj].plot(time, heating_corr[coarse_factor_str][jj],
                linewidth = 1.0,
                color = plot_colors[ii % nplot_colors],
                label = hres_str_list[ii])

        # Row 2 - Downwelling Surface Flux
        ii: int
        for ii in range(0, coarse_factors.size):
            coarse_factor: NP_INT = coarse_factors[ii]
            coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

            axs[2,jj].plot(time, flux_sfc_dn_corr[coarse_factor_str][jj],
                linewidth = 1.0,
                color = plot_colors[ii % nplot_colors],
                label = hres_str_list[ii])

        # Common column-wise plot elements
        # x-ticks
        time: NP_ARRAY[NP_REAL] = daytime_times[jj]
        sza: NP_ARRAY[NP_REAL] = daytime_szas[jj]
        xlim: list[NP_REAL] = np.array([time[0], time[-1]], dtype = NP_REAL)
        time_xticks: NP_ARRAY[NP_REAL] = np.array([time[0], time[NP_INT(time.size/2)], time[-1]], dtype = NP_REAL)
        sza_xticks: NP_ARRAY[NP_REAL] = np.array([sza[0], sza[NP_INT(time.size/2)], sza[-1]], dtype = NP_REAL)
        sza_xtick_labels: list[str] = [r"{:.1f}$^{{\circ}}$".format(solar_zenith_angle) for solar_zenith_angle in sza_xticks]
        ll: int
        for ll in range(0, nrows):
            axs[ll,jj].set_xticks(time_xticks)
            axs[ll,jj].axvline(
                time_xticks[1], 
                color = "gray", 
                linestyle = "solid", 
                linewidth = 0.5)

            ax_2: MPL_AXES = axs[ll,jj].secondary_xaxis("top")
            if ll == 0:
                ax_2.set_xticks(time_xticks, labels = sza_xtick_labels)
            else:
                ax_2.set_xticks(time_xticks, labels = [None, None, None])

    #---------------------------------------------------------------------------
    # Add plot elements
    #---------------------------------------------------------------------------
    fig.suptitle("Correlation",
        y = 1.065)
    fig.supxlabel(r"Time $\left[ h \right]$")

    ii: int
    for ii in range(0, ndays):
        col_title: str = "Day {}".format(ii)
        axs[0,ii].set_title(col_title,
            pad = 24.0)

    axs[0,0].set_ylabel(r"Reflectance")
    axs[1,0].set_ylabel(r"Heating Rate")
    axs[2,0].set_ylabel(r"Downwelling Surface Flux")

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc = "upper center",
        bbox_to_anchor = (0.5, 1.04),
        ncol = len(labels),
        handlelength = 2.0,
        columnspacing = 1.2,
        handletextpad = 0.4
    )

    anticorr_list: list[bool] = [reflectance_anticorr, heating_anticorr, flux_sfc_dn_anticorr]
    ii: int
    for ii in range(0, len(anticorr_list)):
        anticorr: bool = anticorr_list[ii]
        if anticorr:
            for ax in axs[ii,...]:
                ax.axhline(
                    0, 
                    color = "gray", 
                    linestyle = "solid", 
                    linewidth = 0.5
                )

    #---------------------------------------------------------------------------
    # Save the plot to file
    #---------------------------------------------------------------------------
    plt_filename = "rte_rrtmgp_cpp_rad_tran_correlation.png"
    plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
    fig.savefig(plt_filepath, dpi = 512, bbox_inches = "tight")
    plt.close(fig)

if __name__ == "__main__":
    main()