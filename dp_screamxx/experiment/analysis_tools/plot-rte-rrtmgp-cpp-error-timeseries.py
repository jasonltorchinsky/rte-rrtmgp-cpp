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

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_FIGURE, MPL_AXES
from consts.visual import plot_colors
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_sw_flux_abs, calc_sw_flux_sfc_dn, calc_sw_flux_tod_up, \
    print_msg

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
    parser.add_argument("--z-max", nargs = "?", default = 16., type = float,
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
    z_max: NP_REAL = NP_REAL(args.z_max)

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
    flux_tod_up_rt: dict = {}
    flux_tod_up_ts: dict = {}
    flux_sfc_dn_rt: dict = {}
    flux_sfc_dn_ts: dict = {}
    flux_abs_rt: dict = {}
    flux_abs_ts: dict = {}
    heating_rt: dict = {}
    heating_ts: dict = {}

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
        flux_tod_up_rt[coarse_factor_str] = {}
        flux_tod_up_ts[coarse_factor_str] = {}
        flux_sfc_dn_rt[coarse_factor_str] = {}
        flux_sfc_dn_ts[coarse_factor_str] = {}
        flux_abs_rt[coarse_factor_str] = {}
        flux_abs_ts[coarse_factor_str] = {}
        heating_rt[coarse_factor_str] = {}
        heating_ts[coarse_factor_str] = {}

        #-----------------------------------------------------------------------
        # Obtain daytime indices, times, SZAs
        #-----------------------------------------------------------------------
        msg: str = "Obtaining daytime information..."
        print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
        daytime_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, daytime_indices) # Time since simulation start; [h]; [ndays, 3]
        daytime_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(daytime_indices.shape[0])

        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate upwelling top-of-domain flux
            #-------------------------------------------------------------------
            msg: str = "Calculating upwelling top-of-domain fluxes for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            flux_tod_up_rt[coarse_factor_str][jj] = calc_sw_flux_tod_up(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "rt") # Shortwave upwelling top-of-domain flux, ray-tracer; [W m^{-2}]; [time, y, x]

            flux_tod_up_ts[coarse_factor_str][jj] = calc_sw_flux_tod_up(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "ts") # Shortwave upwelling top-of-domain flux, two-stream; [W m^{-2}]; [time, y, x]

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
            msg: str = "Calculating downwelling surface fluxes for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            flux_sfc_dn_rt[coarse_factor_str][jj] = calc_sw_flux_sfc_dn(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

            flux_sfc_dn_ts[coarse_factor_str][jj] = calc_sw_flux_sfc_dn(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

            #-------------------------------------------------------------------
            # Calculate absorbed shortwave flux
            #-------------------------------------------------------------------
            msg: str = "Calculating absorbed fluxes for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            flux_abs_rt[coarse_factor_str][jj] = calc_sw_flux_abs(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max = z_max,
                solver = "rt") # Shortwave absorbed flux, ray-tracer; [W m^{-3}]; [ntime, lay, y, x]

            flux_abs_ts[coarse_factor_str][jj] = calc_sw_flux_abs(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max = z_max,
                solver = "ts") # Shortwave absorbed flux, two-stream; [W m^{-3}]; [ntime, lay, y, x]

            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
            msg: str = "Calculating heating rates for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            heating_rt[coarse_factor_str][jj] = calc_sw_heating(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max = z_max,
                solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

            heating_ts[coarse_factor_str][jj] = calc_sw_heating(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max = z_max,
                solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

    #---------------------------------------------------------------------------
    # Loop through error types and create plots
    #---------------------------------------------------------------------------
    error_type: str
    for error_type in error_types:
        #-----------------------------------------------------------------------
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        nrows: NP_INT = NP_INT(4)
        ncols: NP_INT = NP_INT(ndays)
        fig_height: NP_REAL = NP_REAL(6.)
        fig_base_size = np.array([fig_height, fig_height])
        fig: MPL_FIGURE
        axs: MPL_AXES
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = 3. * fig_base_size)

        if len(axs.shape) == 1:
            axs = axs[...,None]

        #-----------------------------------------------------------------------
        # Loop through days
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate errors
            #-------------------------------------------------------------------
            msg: str = "Calculating {} for day {} of {}...".format(error_type, jj, ndays - 1)
            print_msg(msg)

            flux_sfc_dn_error: dict = {}
            flux_tod_up_error: dict = {}
            flux_abs_error: dict = {}
            heating_error: dict = {}

            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                flux_tod_up_error[coarse_factor_str] = calc_error(
                    flux_tod_up_rt[coarse_factor_str][jj],
                    flux_tod_up_ts[coarse_factor_str][jj],
                    error_type)
                flux_sfc_dn_error[coarse_factor_str] = calc_error(
                    flux_sfc_dn_rt[coarse_factor_str][jj],
                    flux_sfc_dn_ts[coarse_factor_str][jj],
                    error_type)
                flux_abs_error[coarse_factor_str] = calc_error(
                    flux_abs_rt[coarse_factor_str][jj],
                    flux_abs_ts[coarse_factor_str][jj],
                    error_type)
                heating_error[coarse_factor_str] = calc_error(
                    heating_rt[coarse_factor_str][jj],
                    heating_ts[coarse_factor_str][jj],
                    error_type)

            #-------------------------------------------------------------------
            # Obtain quantities common across plots
            #-------------------------------------------------------------------
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

            #-------------------------------------------------------------------
            # Plot errors at each resolution
            #-------------------------------------------------------------------
            msg: str = "Plotting {} for day {} of {}...".format(error_type, jj, ndays - 1)
            print_msg(msg)

            # Row 0 - Upwelling Top-of-Domain Flux
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[0,jj].plot(time, flux_tod_up_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

            # Row 1 - Downwelling Surface Flux
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[1,jj].plot(time, flux_sfc_dn_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

            # Row 2 - Absorbed Flux
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[2,jj].plot(time, flux_abs_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

            # Row 3 - Heating Rate
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[3,jj].plot(time, heating_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

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
                axs[ll,jj].axvline(time_xticks[1], color = "gray", linestyle = "solid", linewidth = 0.5)

                ax_2: MPL_AXES = axs[ll,jj].secondary_xaxis("top")
                if ll == 0:
                    ax_2.set_xticks(time_xticks, labels = sza_xtick_labels)
                else:
                    ax_2.set_xticks(time_xticks, labels = [None, None, None])

        #-----------------------------------------------------------------------
        # Add plot elements
        #-----------------------------------------------------------------------
        # Labels
        error_str: str
        if error_type == "mae":
            error_str = "Mean Absolute Error"
        elif error_type == "mbe":
            error_str = "Mean Bias Error"
        elif error_type == "rmse":
            error_str = "Root-Mean-Square Error"
        title_str: str = "RTE-RRTMGP-CPP " + error_str + " Time Series"

        fig.suptitle(title_str)
        fig.supxlabel(r"Time $\left[ h \right]$")

        for ii in range(0, ndays):
            col_title: str = "Day {}".format(ii)
            axs[0,ii].set_title(col_title)
        axs[0,0].set_ylabel(r"Upwelling Top-of-Domain Flux $\left[ W\,m^{-2} \right]$")
        axs[1,0].set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")
        axs[2,0].set_ylabel(r"Absorbed Flux $\left[ W\,m^{-3} \right]$")
        axs[3,0].set_ylabel(r"Atmospheric Heating Rate $\left[ K\,d^{-1} \right]$")

        axs[0,0].legend()
        
        # Axes Scaling, Limits
        if error_type in ["mae", "rmse"]:
            for ax in axs.flatten():
                ax.set_yscale("log")
        elif error_type in ["mbe"]:
            for ax in axs.flatten():
                ylim: tuple[NP_REAL] = ax.get_ylim()
                ymax: NP_REAL = np.abs(ylim).max()
                ax.set_ylim([-ymax, ymax])
                ax.axhline(0, color = "gray", linewidth = 0.5, linestyle = "solid")

        #-------------------------------------------------------------------
        # Save the plot to file
        #-------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_{}_timeseries.png".format(error_type)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

def calc_error(rt_field: XR_DATAARRAY, ts_field: XR_DATAARRAY, error_type: str):
    assert(error_type in ["mae", "mbe", "rmse"])

    space_dims: list[str] = [dim for dim in rt_field.dims if dim != "time"]

    error: XR_DATAARRAY
    if error_type == "mae":
        error = (np.abs(ts_field - rt_field)).mean(dim = space_dims)
    elif error_type == "mbe":
        error = ts_field.mean(dim = space_dims) - rt_field.mean(dim = space_dims)
    elif error_type == "rmse":
        error = np.sqrt((np.pow(ts_field - rt_field, 2).mean(dim = space_dims)))

    return error

if __name__ == "__main__":
    main()