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
    calc_cloud_wc, calc_sw_heating, calc_sw_flux_abs, calc_sw_flux_sfc_dn, calc_sw_flux_tod_up

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-rad-tran-snapshot"
prog_desc: str = "Visualize absorbed shortwave flux and atmospheric heating rates for RTE-RRTMGP-CPP."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time: str = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line input...".format(current_time)
    print(msg, flush = True)

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
    parser.add_argument("--zmax", nargs = "?", default = 16., type = float,
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
    zmax: NP_REAL = NP_REAL(args.zmax)

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
    sw_flux_tod_up_rt: dict = {}
    sw_flux_tod_up_ts: dict = {}
    sw_flux_sfc_dn_rt: dict = {}
    sw_flux_sfc_dn_ts: dict = {}
    sw_flux_abs_rt: dict = {}
    sw_flux_abs_ts: dict = {}
    sw_heating_rt: dict = {}
    sw_heating_ts: dict = {}

    #---------------------------------------------------------------------------
    # Loop through resolutions
    #---------------------------------------------------------------------------
    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, coarse_factor_str)
        print(msg, flush = True)

        #-----------------------------------------------------------------------
        # Set up dicts for this resolution
        #-----------------------------------------------------------------------
        sw_flux_tod_up_rt[coarse_factor_str] = {}
        sw_flux_tod_up_ts[coarse_factor_str] = {}
        sw_flux_sfc_dn_rt[coarse_factor_str] = {}
        sw_flux_sfc_dn_ts[coarse_factor_str] = {}
        sw_flux_abs_rt[coarse_factor_str] = {}
        sw_flux_abs_ts[coarse_factor_str] = {}
        sw_heating_rt[coarse_factor_str] = {}
        sw_heating_ts[coarse_factor_str] = {}

        #-----------------------------------------------------------------------
        # Obtain daytime indices, times, SZAs
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Obtaining daytime information...".format(current_time)
        print(msg, flush = True)

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
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating upwelling top-of-domain fluxes for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_flux_tod_up_rt[coarse_factor_str][jj] = calc_sw_flux_tod_up(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "rt") # Shortwave upwelling top-of-domain flux, ray-tracer; [W m^{-2}]; [ntime, y, x]

            sw_flux_tod_up_ts[coarse_factor_str][jj] = calc_sw_flux_tod_up(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "ts") # Shortwave upwelling top-of-domain flux, two-stream; [W m^{-2}]; [ntime, y, x]

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating downwelling surface fluxes for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_flux_sfc_dn_rt[coarse_factor_str][jj] = calc_sw_flux_sfc_dn(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [ntime, y, x]

            sw_flux_sfc_dn_ts[coarse_factor_str][jj] = calc_sw_flux_sfc_dn(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [ntime, y, x]

            #-------------------------------------------------------------------
            # Calculate absorbed shortwave flux
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating absorbed fluxes for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_flux_abs_rt[coarse_factor_str][jj] = calc_sw_flux_abs(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "rt", zmax = zmax) # Shortwave absorbed flux, ray-tracer; [W m^{-3}]; [ntime, lay, y, x]

            sw_flux_abs_ts[coarse_factor_str][jj] = calc_sw_flux_abs(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "ts", zmax = zmax) # Shortwave absorbed flux, two-stream; [W m^{-3}]; [ntime, lay, y, x]

            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating heating rates for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_heating_rt[coarse_factor_str][jj] = calc_sw_heating(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "rt", zmax = zmax) # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

            sw_heating_ts[coarse_factor_str][jj] = calc_sw_heating(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "ts", zmax = zmax) # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

    #---------------------------------------------------------------------------
    # Loop through error types and create plots
    #---------------------------------------------------------------------------
    error_type: str
    for error_type in error_types:
        #-----------------------------------------------------------------------
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Setting up figure...".format(current_time)
        print(msg, flush = True)

        nrows: NP_INT = NP_INT(4)
        ncols: NP_INT = NP_INT(ndays)
        fig_height: NP_REAL = NP_REAL(6.)
        fig_base_size = np.array([(ncols / nrows) * fig_height, fig_height])
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
        ii: int
        for ii in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate errors
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating {} for day {} of {}...".format(current_time, error_type, ii, ndays - 1)
            print(msg, flush = True)

            sw_flux_sfc_dn_error: dict = {}
            sw_flux_tod_up_error: dict = {}
            sw_flux_abs_error: dict = {}
            sw_heating_error: dict = {}

            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                sw_flux_tod_up_error[coarse_factor_str] = calc_error(sw_flux_tod_up_rt[coarse_factor_str][jj],
                    sw_flux_tod_up_ts[coarse_factor_str][jj], error_type)
                sw_flux_sfc_dn_error[coarse_factor_str] = calc_error(sw_flux_sfc_dn_rt[coarse_factor_str][jj],
                    sw_flux_sfc_dn_ts[coarse_factor_str][jj], error_type)
                sw_flux_abs_error[coarse_factor_str] = calc_error(sw_flux_abs_rt[coarse_factor_str][jj],
                    sw_flux_abs_ts[coarse_factor_str][jj], error_type)
                sw_heating_error[coarse_factor_str] = calc_error(sw_heating_rt[coarse_factor_str][jj],
                    sw_heating_ts[coarse_factor_str][jj], error_type)

            #-------------------------------------------------------------------
            # Obtain quantities common across plots
            #-------------------------------------------------------------------
            hres_str_list: list[str] = []
            coarse_factor_str: str
            for coarse_factor_str in sw_flux_sfc_dn_rt.keys():
                x: XR_DATAARRAY = sw_flux_sfc_dn_rt[coarse_factor_str][jj]["x"] # [n_x]; [m]
                dx: NP_REAL = NP_REAL(x[1] - x[0]) # [m]
                if dx < 1.0e3:
                    hres_str_list += [r"{:.0f} $m$".format(dx)]
                else:
                    hres_str_list += [r"{:.2f} $km$".format(dx * 1.e-3)]
            time: XR_DATAARRAY = sw_flux_sfc_dn_rt[coarse_factor_str][jj]["time"] # [time]; [h]

            #-------------------------------------------------------------------
            # Plot errors at each resolution
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting {} for day {} of {}...".format(current_time, error_type, ii, ndays - 1)
            print(msg, flush = True)

            # Row 0 - Upwelling Top-of-Domain Flux
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[0,jj].plot(time, sw_flux_tod_up_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

            # Row 1 - Downwelling Surface Flux
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[1,jj].plot(time, sw_flux_sfc_dn_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

            # Row 2 - Absorbed Flux
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[2,jj].plot(time, sw_flux_abs_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

            # Row 3 - Heating Rate
            nplot_colors: NP_INT = NP_INT(len(plot_colors))
            ii: int
            for ii in range(0, coarse_factors.size):
                coarse_factor: NP_INT = coarse_factors[ii]
                coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

                axs[3,jj].plot(time, sw_heating_error[coarse_factor_str],
                    color = plot_colors[ii % nplot_colors], label = hres_str_list[ii])

        #-----------------------------------------------------------------------
        # Add plot elements
        #-----------------------------------------------------------------------
        # Labels
        title: str
        if error_type == "mae":
            title = "RTE-RRTMGP-CPP Mean Absolute Error"
        elif error_type == "mbe":
            title = "RTE-RRTMGP-CPP Mean Bias Error"
        elif error_type == "rmse":
            title = "RTE-RRTMGP-CPP Root-Mean-Square Error"

        fig.suptitle(title)
        fig.supxlabel(r"Time $\left[ h \right]$")

        for ii in range(0, ndays):
            col_title: str = "Day {}".format(ii)
            axs[0,ii].set_title(col_title)
        axs[0,0].set_ylabel(r"Upwelling Top-of-Domain Shortwave Flux $\left[ W\,m^{-2} \right]$")
        axs[1,0].set_ylabel(r"Downwelling Surface Shortwave Flux $\left[ W\,m^{-2} \right]$")
        axs[2,0].set_ylabel(r"Absorbed Shortwave Flux $\left[ W\,m^{-3} \right]$")
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