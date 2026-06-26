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
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_PCOLORMESH
from consts.visual import flux_cmap, heating_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_sw_flux_abs

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

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, coarse_factor_str)
        print(msg, flush = True)

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
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Setting up figure...".format(current_time)
        print(msg, flush = True)

        nrows: NP_INT = NP_INT(4)
        ncols: NP_INT = NP_INT(ndays)
        fig_height: NP_REAL = NP_REAL(6.)
        fig_base_size = np.array([(ncols / nrows) * fig_height, fig_height])
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = 3. * fig_base_size)

        if len(axs.shape) == 1:
            axs = axs[...,None]

        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating heating rates for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_heating_rt: XR_DATAARRAY = calc_sw_heating(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "rt", zmax = zmax) # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

            sw_heating_ts: XR_DATAARRAY = calc_sw_heating(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "ts", zmax = zmax) # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

            sw_heating_diff: XR_DATAARRAY = sw_heating_ts - sw_heating_rt # Shortwave heating rate, ts - rt; [K d^{-1}]; [ntime, lay, y, x]

            #-------------------------------------------------------------------
            # Calculate absorbed shortwave flux
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating absorbed fluxes for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_flux_abs_rt: XR_DATAARRAY = calc_sw_flux_abs(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "rt", zmax = zmax) # Shortwave absorbed flux, ray-tracer; [W m^{-3}]; [ntime, lay, y, x]

            sw_flux_abs_ts: XR_DATAARRAY = calc_sw_flux_abs(rad_tran_infile, rad_tran_outfile,
                daytime_indices[jj,...], solver = "ts", zmax = zmax) # Shortwave absorbed flux, two-stream; [W m^{-3}]; [ntime, lay, y, x]

            sw_flux_abs_diff: XR_DATAARRAY = sw_flux_abs_ts - sw_flux_abs_rt # Shortwave absorbed flux, ts - rt; [W m^{-3}]; [ntime, lay, y, x]

            #-------------------------------------------------------------------
            # Obtain statistics for radiative quantities
            #-------------------------------------------------------------------
            quantiles: NP_ARRAY[NP_REAL] = np.array([0.95, 0.8, 0.6, 0.5, 0.4, 0.2, 0.05], dtype = NP_REAL)
            nquantiles: NP_INT = NP_INT(quantiles.size)

            sw_flux_abs_rt_stats: list[XR_DATAARRAY] = [sw_flux_abs_rt.quantile(qq, dim = ["lay", "y", "x"]) for qq in quantiles]
            sw_flux_abs_ts_stats: list[XR_DATAARRAY] = [sw_flux_abs_ts.quantile(qq, dim = ["lay", "y", "x"]) for qq in quantiles]
            sw_flux_abs_diff_stats: list[XR_DATAARRAY] = [sw_flux_abs_diff.quantile(qq, dim = ["lay", "y", "x"]) for qq in quantiles]

            sw_heating_rt_stats: list[XR_DATAARRAY] = [sw_heating_rt.quantile(qq, dim = ["lay", "y", "x"]) for qq in quantiles]
            sw_heating_ts_stats: list[XR_DATAARRAY] = [sw_heating_ts.quantile(qq, dim = ["lay", "y", "x"]) for qq in quantiles]
            sw_heating_diff_stats: list[XR_DATAARRAY] = [sw_heating_diff.quantile(qq, dim = ["lay", "y", "x"]) for qq in quantiles]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting data...".format(current_time)
            print(msg, flush = True)

            rt_label: str = "Ray-Tracer"
            ts_label: str = "Two-Stream"
            diff_label: str = "Ray-Tracer - Two-Stream"
            rt_color: str = "blue"
            ts_color: str = "red"
            diff_color: str = "black"

            # Row 0: Absorbed Shortwave Flux - Two-Stream and Ray-Tracer
            row: NP_INT = NP_INT(0)
            median_index: NP_INT = NP_INT(nquantiles // 2)
            axs[row,jj].plot(sw_flux_abs_rt_stats[median_index]["time"], sw_flux_abs_rt_stats[median_index],
                color = rt_color, label = rt_label)
            axs[row,jj].plot(sw_flux_abs_ts_stats[median_index]["time"], sw_flux_abs_ts_stats[median_index],
                color = ts_color, label = ts_label)

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(sw_flux_abs_rt_stats[ll]["time"], 
                    sw_flux_abs_rt_stats[ll], sw_flux_abs_rt_stats[(nquantiles - 1) - ll],
                    color = rt_color, edgecolor = None, alpha = 0.15)
                axs[row,jj].fill_between(sw_flux_abs_ts_stats[ll]["time"], 
                    sw_flux_abs_ts_stats[ll], sw_flux_abs_ts_stats[(nquantiles - 1) - ll],
                    color = ts_color, edgecolor = None, alpha = 0.15)

            # Row 1: Absorbed Shortwave Flux - Two-Stream minus Ray-Tracer
            row: NP_INT = NP_INT(1)
            median_index: NP_INT = NP_INT(nquantiles // 2)
            axs[row,jj].plot(sw_flux_abs_diff_stats[median_index]["time"], sw_flux_abs_diff_stats[median_index],
                color = diff_color, label = diff_label)

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(sw_flux_abs_diff_stats[ll]["time"], 
                    sw_flux_abs_diff_stats[ll], sw_flux_abs_diff_stats[(nquantiles - 1) - ll],
                    color = diff_color, edgecolor = None, alpha = 0.15)

            # Row 2: Heating Rate - Two-Stream and Ray-Tracer
            row: NP_INT = NP_INT(2)
            median_index: NP_INT = NP_INT(nquantiles // 2)
            axs[row,jj].plot(sw_heating_rt_stats[median_index]["time"], sw_heating_rt_stats[median_index],
                color = rt_color, label = rt_label)
            axs[row,jj].plot(sw_heating_ts_stats[median_index]["time"], sw_heating_ts_stats[median_index],
                color = ts_color, label = ts_label)

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(sw_heating_rt_stats[ll]["time"], 
                    sw_heating_rt_stats[ll], sw_heating_rt_stats[(nquantiles - 1) - ll],
                    color = rt_color, edgecolor = None, alpha = 0.15)
                axs[row,jj].fill_between(sw_heating_ts_stats[ll]["time"], 
                    sw_heating_ts_stats[ll], sw_heating_ts_stats[(nquantiles - 1) - ll],
                    color = ts_color, edgecolor = None, alpha = 0.15)

            # Row 3: Heating Rate - Two-Stream minus Ray-Tracer
            row: NP_INT = NP_INT(3)
            median_index: NP_INT = NP_INT(nquantiles // 2)
            axs[row,jj].plot(sw_heating_diff_stats[median_index]["time"], sw_heating_diff_stats[median_index],
                color = diff_color, label = diff_label)

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(sw_heating_diff_stats[ll]["time"], 
                    sw_heating_diff_stats[ll], sw_heating_diff_stats[(nquantiles - 1) - ll],
                    color = diff_color, edgecolor = None, alpha = 0.15)

            # Common column-wise plot elements
            # x-ticks
            time: XR_DATARRAY = sw_flux_abs_diff["time"]
            xlim: list[NP_REAL] = np.array([time.min(), time.max()], dtype = NP_REAL)
            xticks: NP_ARRAY[NP_REAL] = np.array([xlim[0], (xlim[0] + xlim[1]) / 2., xlim[1]], dtype = NP_REAL)
            ll: int
            for ll in range(0, nrows):
                axs[ll,jj].set_xticks(xticks)
                axs[ll,jj].axvline(xticks[1], color = "gray", linestyle = "solid", linewidth = 0.5)


        # Labels
        fig.suptitle("RTE-RRTMGP-CPP Radiative Transfer Time Series")
        fig.supxlabel(r"Time $\left[ h \right]$")

        for jj in range(0, ndays):
            col_title: str = "Day {}".format(jj)
            axs[0,jj].set_title(col_title)
        axs[0,0].set_ylabel(r"Absorbed Shortwave Flux $\left[ W\,m^{-3} \right]$")
        axs[1,0].set_ylabel(r"Absorbed Shortwave Flux $\left[ W\,m^{-3} \right]$")
        axs[2,0].set_ylabel(r"Atmospheric Heating Rate $\left[ K\,d^{-1} \right]$")
        axs[3,0].set_ylabel(r"Atmospheric Heating Rate $\left[ K\,d^{-1} \right]$")

        axs[0,0].legend()
        axs[1,0].legend()

        #-------------------------------------------------------------------
        # Save the plot to file
        #-------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_rad_tran_timeseries.{}.png".format(coarse_factor_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()