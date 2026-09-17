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
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_FIGURE, MPL_AXES, MPL_PCOLORMESH
from consts.numeric import NP_INF
from consts.visual import plot_colors
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_grid, find_szas, find_times, \
    calc_cloud_wc, calc_sw_reflectance, calc_sw_flux_sfc_dn, calc_sw_heating, \
    calc_z_max_info, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-rad-tran-timeseries"
prog_desc: str = "Visualize timeseries of radiative quantities for RTE-RRTMGP-CPP."

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
    parser.add_argument("--z-max", nargs = "?", default = 16., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
    parser.add_argument("--quantity", nargs = "?", default = "reflectance", type = str,
        help = "Quantity to calculate and output.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None
    rad_tran_quantity: str = args.quantity

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

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, coarse_factor_str)
        print(msg, flush = True)

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)
        grid: dict = find_grid(rad_tran_infile)

        #-----------------------------------------------------------------------
        # Obtain daytime indices, times, SZAs
        #-----------------------------------------------------------------------
        msg: str = "Obtaining daytime information..."
        print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(
            rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
        daytime_times: NP_ARRAY[NP_REAL] = find_times(
            rad_tran_infile, 
            daytime_indices) # Time since simulation start; [h]; [ndays, 3]
        daytime_szas: NP_ARRAY[NP_REAL] = find_szas(
            rad_tran_infile, 
            daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(daytime_indices.shape[0])
        z_max_info: dict = calc_z_max_info(
            rad_tran_infile, 
            z_max = z_max)

        rad_tran_quantity_max: NP_REAL = -NP_INF
        rad_tran_quantity_min: NP_REAL = NP_INF

        #-----------------------------------------------------------------------
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        nrows: NP_INT = NP_INT(3)
        ncols: NP_INT = NP_INT(ndays)
        fig_height: NP_REAL = NP_REAL(8.)
        fig_width: NP_REAL = NP_REAL(6.5)
        fig_size: list[NP_REAL] = [fig_width, fig_height]
        fig: MPL_FIGURE
        axs: MPL_AXES
        fig, axs = plt.subplots(
            nrows = nrows, ncols = ncols,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = fig_size)

        if ncols == 1:
            axs = axs[...,None]
        elif nrows == 1:
            axs = axs[None,...]

        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            rad_tran_quantity_rt: XR_DATAARRAY
            rad_tran_quantity_ts: XR_DATAARRAY
            rad_tran_quantity_diff: XR_DATAARRAY

            if rad_tran_quantity == "reflectance":
                #-------------------------------------------------------------------
                # Calculate reflectance
                #-------------------------------------------------------------------
                msg: str = "Calculating reflectance for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                rad_tran_quantity_rt = calc_sw_reflectance(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = daytime_indices[jj,...],
                    solver = "rt") # Shortwave reflectance, ray-tracer; [N/A]; [time, y, x]

                rad_tran_quantity_ts = calc_sw_reflectance(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = daytime_indices[jj,...],
                    solver = "ts") # Shortwave reflectance, two-stream; [N/A]; [time, y, x]
            
            elif rad_tran_quantity == "heating":
                #-------------------------------------------------------------------
                # Calculate heating rates
                #-------------------------------------------------------------------
                msg: str = "Calculating heating rates for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                rad_tran_quantity_rt: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile, 
                    rad_tran_outfile,
                    time_indices = daytime_indices[jj,...],
                    z_max_info = z_max_info,
                    solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [time, lay, y, x]

                rad_tran_quantity_ts: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile, 
                    rad_tran_outfile,
                    time_indices = daytime_indices[jj,...],
                    z_max_info = z_max_info,
                    solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [time, lay, y, x]

            elif rad_tran_quantity == "flux_sfc_dn":
                #-------------------------------------------------------------------
                # Calculate downwelling surface flux
                #-------------------------------------------------------------------
                msg: str = "Calculating downwelling surface fluxes for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                rad_tran_quantity_rt = calc_sw_flux_sfc_dn(
                    rad_tran_outfile,
                    time_indices = daytime_indices[jj,...],
                    solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

                rad_tran_quantity_ts = calc_sw_flux_sfc_dn(
                    rad_tran_outfile,
                    time_indices = daytime_indices[jj,...],
                    solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

            rad_tran_quantity_diff = rad_tran_quantity_rt - rad_tran_quantity_ts

            rad_tran_quantity_max = max(
                rad_tran_quantity_max, 
                NP_REAL(rad_tran_quantity_rt.max()), 
                NP_REAL(rad_tran_quantity_ts.max()))
            rad_tran_quantity_min = min(
                rad_tran_quantity_min, 
                NP_REAL(rad_tran_quantity_rt.min()), 
                NP_REAL(rad_tran_quantity_ts.min()))

            #-------------------------------------------------------------------
            # Obtain statistics for radiative quantities
            #-------------------------------------------------------------------
            quantiles: NP_ARRAY[NP_REAL] = np.array([1.0, 0.95, 0.8, 0.6, 0.5, 0.4, 0.2, 0.05, 0.0], dtype = NP_REAL)
            nquantiles: NP_INT = NP_INT(quantiles.size)

            dim_list: list[str]
            if rad_tran_quantity in ["reflectance", "flux_sfc_dn"]:
                dim_list = ["y", "x"]
            elif rad_tran_quantity == "heating":
                dim_list = ["lay", "y", "x"]

            rad_tran_quantity_rt_stats: list[XR_DATAARRAY] = [rad_tran_quantity_rt.quantile(qq, dim = dim_list) for qq in quantiles]
            rad_tran_quantity_ts_stats: list[XR_DATAARRAY] = [rad_tran_quantity_ts.quantile(qq, dim = dim_list) for qq in quantiles]
            rad_tran_quantity_diff_stats: list[XR_DATAARRAY] = [rad_tran_quantity_diff.quantile(qq, dim = dim_list) for qq in quantiles]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting data..."
            print_msg(msg)

            rt_label: str = "Ray-Tracer"
            ts_label: str = "Two-Stream"
            diff_label: str = "Ray-Tracer - Two-Stream"
            ts_color: str = plot_colors[0]
            rt_color: str = plot_colors[1]
            diff_color: str = plot_colors[2]

            median_index: NP_INT = NP_INT(nquantiles // 2)

            # Row 0: Heating Rate - Two-Stream
            row: NP_INT = NP_INT(0)
            axs[row,jj].plot(
                rad_tran_quantity_ts_stats[median_index]["time"], 
                rad_tran_quantity_ts_stats[median_index],
                color = ts_color, 
                label = ts_label)

            ll: int
            for ll in range(1, median_index):
                axs[row,jj].fill_between(
                    rad_tran_quantity_ts_stats[ll]["time"], 
                    rad_tran_quantity_ts_stats[ll], 
                    rad_tran_quantity_ts_stats[(nquantiles - 1) - ll],
                    color = ts_color, 
                    edgecolor = None, 
                    alpha = 0.15)

            axs[row,jj].plot(
                rad_tran_quantity_ts_stats[0]["time"], 
                rad_tran_quantity_ts_stats[0],
                color = ts_color,
                linestyle = "dashed", 
                label = ts_label)

            axs[row,jj].plot(
                rad_tran_quantity_ts_stats[-1]["time"], 
                rad_tran_quantity_ts_stats[-1],
                color = ts_color,
                linestyle = "dashed", 
                label = ts_label)

            # Row 1: Heating Rate - Ray-Tracer
            row: NP_INT = NP_INT(1)
            axs[row,jj].plot(
                rad_tran_quantity_rt_stats[median_index]["time"], 
                rad_tran_quantity_rt_stats[median_index],
                color = rt_color, 
                label = rt_label)

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(
                    rad_tran_quantity_rt_stats[ll]["time"], 
                    rad_tran_quantity_rt_stats[ll], 
                    rad_tran_quantity_rt_stats[(nquantiles - 1) - ll],
                    color = rt_color, 
                    edgecolor = None, 
                    alpha = 0.15)

            axs[row,jj].plot(
                rad_tran_quantity_rt_stats[0]["time"], 
                rad_tran_quantity_rt_stats[0],
                color = rt_color,
                linestyle = "dashed", 
                label = rt_label)

            axs[row,jj].plot(
                rad_tran_quantity_rt_stats[-1]["time"], 
                rad_tran_quantity_rt_stats[-1],
                color = rt_color,
                linestyle = "dashed", 
                label = rt_label)

            # Row 2: Heating Rate - Ray-Tracer minus Two-Stream
            row: NP_INT = NP_INT(2)
            axs[row,jj].plot(
                rad_tran_quantity_diff_stats[median_index]["time"],
                rad_tran_quantity_diff_stats[median_index],
                color = diff_color,
                label = diff_label)

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(
                    rad_tran_quantity_diff_stats[ll]["time"], 
                    rad_tran_quantity_diff_stats[ll], 
                    rad_tran_quantity_diff_stats[(nquantiles - 1) - ll],
                    color = diff_color, 
                    edgecolor = None, 
                    alpha = 0.15)

            axs[row,jj].plot(
                rad_tran_quantity_diff_stats[0]["time"], 
                rad_tran_quantity_diff_stats[0],
                color = diff_color,
                linestyle = "dashed", 
                label = diff_label)

            axs[row,jj].plot(
                rad_tran_quantity_diff_stats[-1]["time"], 
                rad_tran_quantity_diff_stats[-1],
                color = diff_color,
                linestyle = "dashed", 
                label = diff_label)

            #-------------------------------------------------------------------
            # Add common column-wise plot elements
            #-------------------------------------------------------------------
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
                axs[ll,jj].axvline(time_xticks[1], 
                    color = "gray", 
                    linestyle = "solid", 
                    linewidth = 0.5)

                ax_2: MPL_AXES = axs[ll,jj].secondary_xaxis("top")
                if ll == 0:
                    ax_2.set_xticks(time_xticks, labels = sza_xtick_labels)
                else:
                    ax_2.set_xticks(time_xticks, labels = [None, None, None])

        #-----------------------------------------------------------------------
        # Labels
        #-----------------------------------------------------------------------
        dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.1f} $km$".format(dx * 1.e-3)
        title: str
        if rad_tran_quantity == "reflectance":
            title = r"Reflectance"
        elif rad_tran_quantity == "heating":
            title = r"Heating Rate $\left[ K\,d^{-1} \right]$"
        elif rad_tran_quantity == "flux_sfc_dn":
            title = r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$"

        title += " - {}".format(dx_str)
        fig.suptitle(title)
        fig.supxlabel(r"Time $\left[ h \right]$")

        for jj in range(0, ndays):
            col_title: str = "Day {}".format(jj)
            axs[0,jj].set_title(col_title)
        axs[0,0].set_ylabel(r"Two-Stream")
        axs[1,0].set_ylabel(r"Ray-Tracer")
        axs[2,0].set_ylabel(r"Ray-Tracer - Two-Stream")

        #-----------------------------------------------------------------------
        # Set y-scale
        #-----------------------------------------------------------------------
        linthresh: NP_REAL # Linear threshold for symlog scale
        if rad_tran_quantity == "heating":
            linthresh = NP_REAL(1.0e0)
        elif rad_tran_quantity == "flux_sfc_dn":
            linthresh = NP_REAL(1.0e2)

        for ax in axs[0,:]:
            ax.set_ylim([rad_tran_quantity_min, rad_tran_quantity_max])
        for ax in axs[1,:]:
            ax.set_ylim([rad_tran_quantity_min, rad_tran_quantity_max])            
        for ax in axs[2,:]:
            ax.axhline([0], 
                color = "gray", 
                linestyle = "solid", 
                linewidth = 0.5)

        if rad_tran_quantity in ["heating", "flux_sfc_dn"]:
            for ax in axs[0,:]:
                ax.set_yscale("log")
            for ax in axs[1,:]:
                ax.set_yscale("log")          
            for ax in axs[2,:]:
                ax.set_yscale("symlog", linthresh = linthresh)
                ax.axhline([linthresh], 
                    color = "black", 
                    linestyle = "solid", 
                    linewidth = 1.0)
                ax.axhline([-linthresh], 
                    color = "black", 
                    linestyle = "dashed", 
                    linewidth = 1.0)

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_rad_tran_{}_timeseries.{}.png".format(rad_tran_quantity, coarse_factor_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()