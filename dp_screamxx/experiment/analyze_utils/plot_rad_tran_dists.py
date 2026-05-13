# Library imports
import argparse
import glob
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import xarray as xr

# Local imports
from consts import rho_sw, cp_sw, h_m, sec_per_day
from find_zmax_index import find_zmax_index
from find_pairs import find_pairs
from find_daytime_slices import find_daytime_slices
from calc_atm_heating import calc_atm_heating
from calc_abs_flux import calc_abs_flux
from calc_sfc_net import calc_sfc_net
from calc_tod_up import calc_tod_up
from calc_hwp import calc_hwp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rad-tran-indir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer input file directory.")
    parser.add_argument("--rad-tran-outdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer output file directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--lr", nargs = "?", default = "", type = str,
        help = "Resolution factor tag.")
    parser.add_argument("--zmax", nargs = "?", default = 16., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--case", nargs = "?", default = "", type = str,
        help = "Case to determine heating rate calculation parameters.")
    parser.add_argument("--detailed-calc", nargs = "?", default = False, type = bool,
        help = ("True: Compute cloud water mass using VMRs, etc. "
            "False: Compute cloud water mass using standard values."))
    args = parser.parse_args()

    rad_tran_indir  = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir = os.path.normpath(args.rad_tran_vizdir)
    working_dir = args.working_dir
    recalculate = args.recalculate
    lrs = [str(lr) for lr in args.lr.split(",")]
    zmax = args.zmax
    case = args.case
    detailed_calc = args.detailed_calc

    dirs = [rad_tran_vizdir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    #---------------------------------------------------------------------------
    # Verify necessary files are present.
    #---------------------------------------------------------------------------
    [rad_tran_infiles, rad_tran_outfiles] = find_pairs(rad_tran_indir, rad_tran_outdir, lrs)

    #---------------------------------------------------------------------------
    # Read files.
    #---------------------------------------------------------------------------
    for ii in range(len(rad_tran_infiles)):
        rad_tran_infile  = rad_tran_infiles[ii]
        rad_tran_outfile = rad_tran_outfiles[ii]

        #-----------------------------------------------------------------------
        # Get information for plot name
        #-----------------------------------------------------------------------
        lr_re = re.compile("lr_..")
        lr_str = re.search(lr_re, rad_tran_infile).group()

        #-----------------------------------------------------------------------
        # Determine daytime slices
        #-----------------------------------------------------------------------
        in_daytime_slices = find_daytime_slices(rad_tran_infile) # daytime_slices for rad_tran_input

        # ASSUME: That rad_tran_outfile time dimension has indexes of timesteps,
        # which may be off-set.
        out_daytime_slice_offset = int(xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = 0))
        out_daytime_slices = []
        for in_daytime_slice in in_daytime_slices:
            out_daytime_slices += [slice(in_daytime_slice.start - out_daytime_slice_offset, in_daytime_slice.stop - out_daytime_slice_offset)]

        #-----------------------------------------------------------------------
        # Obtain information common across each day
        #-----------------------------------------------------------------------
        x = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["x"].values / 1000 # [km]
        y = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["y"].values / 1000 # [km]
        lay = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["lay"].values / 1000 # [km]

        zmax_index = find_zmax_index(lay, zmax)
        lay = lay[:zmax_index]

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y

        #-----------------------------------------------------------------------
        # Plot multiple quantities
        #-----------------------------------------------------------------------
        if case == "GATEIII":
            field_keys = ["net_sfc_flux", "abs_flux", "sfc_heating", "atm_heating", "tod_up"]
            supylabels = [r"Net Surface Flux $\left[ W\,m^{-2} \right]$",
                r"Absorbed Flux $\left[ W\,m^{-3} \right]$",
                r"Surface Heating Rate $\left[ K\,d^{-1} \right]$", 
                r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$",
                r"Top-of-Domain Upwelling Flux $\left[ W\,m^{-2} \right]$"]
        else:
            field_keys = ["net_sfc_flux", "abs_flux", "atm_heating", "tod_up"]
            supylabels = [r"Net Surface Flux $\left[ W\,m^{-2} \right]$",
                r"Absorbed Flux $\left[ W\,m^{-3} \right]$",
                r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$",
                r"Top-of-Domain Upwelling Flux $\left[ W\,m^{-2} \right]$"]

        for jj in range(len(field_keys)):
            field_key = field_keys[jj]
            supylabel = supylabels[jj]

            #-----------------------------------------------------------------------
            # Set up for main figure
            #-----------------------------------------------------------------------
            legend_elements = [
                Line2D([0], [0], color = "#56B4E9", lw = 2.0, label = "Min - Max"),
                Patch(facecolor = "#56B4E940", edgecolor = "None", label = "5% - 95%"),
                Patch(facecolor = "#56B4E960", edgecolor = "None", label = "20% - 80%"),
                Patch(facecolor = "#56B4E980", edgecolor = "None", label = "40% - 60%"),
                Line2D([0], [0], color = "#000000", lw = 2.0, label = "Median"),
            ]

            #-----------------------------------------------------------------------
            # Set up plot for multiple days
            #-----------------------------------------------------------------------
            ndays = len(in_daytime_slices)

            fig, axs = plt.subplots(nrows = 3, ncols = ndays,
                sharex = "col", sharey = "row",
                constrained_layout = True,
                figsize = (14, 14))

            # First two rows share y-axis
            axs[1,0].sharey(axs[0,0])

            # Labels
            lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
            fig.suptitle("Horizontal Resolution - {}".format(lr_label))
            fig.supxlabel(r"Time [h]")
            fig.supylabel(supylabel)

            for kk in range(ndays):
                axs[0,kk].set_title(r"Day {}".format(kk))
            axs[0,0].set_ylabel("Two-Stream")
            axs[1,0].set_ylabel("Ray-Tracer")
            axs[2,0].set_ylabel("Ray-Tracer - Two-Stream")

            # Legend
            axs[0,0].legend(handles = legend_elements, loc = "upper left")

            for kk in range(ndays):
                in_daytime_slice = in_daytime_slices[kk]
                out_daytime_slice = out_daytime_slices[kk]

                #-------------------------------------------------------------------
                # Obtain day-specific information for plots
                #-------------------------------------------------------------------
                time = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_daytime_slice).values

                #-------------------------------------------------------------------
                # Obtain two-stream (ts) and ray-tracer (rt) data
                #-------------------------------------------------------------------
                if field_key == "sfc_heating":
                    [ts_field, rt_field] = calc_sfc_net(rad_tran_outfile, 
                        out_daytime_slice) # [W m^{-2}], [time, y, x]
                    # ASSUME - THIS IS GATEIII
                    ts_field = (ts_field / (rho_sw * cp_sw * h_m)) * sec_per_day # [K d^{-1}], [time, y, x]
                    rt_field = (rt_field / (rho_sw * cp_sw * h_m)) * sec_per_day # [K d^{-1}], [time, y, x]
                elif field_key == "atm_heating":
                    [ts_field, rt_field] = calc_atm_heating(rad_tran_infile,
                        rad_tran_outfile, in_daytime_slice, out_daytime_slice, 
                        zmax_index = zmax_index, detailed_calc = detailed_calc) # [K d^{-1}], [time, lay, x]
                elif field_key == "net_sfc_flux":
                    [ts_field, rt_field] = calc_sfc_net(rad_tran_outfile, 
                        out_daytime_slice) # [W m^{-2}], [time, y, x]
                elif field_key == "abs_flux":
                    [ts_field, rt_field] = calc_abs_flux(rad_tran_outfile, 
                        out_daytime_slice, zmax_index = zmax_index) # [W m^{-3}], [time, lay, x]
                elif field_key == "tod_up":
                    [ts_field, rt_field] = calc_tod_up(rad_tran_outfile, 
                        out_daytime_slice) # [W m^{-2}], [time, y, x]

                diff_field = rt_field - ts_field

                #-------------------------------------------------------------------
                # Prepare data for plotting
                #-------------------------------------------------------------------
                quantiles = [0.00, 0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95, 1.00]
                ts_quantiles = []
                rt_quantiles = []
                diff_quantiles = []
                for qq in quantiles:
                    # Assume time is zeroth dimension
                    ts_quantiles += [ts_field.quantile(qq, dim = ts_field.dims[1:]).values] # [time]
                    rt_quantiles += [rt_field.quantile(qq, dim = rt_field.dims[1:]).values] # [time]
                    diff_quantiles += [diff_field.quantile(qq, dim = diff_field.dims[1:]).values] # [time]

                #-------------------------------------------------------------------
                # Plot the data
                #-------------------------------------------------------------------
                # Row 1: Two-Stream
                axs[0,kk].fill_between(time, 
                    ts_quantiles[0], ts_quantiles[-1],
                    color = "None", edgecolor = "#56B4E9", linewidth = 2.0)
                for ll in range(1, int(len(quantiles) / 2)):
                    axs[0,kk].fill_between(time, 
                        ts_quantiles[ll], ts_quantiles[-(ll + 1)],
                        color = "#56B4E940", edgecolor = "None")
                axs[0,kk].plot(time, ts_quantiles[int(len(quantiles) / 2) + 1],
                    color = "#000000", linewidth = 2.0)

                # Row 2: Ray-Tracer
                axs[1,kk].fill_between(time, 
                    rt_quantiles[0], rt_quantiles[-1],
                    color = "None", edgecolor = "#56B4E9", linewidth = 2.0)
                for ll in range(1, int(len(quantiles) / 2)):
                    axs[1,kk].fill_between(time, 
                        rt_quantiles[ll], rt_quantiles[-(ll + 1)],
                        color = "#56B4E940", edgecolor = "None")
                axs[1,kk].plot(time, rt_quantiles[int(len(quantiles) / 2) + 1],
                    color = "#000000", linewidth = 2.0)

                # Row 3: Ray-Tracer - Two-Stream
                axs[2,kk].fill_between(time, 
                    diff_quantiles[0], diff_quantiles[-1],
                    color = "None", edgecolor = "#56B4E9", linewidth = 2.0)
                for ll in range(1, int(len(quantiles) / 2)):
                    axs[2,kk].fill_between(time, 
                        diff_quantiles[ll], diff_quantiles[-(ll + 1)],
                        color = "#56B4E940", edgecolor = "None")
                axs[2,kk].plot(time, diff_quantiles[int(len(quantiles) / 2) + 1],
                    color = "#000000", linewidth = 2.0)

                #---------------------------------------------------------------
                # Set common elements for plots in column
                #---------------------------------------------------------------
                xticks = [time.min(), (time.min() + time.max()) / 2., time.max()]
                for ll in range(3):
                    axs[ll,kk].axvline(xticks[1], color = "gray", linewidth = 1.0)
                    axs[ll,kk].set_xlim(xticks[0], xticks[2])
                    axs[ll,kk].set_xticks(xticks)
                axs[2,kk].set_xticklabels(["{0:g}".format(v) for v in xticks])

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "{}_{}.png".format(lr_str, field_key)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()