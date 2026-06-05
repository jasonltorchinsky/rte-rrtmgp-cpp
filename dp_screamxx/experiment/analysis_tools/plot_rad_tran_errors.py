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
from consts import rho_sw, cp_sw, h_m, sec_per_day, plot_colors
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
    [rad_tran_infiles, rad_tran_outfiles] = find_pairs(rad_tran_indir, 
        rad_tran_outdir, lrs)

    #---------------------------------------------------------------------------
    # Loop through error types
    #---------------------------------------------------------------------------
    error_keys = ["rmse", "mae", "mbe"]
    supylabels = ["Root-Mean-Square Error",
        "Mean Absolute Error",
        "Mean Bias Error: Ray-Tracer - Two-Stream"]
    
    for ii in range(len(error_keys)):
        error_key = error_keys[ii]
        supylabel = supylabels[ii]

        #---------------------------------------------------------------------------
        # Set up plot for multiple days
        #---------------------------------------------------------------------------
        # Obtain number of quantities
        if case == "GATEIII":
            field_keys = ["tod_up", "abs_flux", "net_sfc_flux", "atm_heating", "sfc_heating"]
            ylabels = [r"Top-of-Domain Upwelling Flux $\left[ W\,m^{-2} \right]$",
                r"Absorbed Flux $\left[ W\,m^{-3} \right]$", 
                r"Net Surface Flux $\left[ W\,m^{-2} \right]$",
                r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$",
                r"Surface Heating Rate $\left[ K\,d^{-1} \right]$"]
        else:
            field_keys = ["tod_up", "abs_flux", "net_sfc_flux", "atm_heating"]
            ylabels = [r"Top-of-Domain Upwelling Flux $\left[ W\,m^{-2} \right]$",
                r"Absorbed Flux $\left[ W\,m^{-3} \right]$", 
                r"Net Surface Flux $\left[ W\,m^{-2} \right]$",
                r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$"]
        nfields = len(field_keys)

        # Obtain number of days
        rad_tran_infile  = rad_tran_infiles[0]
        rad_tran_outfile = rad_tran_outfiles[0]
        in_daytime_slices = find_daytime_slices(rad_tran_infile)
        ndays = len(in_daytime_slices)

        fig, axs = plt.subplots(nrows = nfields, ncols = ndays,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = (14, 14))

        # Labels
        fig.supxlabel(r"Time [h]")
        fig.supylabel(supylabel)

        for jj in range(ndays):
            axs[0,jj].set_title(r"Day {}".format(jj))
        for jj in range(nfields):
            axs[jj,0].set_ylabel(ylabels[jj])
            #if error_key in ["rmse", "mae"]:
            #    axs[jj,0].set_yscale("log")
            #elif error_key == "mbe":
            #    axs[jj,0].set_yscale("symlog")

        #---------------------------------------------------------------------------
        # Read files.
        #---------------------------------------------------------------------------
        for jj in range(len(rad_tran_infiles)):
            rad_tran_infile  = rad_tran_infiles[jj]
            rad_tran_outfile = rad_tran_outfiles[jj]

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
            lay = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["lay"].values / 1000 # [km]

            zmax_index = find_zmax_index(lay, zmax)
            lay = lay[:zmax_index]

            dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y 

            #-----------------------------------------------------------------------
            # Get resolution-specific plot information
            #-----------------------------------------------------------------------
            lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
            lr_color = plot_colors[jj]

            for kk in range(ndays):
                in_daytime_slice = in_daytime_slices[kk]
                out_daytime_slice = out_daytime_slices[kk]

                #-------------------------------------------------------------------
                # Obtain day-specific information for plots
                #-------------------------------------------------------------------
                time = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_daytime_slice).values

                #---------------------------------------------------------------
                # Set common elements for plots in column
                #---------------------------------------------------------------
                xticks = [time.min(), (time.min() + time.max()) / 2., time.max()]
                for ll in range(nfields):
                    axs[ll,kk].axvline(xticks[1], color = "gray", linewidth = 1.0)
                    axs[ll,kk].set_xlim(xticks[0], xticks[2])
                    axs[ll,kk].set_xticks(xticks)
                axs[2,kk].set_xticklabels(["{0:g}".format(v) for v in xticks])

                for ll in range(len(field_keys)):
                    field_key = field_keys[ll]

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
                            zmax_index = zmax_index, detailed_calc = False) # [K d^{-1}], [time, lay, x]
                    elif field_key == "net_sfc_flux":
                        [ts_field, rt_field] = calc_sfc_net(rad_tran_outfile, 
                            out_daytime_slice) # [W m^{-2}], [time, y, x]
                    elif field_key == "abs_flux":
                        [ts_field, rt_field] = calc_abs_flux(rad_tran_outfile, 
                            out_daytime_slice, zmax_index = zmax_index) # [W m^{-3}], [time, lay, x]
                    elif field_key == "tod_up":
                        [ts_field, rt_field] = calc_tod_up(rad_tran_outfile, 
                            out_daytime_slice) # [W m^{-2}], [time, y, x]

                    if error_key == "rmse":
                        error_field = np.sqrt(np.pow(rt_field - ts_field, 2).mean(dim = rt_field.dims[1:])) # [time]
                    elif error_key == "mae":
                        error_field = np.abs(rt_field - ts_field).mean(dim = rt_field.dims[1:])
                    elif error_key == "mbe":
                        error_field = rt_field.mean(dim = rt_field.dims[1:]) - ts_field.mean(dim = ts_field.dims[1:])

                    #-------------------------------------------------------------------
                    # Prepare data for plotting
                    #-------------------------------------------------------------------
                    error_field = error_field.values

                    #-------------------------------------------------------------------
                    # Plot the data
                    #-------------------------------------------------------------------
                    if field_key == "tod_up":
                        row = 0
                    elif field_key == "abs_flux":
                        row = 1
                    elif field_key == "net_sfc_flux":
                        row = 2
                    elif field_key == "atm_heating":
                        row = 3
                    elif field_key == "sfc_heating":
                        row = 4

                    axs[row,kk].plot(time, error_field,
                        color = lr_color, linewidth = 2.0, label = lr_label)

        #---------------------------------------------------------------------------
        # Final figure information
        #---------------------------------------------------------------------------
        # Legend
        axs[0,0].legend(loc = "upper left")

        #---------------------------------------------------------------------------
        # Save the plot to file
        #---------------------------------------------------------------------------
        plt_filename = "{}.png".format(error_key)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()