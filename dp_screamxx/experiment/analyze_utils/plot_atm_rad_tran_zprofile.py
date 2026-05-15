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
from consts import plot_colors
from find_zmax_index import find_zmax_index
from find_pairs import find_pairs
from find_daytime_slices import find_daytime_slices
from calc_atm_heating import calc_atm_heating
from calc_abs_flux import calc_abs_flux
from calc_mass_air import calc_mass_air
from calc_vmr import calc_vmr
from calc_wc import calc_wc

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
    # Set fields to plot
    #---------------------------------------------------------------------------
    field_keys = ["vmr", "mass_air", "wc", "abs_flux", "atm_heating"]
    ylabels = [r"Volume Mixing Ratio $\left[ mol\,mol^{-1} \right]$",
        r"Density of Dry Air $\left[ kg\,m^{-3} \right]$",
        r"Water Content $\left[ g\,m^{-3} \right]$",
        r"Absorbed Flux $\left[ W\,m^{-3} \right]$",
        r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$"]
    vmr_labels = {
        "vmr_co2"     : r"Carbon Dioxide $\left( CO_{2} \right)$",
        "vmr_ch4"     : r"Methane $\left( CH_{4} \right)$",
        "vmr_n2o"     : r"Nitrous Oxide $\left( N_{2}O \right)$",
        "vmr_o3"      : r"Ozone $\left( O_{3} \right)$",
        "vmr_h2o"     : r"Water Vapor $\left( H_{2}O \right)$",
        "vmr_n2"      : r"Nitrogen $\left( N_{2} \right)$",
        "vmr_o2"      : r"Oxygen $\left( O_{2} \right)$",
        "vmr_co"      : r"Carbon Monoxide $\left( CO \right)$",
        "vmr_ccl4"    : r"Carbon Tertachloride $\left( CCl_{4} \right)$",
        "vmr_cfc11"   : r"Trichlorofluoromethane $\left( CFC\text{-}11 \right)$",
        "vmr_cfc12"   : r"Dichlorodifluoromethane $\left( CFC\text{-}12 \right)$",
        "vmr_cfc22"   : r"Chlorodifluoromethane $\left( CFC\text{-}22 \right)$",
        "vmr_hfc143a" : r"1,1,1-Trifluoroethane $\left( HFC\text{-}143a \right)$",
        "vmr_hfc125"  : r"Pentafluoroethane $\left( HFC\text{-}125 \right)$",
        "vmr_hfc23"   : r"Trifluoromethane $\left( HFC\text{-}23 \right)$",
        "vmr_hfc32"   : r"Difluoromethane $\left( HFC\text{-}32 \right)$",
        "vmr_hfc134a" : r"1,1,1,2-Tetrafluoroethane $\left( HFC\text{-}134a \right)$",
        "vmr_cf4"     : r"Carbon Tetrafluoride $\left( CF_{4} \right)$",
        "vmr_no2"     : r"Nitrogen Dioxide $\left( NO_{2} \right)$",
    }
    
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

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y, [km]
        dz = lay[1] - lay[0] # [km]
        vol = dx**2 * dz * (1.e9) # [m^{3}]

        #-----------------------------------------------------------------------
        # Set up plot for multiple days and quantities
        #-----------------------------------------------------------------------
        nfields = len(field_keys)
        ndays = len(in_daytime_slices)

        fig, axs = plt.subplots(nrows = nfields, ncols = ndays,
            sharex = True, sharey = "row",
            constrained_layout = True,
            figsize = (14, 14))

        # Labels
        lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
        fig.suptitle("Horizontal Resolution - {}".format(lr_label))
        fig.supxlabel(r"$z$ $\left[ km \right]$")

        for jj in range(ndays):
            axs[0,jj].set_title(r"Day {}".format(jj))
        for jj in range(nfields):
            axs[jj,0].set_ylabel(ylabels[jj])

        for jj in range(len(field_keys)):
            field_key = field_keys[jj]

            for kk in range(ndays):
                in_daytime_slice = in_daytime_slices[kk]
                out_daytime_slice = out_daytime_slices[kk]

                #-------------------------------------------------------------------
                # Obtain two-stream (ts) and ray-tracer (rt) data
                #-------------------------------------------------------------------
                if field_key == "atm_heating":
                    [ts_field, rt_field] = calc_atm_heating(rad_tran_infile,
                        rad_tran_outfile, in_daytime_slice, out_daytime_slice, 
                        zmax_index = zmax_index, detailed_calc = False) # [K d^{-1}], [time, lay, y, x]
                elif field_key == "abs_flux":
                    [ts_field, rt_field] = calc_abs_flux(rad_tran_outfile, 
                        out_daytime_slice, zmax_index = zmax_index) # [W m^{-3}], [time, lay, y, x]
                elif field_key == "wc":
                    field = calc_wc(rad_tran_infile, in_daytime_slice, zmax_index = zmax_index) # [g m^{-3}], [time, lay, y, x]
                elif field_key == "mass_air":
                    field = calc_mass_air(rad_tran_infile, in_daytime_slice, zmax_index = zmax_index) # [kg], [time, lay, y, x]
                    field = field / vol # [kg m^{-3}]
                elif field_key == "vmr":
                    field = calc_vmr(rad_tran_infile, in_daytime_slice, zmax_index = zmax_index) # [mol mol^{-1}], [time, lay, y, x]

                #-------------------------------------------------------------------
                # Prepare data for plotting
                #-------------------------------------------------------------------
                if field_key in ["atm_heating", "abs_flux"]:
                    ts_field = ts_field.mean(dim = ["time", "y", "x"]).values # [lay]
                    rt_field = rt_field.mean(dim = ["time", "y", "x"]).values # [lay]
                elif field_key in ["wc", "mass_air"]:
                    field = field.mean(dim = ["time", "y", "x"]).values # [lay]
                elif field_key in ["vmr"]:
                    field = field.mean(dim = ["time", "y", "x"]) # [lay]

                #-------------------------------------------------------------------
                # Plot the data
                #-------------------------------------------------------------------
                if field_key in ["atm_heating", "abs_flux"]:
                    axs[jj,kk].plot(lay, ts_field, color = plot_colors[0], linewidth = 2,
                        linestyle = "dashed", label = r"Two-Stream")
                    axs[jj,kk].plot(lay, rt_field, color = plot_colors[1], linewidth = 2,
                        linestyle = "dotted", label = r"Ray-Tracer")
                elif field_key in ["wc", "mass_air"]:
                    axs[jj,kk].plot(lay, field, color = "#000000", linewidth = 2,
                        linestyle = "solid")
                elif field_key in ["vmr"]:
                    vmr_keys = list(field.keys())
                    ncolors = len(plot_colors)
                    for ll in range(len(vmr_keys)):
                        vmr_key = vmr_keys[ll]
                        axs[jj,kk].plot(lay, field[vmr_key], color = plot_colors[ll%ncolors],
                        linewidth = 2, linestyle = "solid", label = vmr_labels[vmr_key])
                
                #-------------------------------------------------------------------
                # Set elements for plots
                #-------------------------------------------------------------------
                # Legends only on Day 1
                if kk == 0:
                    # ASSUME "atm_heating" is first radiative quantity
                    if field_key in ["atm_heating"]:
                        axs[jj,kk].legend(loc = "lower left")
                    if field_key in ["vmr"]:
                        axs[jj,kk].legend(loc = "upper right")
                
                # vmr gets logscale y-axis
                if field_key in ["vmr"]:
                    axs[jj,kk].set_yscale("log")

        #-------------------------------------------------------------------
        # Save the plot to file
        #-------------------------------------------------------------------
        plt_filename = "{}_rad_tran_zprofile.png".format(lr_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()