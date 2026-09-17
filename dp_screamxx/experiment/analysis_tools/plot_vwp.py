# Library imports
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import xarray as xr

# Local imports
from consts import flux_cmap, heating_cmap, rho_sw, cp_sw, h_m, sec_per_day
from find_pairs import find_pairs
from find_mnn_indices import find_mnn_indices
from calc_sfc_net import calc_sfc_net
from calc_vwp import calc_vwp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rad-tran-indir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer input file directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--lr", nargs = "?", default = "", type = str,
        help = "Resolution factor tag.")
    parser.add_argument("--detailed-calc", nargs = "?", default = False, type = bool,
        help = ("True: Compute cloud water mass using VMRs, etc. "
            "False: Compute cloud water mass using standard values."))
    args = parser.parse_args()

    rad_tran_indir  = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir = os.path.normpath(args.rad_tran_vizdir)
    working_dir = args.working_dir
    recalculate = args.recalculate
    lrs = [int(lr) for lr in args.lr.split(",")]
    detailed_calc = args.detailed_calc

    dir_names = [rad_tran_vizdir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Read files.
    #---------------------------------------------------------------------------
    all_rad_tran_infiles = sorted(glob.glob(os.path.join(rad_tran_indir, "*.in.nc")), reverse = True)
    rad_tran_infiles = []
    for ii in range(len(all_rad_tran_infiles)):
        infile_name = os.path.basename(all_rad_tran_infiles[ii])
        ext_re = re.compile(".in.nc")
        infile_base = re.sub(ext_re, "", infile_name)

        for lr in lrs:
            lr_str = "lr_{:02}".format(lr)
            if lr_str in infile_base:
                rad_tran_infiles += [all_rad_tran_infiles[ii]]
                break

    for ii in range(len(rad_tran_infiles)):
        rad_tran_infile  = rad_tran_infiles[ii]

        #-----------------------------------------------------------------------
        # Get information for plot name
        #-----------------------------------------------------------------------
        lr_re = re.compile("lr_..")
        lr_str = re.search(lr_re, rad_tran_infile).group()

        #-----------------------------------------------------------------------
        # Determine morning, noon, night indices
        #-----------------------------------------------------------------------
        in_mnn_indices = find_mnn_indices(rad_tran_infile) # mnn_indices for rad_tran_input

        #-----------------------------------------------------------------------
        # Obtain information common across each day
        #-----------------------------------------------------------------------
        x = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["x"].values / 1000 # [km]
        y = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["y"].values / 1000 # [km]

        breakpoint()

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y

        #-----------------------------------------------------------------------
        # Plot multiple quantities
        #-----------------------------------------------------------------------
        ndays = in_mnn_indices.shape[0]
        for kk in range(ndays):
            in_mnn_index = in_mnn_indices[kk]

            #-------------------------------------------------------------------
            # Obtain day-specific information for plot labels
            #-------------------------------------------------------------------
            mnn_times = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_mnn_index).values

            #-------------------------------------------------------------------
            # Obtain vertical cloud water path
            #-------------------------------------------------------------------
            vwp = calc_vwp(rad_tran_infile, in_mnn_index, detailed_calc = detailed_calc)

            #-------------------------------------------------------------------
            # Prepare data for plotting
            #-------------------------------------------------------------------
            vwp = vwp.to_numpy() # [g m^{-2}], [time, x, y]

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max = np.max(vwp, axis = (1, 2))
            vwp_min = np.min(vwp, axis = (1, 2))

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            fig, axs = plt.subplots(nrows = 1, ncols = 3,
                sharex = True, sharey = True,
                constrained_layout = True,
                figsize = (12, 4))

            # Row 1: Vertical Water Path
            vwp_pcm = [[] for ll in range(3)]
            for ll in range(3):
                vwp_pcm[ll] = axs[ll].pcolormesh(x, y, vwp[ll,...],
                    vmin = vwp_min[ll], vmax = vwp_max[ll],
                    cmap = "Blues")

            # Colorbars
            for ll in range(3):
                vwp_cbar = fig.colorbar(vwp_pcm[ll], ax = axs[ll])

            # Labels
            lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
            fig.suptitle("Horizontal Resolution - {}".format(lr_label))
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"y $\left[ km \right]$")

            for ll in range(3):
                axs[ll].set_title(r"{:.2f} Hours".format(mnn_times[ll]))

            vwp_cbar.ax.set_ylabel(r"Vertical Cloud Water Path $\left[ g\,m^{-2} \right]$")

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "{}_vwp_day_{}.png".format(lr_str, kk)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()