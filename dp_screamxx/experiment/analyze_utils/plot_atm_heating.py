# Library imports
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import xarray as xr

# Local imports
from find_yslice_index import find_yslice_index
from find_zmax_index import find_zmax_index
from find_pairs import find_pairs
from find_mnn_indices import find_mnn_indices
from calc_flux_abs import calc_flux_abs
from calc_atm_heating import calc_atm_heating
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
    parser.add_argument("--yslice", nargs = "?", default = 132.5, type = float,
        help = "ycoordinate for XZ-slice [km].")
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
    yslice = args.yslice
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
        # Determine morning, noon, night indices
        #-----------------------------------------------------------------------
        in_mnn_indices = find_mnn_indices(rad_tran_infile) # mnn_indices for rad_tran_input

        # ASSUME: That rad_tran_outfile time dimension has indexes of timesteps,
        # which may be off-set.
        out_mnn_index_offset = int(xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = 0))
        out_mnn_indices = in_mnn_indices - out_mnn_index_offset

        #-----------------------------------------------------------------------
        # Obtain information common across each day
        #-----------------------------------------------------------------------
        x = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["x"].values / 1000 # [km]
        y = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["y"].values / 1000 # [km]
        lay = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["lay"].values / 1000 # [km]

        yslice_index = find_yslice_index(y, yslice)
        zmax_index = find_zmax_index(lay, zmax)
        lay = lay[:zmax_index]

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y

        ndays = in_mnn_indices.shape[0]
        for ii in range(ndays):
            in_mnn_index = in_mnn_indices[ii]
            out_mnn_index = out_mnn_indices[ii]

            #-------------------------------------------------------------------
            # Obtain day-specific information for plot labels
            #-------------------------------------------------------------------
            mnn_szas = np.rad2deg(np.acos(xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0, time = in_mnn_index))).values
            mnn_times = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_mnn_index).values

            #-------------------------------------------------------------------
            # Obtain cloud water path
            #-------------------------------------------------------------------
            hwp = calc_hwp(rad_tran_infile, in_mnn_index, yslice_index, zmax_index, detailed_calc) # [time, lay, x]

            #-------------------------------------------------------------------
            # Obtain two-stream (ts) and ray-tracer (rt) data
            #-------------------------------------------------------------------
            if case == "GATEIII":
                [ts_field, rt_field] = calc_atm_heating(rad_tran_infile, rad_tran_outfile, in_mnn_index, out_mnn_index, 
                    yslice_index, zmax_index, detailed_calc = detailed_calc) # [K d^{-1}], [time, lay, x]
            else:
                [ts_field, rt_field] = calc_flux_abs(rad_tran_outfile, out_mnn_index, 
                    yslice_index, zmax_index) # [W m^{-3}], [time, lay, x]
                
            #-------------------------------------------------------------------
            # Prepare data for plotting
            #-------------------------------------------------------------------
            hwp = hwp.values # [g m^{-3}], [time, lay, x]
            ts_field = ts_field.values # [time, lay, y]
            rt_field = rt_field.values # [time, lay, y]
            diff_field = rt_field - ts_field

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            hwp_max = np.max(hwp, axis = (1, 2))
            hwp_min = np.min(hwp, axis = (1, 2))

            rad_tran_max = np.max(np.stack([np.max(ts_field, axis = (1, 2)), np.max(rt_field, axis = (1, 2))]), axis = 0)
            rad_tran_min = np.min(np.stack([np.min(ts_field, axis = (1, 2)), np.min(rt_field, axis = (1, 2))]), axis = 0)

            diff_max = np.max(np.abs(diff_field), axis = (1, 2))
            diff_min = -diff_max

            #-------------------------------------------------------------------
            # Set case-dependent visualization options
            #-------------------------------------------------------------------
            heating_cmap = "hot"
            flux_cmap = "magma"

            if case == "GATEIII":
                rad_tran_cmap = heating_cmap
                rad_tran_label = r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$"
            else:
                rad_tran_cmap = flux_cmap
                rad_tran_label = r"Net Absorbed Flux $\left[ W\,m^{-3} \right]$"

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            fig, axs = plt.subplots(nrows = 4, ncols = 3,
                sharex = True, sharey = True,
                constrained_layout = True,
                figsize = (14, 14))

            # Row 1: Horizontal Water Path
            hwp_pcm = [[] for jj in range(3)]
            for jj in range(3):
                hwp_pcm[jj] = axs[0, jj].pcolormesh(x, lay, hwp[jj,:],
                    vmin = hwp_min[jj], vmax = hwp_max[jj],
                    cmap = "Blues")

            # Row 1: Two-Stream
            ts_pcm = [[] for jj in range(3)]
            for jj in range(3):
                ts_pcm[jj] = axs[1, jj].pcolormesh(x, lay, ts_field[jj,:],
                    vmin = rad_tran_min[jj], vmax = rad_tran_max[jj],
                    cmap = rad_tran_cmap)

            # Row 2: Ray-Tracer
            rt_pcm = [[] for jj in range(3)]
            for jj in range(3):
                rt_pcm[jj] = axs[2, jj].pcolormesh(x, lay, rt_field[jj,:],
                    vmin = rad_tran_min[jj], vmax = rad_tran_max[jj],
                    cmap = rad_tran_cmap)
            
            # Row 2: Ray-Tracer - Two-Stream
            diff_pcm = [[] for jj in range(3)]
            for jj in range(3):
                diff_pcm[jj] = axs[3, jj].pcolormesh(x, lay, diff_field[jj,:],
                    vmin = diff_min[jj], vmax = diff_max[jj],
                    cmap = "RdBu")

            # Colorbars
            for jj in range(3):
                hwp_cbar = fig.colorbar(hwp_pcm[jj], ax = axs[0,jj])
                rt_cbar = fig.colorbar(rt_pcm[jj], ax = axs[1:3,jj])
                diff_cbar = fig.colorbar(diff_pcm[jj], ax = axs[3,jj])

            # Labels
            lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
            fig.suptitle("Horizontal Resolution - {}".format(lr_label))
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"z $\left[ km \right]$")

            for jj in range(3):
                axs[0,jj].set_title(r"{:.2f} Hours - Solar Zenith Angle {:.1f}$^{{\circ}}$".format(mnn_times[jj], mnn_szas[jj]))
            axs[1,0].set_ylabel("Two-Stream")
            axs[2,0].set_ylabel("Ray-Tracer")
            axs[3,0].set_ylabel("Ray-Tracer - Two-Stream")
            
            hwp_cbar.ax.set_ylabel(r"Horizontal Water Path $\left[ g\,m^{-2} \right]$")
            rt_cbar.ax.set_ylabel(rad_tran_label)
            diff_cbar.ax.set_ylabel(rad_tran_label)

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            if case in ["GATEIII"]:
                plt_filename = "{}_atm_heating_day_{}.png".format(lr_str, ii)
            else:
                plt_filename = "{}_flux_abs_day_{}.png".format(lr_str, ii)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()