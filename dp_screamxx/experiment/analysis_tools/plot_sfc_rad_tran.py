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

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y

        #-----------------------------------------------------------------------
        # Plot multiple quantities
        #-----------------------------------------------------------------------
        # Net surface flux, surface heating, absorbed radiative flux, 
        # atmospheric heating
        if case == "GATEIII":
            field_keys = ["net_sfc_flux", "sfc_heating"]
            rad_tran_labels = [r"Net Surface Flux $\left[ W\,m^{-2} \right]$",
                r"Surface Heating Rate $\left[ K\,d^{-1} \right]$"]
        else:
            field_keys = ["net_sfc_flux"]
            rad_tran_labels = [r"Net Surface Flux $\left[ W\,m^{-2} \right]$"]

        for jj in range(len(field_keys)):
            field_key = field_keys[jj]
            rad_tran_label = rad_tran_labels[jj]

            ndays = in_mnn_indices.shape[0]
            for kk in range(ndays):
                in_mnn_index = in_mnn_indices[kk]
                out_mnn_index = out_mnn_indices[kk]

                #-------------------------------------------------------------------
                # Obtain day-specific information for plot labels
                #-------------------------------------------------------------------
                mnn_szas = np.rad2deg(np.acos(xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0, time = in_mnn_index))).values
                mnn_times = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_mnn_index).values

                #-------------------------------------------------------------------
                # Obtain vertical cloud water path
                #-------------------------------------------------------------------
                vwp = calc_vwp(rad_tran_infile, in_mnn_index, detailed_calc = detailed_calc)

                #-------------------------------------------------------------------
                # Obtain two-stream (ts) and ray-tracer (rt) data
                #-------------------------------------------------------------------
                [ts_sfc_net, rt_sfc_net] = calc_sfc_net(rad_tran_outfile, out_mnn_index)
                if field_key == "sfc_heating":
                    if case == "GATEIII":
                        ts_field = (ts_sfc_net / (rho_sw * cp_sw * h_m)) * sec_per_day # [K d^{-1}], [time, y, x]
                        rt_field = (rt_sfc_net / (rho_sw * cp_sw * h_m)) * sec_per_day # [K d^{-1}], [time, y, x]
                elif field_key == "net_sfc_flux":
                    ts_field = ts_sfc_net # [W m^{-2}], [time, y, x]
                    rt_field = rt_sfc_net # [W m^{-2}], [time, y, x]

                # BUG: It seems like the x and y coordinates are transposed when compared to the absorbed fluxes.
                # We adjust here by not transposing.

                #-------------------------------------------------------------------
                # Prepare data for plotting
                #-------------------------------------------------------------------
                vwp = vwp.values # [g m^{-2}], [time, x, y]
                ts_field = ts_sfc_net.values # [time, x, y]
                rt_field = rt_sfc_net.values # [time, x, y]
                diff_field = rt_field - ts_field # [time, x, y]

                #-------------------------------------------------------------------
                # Obtain data bounds
                #-------------------------------------------------------------------
                vwp_max = np.max(vwp, axis = (1, 2))
                vwp_min = np.min(vwp, axis = (1, 2))

                rad_tran_max = np.max(np.stack([np.max(ts_field, axis = (1, 2)), np.max(rt_field, axis = (1, 2))]), axis = 0)
                rad_tran_min = np.min(np.stack([np.min(ts_field, axis = (1, 2)), np.min(rt_field, axis = (1, 2))]), axis = 0)

                diff_max = np.max(np.abs(diff_field), axis = (1, 2))
                diff_min = -diff_max

                #-------------------------------------------------------------------
                # Set case-dependent visualization options
                #-------------------------------------------------------------------
                if field_key == "sfc_heating":
                    rad_tran_cmap = heating_cmap
                elif field_key == "net_sfc_flux":
                    rad_tran_cmap = flux_cmap

                #-------------------------------------------------------------------
                # Plot the data
                #-------------------------------------------------------------------
                fig, axs = plt.subplots(nrows = 4, ncols = 3,
                    sharex = True, sharey = True,
                    constrained_layout = True,
                    figsize = (14, 14))

                # Row 1: Vertical Water Path
                vwp_pcm = [[] for ll in range(3)]
                for ll in range(3):
                    vwp_pcm[ll] = axs[0, ll].pcolormesh(x, y, vwp[ll,...],
                        vmin = vwp_min[ll], vmax = vwp_max[ll],
                        cmap = "Blues")

                # Row 1: Two-Stream
                ts_pcm = [[] for ll in range(3)]
                for ll in range(3):
                    ts_pcm[ll] = axs[1, ll].pcolormesh(x, y, ts_field[ll,...],
                        vmin = rad_tran_min[ll], vmax = rad_tran_max[ll],
                        cmap = rad_tran_cmap)

                # Row 2: Ray-Tracer
                rt_pcm = [[] for ll in range(3)]
                for ll in range(3):
                    rt_pcm[ll] = axs[2, ll].pcolormesh(x, y, rt_field[ll,...],
                        vmin = rad_tran_min[ll], vmax = rad_tran_max[ll],
                        cmap = rad_tran_cmap)

                # Row 2: Ray-Tracer - Two-Stream
                diff_pcm = [[] for ll in range(3)]
                for ll in range(3):
                    diff_pcm[ll] = axs[3, ll].pcolormesh(x, y, diff_field[ll,...],
                        vmin = diff_min[ll], vmax = diff_max[ll],
                        cmap = "RdBu")

                # Colorbars
                for ll in range(3):
                    vwp_cbar = fig.colorbar(vwp_pcm[ll], ax = axs[0,ll])
                    rt_cbar = fig.colorbar(rt_pcm[ll], ax = axs[1:3,ll])
                    diff_cbar = fig.colorbar(diff_pcm[ll], ax = axs[3,ll])

                # Labels
                lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
                fig.suptitle("Horizontal Resolution - {}".format(lr_label))
                fig.supxlabel(r"x $\left[ km \right]$")
                fig.supylabel(r"y $\left[ km \right]$")

                for ll in range(3):
                    axs[0,ll].set_title(r"{:.2f} Hours - Solar Zenith Angle {:.1f}$^{{\circ}}$".format(mnn_times[ll], mnn_szas[ll]))
                axs[1,0].set_ylabel("Two-Stream")
                axs[2,0].set_ylabel("Ray-Tracer")
                axs[3,0].set_ylabel("Ray-Tracer - Two-Stream")

                vwp_cbar.ax.set_ylabel(r"Vertical Cloud Water Path $\left[ g\,m^{-2} \right]$")
                rt_cbar.ax.set_ylabel(rad_tran_label)
                diff_cbar.ax.set_ylabel(rad_tran_label)

                #-------------------------------------------------------------------
                # Save the plot to file
                #-------------------------------------------------------------------
                plt_filename = "{}_{}_day_{}.png".format(lr_str, field_key, kk)
                plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
                fig.savefig(plt_filepath, dpi = 200)
                plt.close(fig)

if __name__ == "__main__":
    main()