# Library imports
import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import xarray as xr

# Local imports
from consts import flux_cmap, heating_cmap
from find_zmax_index import find_zmax_index
from find_pairs import find_pairs
from find_mnn_indices import find_mnn_indices
from calc_abs_flux import calc_abs_flux
from calc_atm_heating import calc_atm_heating
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
    parser.add_argument("--detailed-calc", nargs = "?", default = False, type = bool,
        help = ("True: Compute cloud water mass using VMRs, etc. "
            "False: Compute cloud water mass using standard values."))
    args = parser.parse_args()

    rad_tran_indir  = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir = os.path.normpath(args.rad_tran_vizdir)
    working_dir = args.working_dir
    recalculate = args.recalculate
    lrs = [int(lr) for lr in args.lr.split(",")]
    zmax = args.zmax
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
    # Set options common across all files and plots
    #---------------------------------------------------------------------------
    hwp_cmap = plt.get_cmap("Blues")
    hwp_cmap.set_under("gray")

    p_levels = [100., 300., 500., 750., 1000.]

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
        x = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["x"].to_numpy() / 1000 # [km]
        y = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["y"].to_numpy() / 1000 # [km]
        lay = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["lay"].to_numpy() / 1000 # [km]

        zmax_index = find_zmax_index(lay, zmax)
        lay = lay[:zmax_index]

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y

        #-----------------------------------------------------------------------
        # Plot multiple quantities
        #-----------------------------------------------------------------------
        # Net surface flux, surface heating, absorbed radiative flux, 
        # atmospheric heating
        field_keys = ["abs_flux", "atm_heating"]
        rad_tran_labels = [r"Absorbed Flux $\left[ W\,m^{-3} \right]$",
            r"Atmosphere Heating Rate $\left[ K\,d^{-1} \right]$"]

        for jj in range(len(field_keys)):
            field_key = field_keys[jj]
            rad_tran_label = rad_tran_labels[jj]

            ndays = in_mnn_indices.shape[0]
            for kk in range(ndays):
                in_mnn_index = in_mnn_indices[kk]
                out_mnn_index = out_mnn_indices[kk]

                #---------------------------------------------------------------
                # Obtain day-specific information for plot labels
                #---------------------------------------------------------------
                mnn_szas = np.rad2deg(np.acos(xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0, time = in_mnn_index))).to_numpy()
                mnn_times = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_mnn_index).to_numpy()

                #---------------------------------------------------------------
                # Get YZ-slice at each time using maximal water content
                #---------------------------------------------------------------
                wc = calc_wc(rad_tran_infile, in_mnn_index, zmax_index = zmax_index) # [g m^{-3}], [time, lay, y, x]
                xslice_indexes = [np.unravel_index(np.argmax(wc.isel(time = ll).to_numpy()), wc.shape[1:])[2] for ll in range(3)]
                xslices = x[xslice_indexes]

                #---------------------------------------------------------------
                # Calculate horizontal water path at each YZ-slice
                #---------------------------------------------------------------
                hwp = [((dx * 1.e3) * wc.isel(time = ll, x = xslice_indexes[ll])) for ll in range(3)] # [g m^{-2}], [3][lay, y]

                #---------------------------------------------------------------
                # Obtain plotting bounds
                #---------------------------------------------------------------
                y_window_width = 3. * zmax # Width of y window [km]
                
                if y_window_width > y.max() - y.min():
                    yslice_indexes = [np.array([0,-1], dtype = np.int32) for _ in range(3)]
                    hlim_loc = np.tile(np.array([y.min(), y.max()]), [3, 1])
                else:
                    hlim_loc = np.zeros([3,2], dtype = np.float64) # Limits for horizontal plotting axis (called xlim in matplotlib)
                    yslice_indexes = [np.zeros([2], dtype = np.int32) for _ in range(3)]
                    for ll in range(3):
                        hwp_max_index = np.unravel_index(np.argmax(hwp[ll].to_numpy()), hwp[ll].shape) # [lay_index, y_index] of maximal hwp
                        y_loc = y[hwp_max_index[1]] # Y-location of maximal hwp [km]
                        hlim_loc[ll,:] = [y_loc - y_window_width / 2., y_loc + y_window_width / 2.]

                        # Don't have to worry about shifting window past the edge after this block
                        # because the window is not as wide as the domain
                        if hlim_loc[ll,0] < y.min():
                            hlim_loc[ll,:] += (y.min() - hlim_loc[ll,0])

                        if hlim_loc[ll,1] > y.max():
                            hlim_loc[ll,:] += (y.max() - hlim_loc[ll,1])

                        yslice_indexes[ll][0] = np.max(np.where(y - hlim_loc[ll,0] <= 0)[0])
                        yslice_indexes[ll][1] = np.min(np.where(hlim_loc[ll,1] - y <= 0)[0]) + 1 # To include endpoint, add 1
                yslices = [y[yslice_indexes[ll][0]:yslice_indexes[ll][1]] for ll in range(3)]

                #---------------------------------------------------------------
                # Trim hwp to the yslice
                #---------------------------------------------------------------
                hwp = [hwp[ll].isel(y = slice(yslice_indexes[ll][0], yslice_indexes[ll][1])) for ll in range(3)]

                #---------------------------------------------------------------
                # Obtain two-stream (ts) and ray-tracer (rt) data
                #---------------------------------------------------------------
                ts_field = [[] for _ in range(3)]
                rt_field = [[] for _ in range(3)]

                if field_key == "atm_heating":
                    for ll in range(3):
                        [ts_field[ll], rt_field[ll]] = calc_atm_heating(rad_tran_infile, 
                            rad_tran_outfile, in_mnn_index[ll], out_mnn_index[ll], 
                            x_index = xslice_indexes[ll], y_index = slice(yslice_indexes[ll][0], yslice_indexes[ll][1]),
                            zmax_index = zmax_index, detailed_calc = False) # [K d^{-1}], [lay, y]
                elif field_key == "abs_flux":
                    for ll in range(3):
                        [ts_field[ll], rt_field[ll]] = calc_abs_flux(rad_tran_outfile, 
                            out_mnn_index[ll], x_index = xslice_indexes[ll], 
                            y_index = slice(yslice_indexes[ll][0], yslice_indexes[ll][1]),
                            zmax_index = zmax_index) # [W m^{-3}], [time, lay, y]

                #---------------------------------------------------------------
                # Prepare data for plotting
                #---------------------------------------------------------------
                hwp = [hwp[ll].to_numpy() for ll in range(3)] # [3][lay, y]
                ts_field = [ts_field[ll].to_numpy() for ll in range(3)] # [3][lay, y]
                rt_field = [rt_field[ll].to_numpy() for ll in range(3)] # [3][lay, y]
                diff_field = [rt_field[ll] - ts_field[ll] for ll in range(3)] # [3][lay, y]

                #---------------------------------------------------------------
                # Obtain data bounds
                #---------------------------------------------------------------
                hwp_max = [hwp[ll].max() for ll in range(3)]
                hwp_min = [hwp[ll].min() for ll in range(3)]

                rad_tran_max = [max(ts_field[ll].max(), rt_field[ll].max()) for ll in range(3)]
                rad_tran_min = [min(ts_field[ll].min(), rt_field[ll].min()) for ll in range(3)]

                diff_max = [np.abs(diff_field[ll]).max() for ll in range(3)]
                diff_min = [-diff_max[ll] for ll in range(3)]

                #---------------------------------------------------------------
                # Obtain information for plotting pressure contours
                #---------------------------------------------------------------
                p_lay = [[] for _ in range(3)]
                rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)
                for ll in range(3):
                    p_lay[ll] = rad_tran_inds["p_lay"].isel(time = in_mnn_index[ll],
                        lay = slice(0, zmax_index), 
                        y = slice(yslice_indexes[ll][0], yslice_indexes[ll][1]), x = xslice_indexes[ll]).to_numpy() / 100. # [hPa], [3][lay, y]
                rad_tran_inds.close()

                #---------------------------------------------------------------
                # Set case-dependent visualization options
                #---------------------------------------------------------------
                if field_key == "atm_heating":
                    rad_tran_cmap = heating_cmap
                elif field_key == "abs_flux":
                    rad_tran_cmap = flux_cmap

                #---------------------------------------------------------------
                # Plot the data
                #---------------------------------------------------------------
                fig_height = 5.25
                fig_base_size = np.array([(y_window_width / zmax) * fig_height, fig_height])
                fig, axs = plt.subplots(nrows = 4, ncols = 3,
                    sharex = "col", sharey = True,
                    constrained_layout = True,
                    figsize = 3. * fig_base_size)

                # Row 0: Horizontal Water Path
                hwp_pcm = [[] for _ in range(3)]
                for ll in range(3):
                    hwp_pcm[ll] = axs[0, ll].pcolormesh(yslices[ll], lay, hwp[ll],
                        vmin = hwp_min[ll], vmax = hwp_max[ll],
                        cmap = hwp_cmap)
                    #p_contour = axs[0, ll].contour(yslices[ll], lay, p_lay[ll],
                    #    levels = p_levels, colors = "#000000", linewidths = 2.)
                    #axs[0,ll].clabel(p_contour)

                # Row 1: Two-Stream
                ts_pcm = [[] for _ in range(3)]
                for ll in range(3):
                    ts_pcm[ll] = axs[1, ll].pcolormesh(yslices[ll], lay, ts_field[ll],
                        norm = colors.LogNorm(vmin = rad_tran_min[ll], vmax = rad_tran_max[ll]),
                        cmap = rad_tran_cmap)

                # Row 2: Ray-Tracer
                rt_pcm = [[] for _ in range(3)]
                for ll in range(3):
                    rt_pcm[ll] = axs[2, ll].pcolormesh(yslices[ll], lay, rt_field[ll],
                        norm = colors.LogNorm(vmin = rad_tran_min[ll], vmax = rad_tran_max[ll]),
                        cmap = rad_tran_cmap)

                # Row 3: Ray-Tracer - Two-Stream
                diff_pcm = [[] for _ in range(3)]
                for ll in range(3):
                    diff_pcm[ll] = axs[3, ll].pcolormesh(yslices[ll], lay, diff_field[ll],
                        vmin = diff_min[ll], vmax = diff_max[ll],
                        cmap = "RdBu")

                # Colorbars
                for ll in range(3):
                    hwp_cbar = fig.colorbar(hwp_pcm[ll], ax = axs[0,ll])
                    rt_cbar = fig.colorbar(rt_pcm[ll], ax = axs[1:3,ll])
                    diff_cbar = fig.colorbar(diff_pcm[ll], ax = axs[3,ll])

                # Labels
                lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
                fig.suptitle("Horizontal Resolution - {}".format(lr_label))
                fig.supxlabel(r"y $\left[ km \right]$")
                fig.supylabel(r"z $\left[ km \right]$")

                for ll in range(3):
                    axs[0,ll].set_title(r"{:.2f} Hours - Solar Zenith Angle {:.1f}$^{{\circ}}$ - $x$ = {:.2f} $\left[ km \right]$".format(mnn_times[ll], mnn_szas[ll], xslices[ll]))
                axs[1,0].set_ylabel("Two-Stream")
                axs[2,0].set_ylabel("Ray-Tracer")
                axs[3,0].set_ylabel("Ray-Tracer - Two-Stream")

                hwp_cbar.ax.set_ylabel(r"Horizontal Cloud Water Path $\left[ g\,m^{-2} \right]$")
                rt_cbar.ax.set_ylabel(rad_tran_label)
                diff_cbar.ax.set_ylabel(rad_tran_label)

                # Set horizontal limits
                for ll in range(4):
                    for mm in range(3):
                        axs[ll,mm].set_xlim(hlim_loc[mm,:])

                # Aspect ratio
                for ll in range(4):
                    for mm in range(3):
                        axs[ll,mm].set_aspect("equal")

                #---------------------------------------------------------------
                # Save the plot to file
                #---------------------------------------------------------------
                plt_filename = "{}_{}_day_{}.png".format(lr_str, field_key, kk)
                plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
                fig.savefig(plt_filepath, dpi = 200)
                plt.close(fig)

if __name__ == "__main__":
    main()