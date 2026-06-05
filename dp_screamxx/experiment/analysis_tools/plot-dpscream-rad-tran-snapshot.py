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
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY
from dpscream import find_mnn_indices, find_szas, find_times, get_sort_mask, \
    calc_cloud_wc, calc_sw_heating

# Script variables
prog_name: str = "plot-dpscream-rad-tran-snapshot"
prog_desc: str = "Visualize absorbed shortwave flux and atmospheric heating hates for DP-SCREAM."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time: str = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line input...".format(current_time)
    print(msg, flush = True)

    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--dp-scream-file", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path to DP-SCREAM output file.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--zmax", nargs = "?", default = 16., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--detailed-calc", nargs = "?", default = False, type = bool,
        help = ("True: Compute cloud water mass using VMRs, etc. "
            "False: Compute cloud water mass using standard values."))
        
    args: Namespace = parser.parse_args()

    dp_scream_file: str = os.path.normpath(args.dp_scream_file)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    zmax: NP_REAL = NP_REAL(args.zmax)
    detailed_calc: bool = args.detailed_calc

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Set variables used throughout the script
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Obtain Morning-Noon-Night time indices, times, SZAs
    #---------------------------------------------------------------------------
    current_time: str = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Obtaining morning-noon-night information...".format(current_time)
    print(msg, flush = True)

    mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(dp_scream_file) # [ndays, 3]
    mnn_times: NP_ARRAY[NP_REAL] = find_times(dp_scream_file, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
    mnn_szas: NP_ARRAY[NP_REAL] = find_szas(dp_scream_file, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
    ndays: NP_INT = mnn_indices.shape[0]

    sort_mask: NP_ARRAY[NP_INT] = get_sort_mask(dp_scream_file)

    #---------------------------------------------------------------------------
    # Calculate fields for each MNN of each day
    #---------------------------------------------------------------------------
    int: ii
    for ii in range(0, ndays):
        #-----------------------------------------------------------------------
        # Calculate cloud water content, x-indices for yz-slices for calculations
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating cloud water content for day {} of {}...".format(current_time, ii, ndays - 1)
        print(msg, flush = True)

        cloud_wc: XR_DATAARRAY = calc_cloud_wc(dp_scream_file, sort_mask, mnn_indices[ii]) # Cloud water content; [g m^{-3}]; [3, lev, y, x]
        x_indices: NP_ARRAY[NP_INT] = np.array([np.unravel_index(np.argmax(cloud_wc.isel(time = ll).to_numpy()), cloud_wc.shape[1:])[2] for ll in range(3)]) # [3]
        yz_slices_x: NP_ARRAY[NP_REAL] = cloud_wc["x"][x_indices].to_numpy().astype(NP_REAL) # x-location of yz-slices; [m]; [3]

        # Sneak in getting grid information before converting cloud_wc
        dx: NP_REAL = NP_REAL(cloud_wc["x"][1] - cloud_wc["x"][0])
        y: NP_ARRAY[NP_REAL] = cloud_wc["y"].to_numpy().astype(NP_REAL)

        cloud_wc: list[NP_ARRAY[NP_REAL]] = [cloud_wc.isel(time = ll, x = x_indices[ll]).to_numpy().astype(NP_REAL) for ll in range(0, 3)] # [g m^{-3}]; 3 * [lev, y]

        #-----------------------------------------------------------------------
        # Calculate horizontal water path at each YZ-slice
        #-----------------------------------------------------------------------
        hwp: list[NP_ARRAY[NP_REAL]] = [(dx * cloud_wc[ll]) for ll in range(0, 3)] # [g m^{-2}], 3 * [lay, y]

        #-----------------------------------------------------------------------
        # Calculate shortwave radiative heating rates
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating radiative quantities for day {} of {}...".format(current_time, ii, ndays - 1)
        print(msg, flush = True)

        sw_heating: list[NP_ARRAY[NP_REAL]] = calc_sw_heating(dp_scream_file, sort_mask, 
            mnn_indices[ii], x_indices, method = "pdel") # Shortwave heating rate; [K d^{-1}]; 3 * [lev, y]

        #-----------------------------------------------------------------------
        # Obtain plotting bounds
        #-----------------------------------------------------------------------
        y_window_width: NP_REAL = 3. * zmax # Width of y window [km]
        
        y_bounds: NP_ARRAY[NP_REAL]
        y_bound_indices: list[NP_ARRAY[NP_INT]]
        if y_window_width > y.max() - y.min():
            y_bounds = np.tile(np.array([y.min(), y.max()]), [3, 1]) # [3, 2]
            y_bound_indices = [np.array([0, -1], dtype = NP_INT) for _ in range(0, 3)]
        else:
            y_bounds = np.zeros([3, 2], dtype = NP_REAL) # Limits for horizontal plotting axis (called xlim in matplotlib)
            y_bound_indices = [np.zeros([2], dtype = NP_INT) for _ in range(0, 3)]
            for ll in range(3):
                hwp_max_index: NP_ARRAY[NP_INT] = np.unravel_index(np.argmax(hwp[ll].to_numpy()), hwp[ll].shape) # [lay_index, y_index] of maximal hwp
                y_loc: NP_REAL = y[hwp_max_index[1]] # Y-location of maximal hwp [km]
                y_bounds[ll,:] = [y_loc - y_window_width / 2., y_loc + y_window_width / 2.]

                # Don't have to worry about shifting window past the edge after this block
                # because the window is not as wide as the domain
                if y_bounds[ll,0] < y.min():
                    y_bounds[ll,:] += (y.min() - y_bounds[ll,0])

                if y_bounds[ll,1] > y.max():
                    y_bounds[ll,:] += (y.max() - y_bounds[ll,1])

                y_bound_indices[ll][0] = np.max(np.where(y - y_bounds[ll,0] <= 0)[0])
                y_bound_indices[ll][1] = np.min(np.where(y_bounds[ll,1] - y <= 0)[0]) + 1 # To include endpoint, add 1
        y_slices = [y[y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(0, 3)]

        #-----------------------------------------------------------------------
        # Trim hwp to the yslice
        #-----------------------------------------------------------------------
        hwp = [hwp[ll].isel(y = slice(y_bound_indices[ll][0], y_bound_indices[ll][1])) for ll in range(3)]

        # TO-DO: CONTINUE FROM HERE

        #---------------------------------------------------------------
        # Obtain two-stream (ts) and ray-tracer (rt) data
        #---------------------------------------------------------------
        ts_field = [[] for _ in range(3)]
        rt_field = [[] for _ in range(3)]

        if field_key == "atm_heating":
            for ll in range(3):
                [ts_field[ll], rt_field[ll]] = calc_atm_heating(rad_tran_infile, 
                    rad_tran_outfile, in_mnn_index[ll], out_mnn_index[ll], 
                    x_index = xslice_indexes[ll], y_index = slice(y_bound_indices[ll][0], y_bound_indices[ll][1]),
                    zmax_index = zmax_index, detailed_calc = False) # [K d^{-1}], [lay, y]
        elif field_key == "abs_flux":
            for ll in range(3):
                [ts_field[ll], rt_field[ll]] = calc_abs_flux(rad_tran_outfile, 
                    out_mnn_index[ll], x_index = xslice_indexes[ll], 
                    y_index = slice(y_bound_indices[ll][0], y_bound_indices[ll][1]),
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
                y = slice(y_bound_indices[ll][0], y_bound_indices[ll][1]), x = xslice_indexes[ll]).to_numpy() / 100. # [hPa], [3][lay, y]
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
            hwp_pcm[ll] = axs[0, ll].pcolormesh(y_slices[ll], lay, hwp[ll],
                vmin = hwp_min[ll], vmax = hwp_max[ll],
                cmap = hwp_cmap)
            #p_contour = axs[0, ll].contour(y_slices[ll], lay, p_lay[ll],
            #    levels = p_levels, colors = "#000000", linewidths = 2.)
            #axs[0,ll].clabel(p_contour)

        # Row 1: Two-Stream
        ts_pcm = [[] for _ in range(3)]
        for ll in range(3):
            ts_pcm[ll] = axs[1, ll].pcolormesh(y_slices[ll], lay, ts_field[ll],
                norm = colors.LogNorm(vmin = rad_tran_min[ll], vmax = rad_tran_max[ll]),
                cmap = rad_tran_cmap)

        # Row 2: Ray-Tracer
        rt_pcm = [[] for _ in range(3)]
        for ll in range(3):
            rt_pcm[ll] = axs[2, ll].pcolormesh(y_slices[ll], lay, rt_field[ll],
                norm = colors.LogNorm(vmin = rad_tran_min[ll], vmax = rad_tran_max[ll]),
                cmap = rad_tran_cmap)

        # Row 3: Ray-Tracer - Two-Stream
        diff_pcm = [[] for _ in range(3)]
        for ll in range(3):
            diff_pcm[ll] = axs[3, ll].pcolormesh(y_slices[ll], lay, diff_field[ll],
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
                axs[ll,mm].set_xlim(y_bounds[mm,:])

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