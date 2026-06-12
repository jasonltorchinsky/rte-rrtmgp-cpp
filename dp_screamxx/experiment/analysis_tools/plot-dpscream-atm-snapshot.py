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
from consts.visual import heating_cmap, lw_cmap, iw_cmap, cw_cmap
from dpscream import find_mnn_indices, find_szas, find_times, get_sort_mask, get_z, \
    calc_cloud_wc, calc_dei, calc_rel, calc_rh

# Script variables
prog_name: str = "plot-dpscream-atm-snapshot"
prog_desc: str = "Visualize atmospheric state for DP-SCREAM."

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
    # Obtain Morning-Noon-Night time indices, times, SZAs, zmax index
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
        yz_slices_x: NP_ARRAY[NP_REAL] = cloud_wc["x"][x_indices].to_numpy().astype(NP_REAL) * 1.e-3 # x-location of yz-slices; [km]; [3]

        # Sneak in getting grid information before converting cloud_wc
        dx: NP_REAL = NP_REAL(cloud_wc["x"][1] - cloud_wc["x"][0]) # [m]
        y: NP_ARRAY[NP_REAL] = cloud_wc["y"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]

        cloud_wc: list[NP_ARRAY[NP_REAL]] = [cloud_wc.isel(time = ll, x = x_indices[ll]).to_numpy().astype(NP_REAL) for ll in range(0, 3)] # [g m^{-3}]; 3 * [lev, y]

        #-----------------------------------------------------------------------
        # Calculate horizontal water path at each YZ-slice
        #-----------------------------------------------------------------------
        hwp: list[NP_ARRAY[NP_REAL]] = [(dx * cloud_wc[ll]) for ll in range(0, 3)] # [g m^{-2}], 3 * [lay, y]

        #-----------------------------------------------------------------------
        # Calculate relative humidity, cloud liquid water effective radius, 
        # cloud ice water effective diameter
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating relative humidity for day {} of {}...".format(current_time, ii, ndays - 1)
        print(msg, flush = True)

        rh: list[NP_ARRAY[NP_REAL]] = calc_rh(dp_scream_file, sort_mask,
            mnn_indices[ii], x_indices) # Relative humidity; [N/A]; 3 * [lay, y]

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating cloud liquid water effective radius for day {} of {}...".format(current_time, ii, ndays - 1)
        print(msg, flush = True)

        rel: list[NP_ARRAY[NP_REAL]] = calc_rel(dp_scream_file, sort_mask,
            mnn_indices[ii], x_indices) # Cloud liquid water effective radius; [μm]; 3 * [lay, y]

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating cloud ice water effective diameter for day {} of {}...".format(current_time, ii, ndays - 1)
        print(msg, flush = True)

        dei: list[NP_ARRAY[NP_REAL]] = calc_dei(dp_scream_file, sort_mask,
            mnn_indices[ii], x_indices) # Cloud ice water effective diameter; [μm]; 3 * [lay, y]
        
        #-----------------------------------------------------------------------
        # Get vertical grids
        #-----------------------------------------------------------------------
        z_mid: list[NP_ARRAY[NP_REAL]] = get_z(dp_scream_file, sort_mask, mnn_indices[ii],
            x_indices, levels = "mid") # [m]
        z_mid = [z_mid[ll] * 1.e-3 for ll in range(0, 3)] # [m] => [km]

        #-----------------------------------------------------------------------
        # Obtain plotting bounds
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Obtaining plotting bounds...".format(current_time)
        print(msg, flush = True)

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
                hwp_max_index: NP_ARRAY[NP_INT] = np.unravel_index(np.argmax(hwp[ll]), hwp[ll].shape) # [lay_index, y_index] of maximal hwp
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
        # Trim fields to the yz-slice
        #-----------------------------------------------------------------------
        hwp = [hwp[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
        rh = [rh[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
        rel = [rel[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
        dei = [dei[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
        
        #-----------------------------------------------------------------------
        # Get grids for the yz-slices
        #-----------------------------------------------------------------------
        z_grid = [z_mid[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
        nz: NP_INT = NP_INT(z_mid[0].shape[0])
        y_grid: list[NP_ARRAY[NP_REAL]] = [np.tile(y_slices[ll], (nz, 1)) for ll in range(0, 3)]

        #-----------------------------------------------------------------------
        # Obtain data bounds
        #-----------------------------------------------------------------------
        hwp_max: list[NP_REAL] = [hwp[ll].max() for ll in range(3)]
        hwp_min: list[NP_REAL] = [hwp[ll].min() for ll in range(3)]

        rh_max: list[NP_REAL] = [rh[ll].max() for ll in range(3)]
        rh_min: list[NP_REAL] = [rh[ll].min() for ll in range(3)]

        rel_max: list[NP_REAL] = [rel[ll].max() for ll in range(3)]
        rel_min: list[NP_REAL] = [rel[ll].min() for ll in range(3)]

        dei_max: list[NP_REAL] = [dei[ll].max() for ll in range(3)]
        dei_min: list[NP_REAL] = [dei[ll].min() for ll in range(3)]

        #-----------------------------------------------------------------------
        # Plot the data
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Plotting data...".format(current_time)
        print(msg, flush = True)

        nrows: NP_INT = NP_INT(4)
        ncols: NP_INT = NP_INT(3)
        fig_height: NP_REAL = NP_REAL(3.)
        fig_base_size = np.array([(y_window_width / zmax) * fig_height, fig_height])
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = "col", sharey = True,
            constrained_layout = True,
            figsize = 3. * fig_base_size)

        # Row 0: Horizontal Water Path
        hwp_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, 3)]
        ll: int
        for ll in range(0, ncols):
            hwp_pcm[ll] = axs[0, ll].pcolormesh(y_grid[ll], z_grid[ll], hwp[ll],
                vmin = hwp_min[ll], vmax = hwp_max[ll],
                cmap = cw_cmap)

        # Row 1: Relative Humidity
        rh_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
        ll: int
        for ll in range(0, ncols):
            rh_pcm[ll] = axs[1, ll].pcolormesh(y_slices[ll], z_grid[ll], rh[ll],
                norm = colors.LogNorm(vmin = rh_min[ll], vmax = rh_max[ll]),
                cmap = lw_cmap)

        # Row 2: Cloud Liquid Water Effective Radius
        rel_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
        ll: int
        for ll in range(0, ncols):
            rel_pcm[ll] = axs[2, ll].pcolormesh(y_slices[ll], z_grid[ll], rel[ll],
                vmin = rel_min[ll], vmax = rel_max[ll],
                cmap = lw_cmap)

        # Row 3: Cloud Ice Water Effective Diameter
        dei_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
        ll: int
        for ll in range(0, ncols):
            dei_pcm[ll] = axs[3, ll].pcolormesh(y_slices[ll], z_grid[ll], dei[ll],
                vmin = rel_min[ll], vmax = rel_max[ll],
                cmap = iw_cmap)

        # Colorbars
        for ll in range(0, ncols):
            hwp_cbar = fig.colorbar(hwp_pcm[ll], ax = axs[0,ll])
            rh_cbar = fig.colorbar(rh_pcm[ll], ax = axs[1,ll])
            rel_cbar = fig.colorbar(rel_pcm[ll], ax = axs[2,ll])
            dei_cbar = fig.colorbar(dei_pcm[ll], ax = axs[3,ll])

        # Labels
        fig.suptitle("DP-SCREAM Atmospheric Radiative Transfer")
        fig.supxlabel(r"y $\left[ km \right]$")
        fig.supylabel(r"z $\left[ km \right]$")

        for ll in range(0, ncols):
            col_title: str = (r"{:.2f} Hours - ".format(mnn_times[ii,ll])
                + r"Solar Zenith Angle {:.1f}$^{{\circ}}$ - ".format(mnn_szas[ii,ll])
                + r"$x$ = {:.2f} $\left[ km \right]$".format(yz_slices_x[ll]))
            axs[0,ll].set_title(col_title)

        hwp_cbar.ax.set_ylabel(r"Horizontal Cloud Water Path $\left[ g\,m^{-2} \right]$")
        rh_cbar.ax.set_ylabel(r"$rh$ $\left[ Pa\,Pa^{-1} \right]$")
        rel_cbar.ax.set_ylabel(r"$rel$ $\left[ \mu m \right]$")
        dei_cbar.ax.set_ylabel(r"$dei$ $\left[ \mu m \right]$")

        # Set horizontal limits
        ll: int
        mm: int
        for ll in range(0, nrows):
            for mm in range(0, ncols):
                axs[ll,mm].set_xlim(y_bounds[mm,:])
                axs[ll,mm].set_ylim((0.0, zmax))

        # Aspect ratio
        ll: int
        mm: int
        for ll in range(0, nrows):
            for mm in range(0, ncols):
                axs[ll,mm].set_aspect("equal")

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        plt_filename = "dpscream_atm_day_{}.png".format(ii)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()