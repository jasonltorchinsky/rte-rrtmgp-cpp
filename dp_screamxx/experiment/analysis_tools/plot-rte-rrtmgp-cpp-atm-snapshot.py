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
from consts.numeric import NP_SMALL
from consts.visual import lw_cmap, iw_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_dei, calc_rel, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-atm-snapshot"
prog_desc: str = "Visualize atmospheric state for RTE-RRTMGP-CPP."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    msg: str = "Parsing command-line input..."
    print_msg(msg)

    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--zmax", nargs = "?", default = 16., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    zmax: NP_REAL = NP_REAL(args.zmax)

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
    [rad_tran_infiles, _] = find_inout_pairs(rad_tran_indir, None, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)
        grid: dict = find_grid(rad_tran_infile)

        #-----------------------------------------------------------------------
        # Obtain Morning-Noon-Night time indices, times, SZAs, zmax index
        #-----------------------------------------------------------------------
        msg: str = "Obtaining morning-noon-night information..."
        print_msg(msg)

        mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
        mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(mnn_indices.shape[0])

        #-----------------------------------------------------------------------
        # Calculate fields for each MNN of each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate cloud water content, x-indices for yz-slices for calculations
            #-------------------------------------------------------------------
            msg: str = "Calculating cloud water content for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj], zmax = zmax) # Cloud water content; [g m^{-3}]; [3, lev, y, x]
            x_indices: NP_ARRAY[NP_INT] = np.array([np.unravel_index(np.argmax(cloud_wc.isel(time = ll).to_numpy()), cloud_wc.shape[1:])[2] for ll in range(3)]) # [3]
            yz_slices_x: NP_ARRAY[NP_REAL] = cloud_wc["x"][x_indices].to_numpy().astype(NP_REAL) * 1.e-3 # x-location of yz-slices; [km]; [3]

            # Sneak in getting grid information before converting cloud_wc
            dx: NP_REAL = NP_REAL(cloud_wc["x"][1] - cloud_wc["x"][0]) # [m]
            y: NP_ARRAY[NP_REAL] = cloud_wc["y"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]
            z: NP_ARRAY[NP_REAL] = cloud_wc["lay"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]

            cloud_wc: list[NP_ARRAY[NP_REAL]] = [cloud_wc.isel(time = ll, x = x_indices[ll]).to_numpy().astype(NP_REAL) for ll in range(0, 3)] # [g m^{-3}]; 3 * [lev, y]

            #-------------------------------------------------------------------
            # Calculate effective sizes
            #-------------------------------------------------------------------
            msg: str = "Calculating cloud water effective sizes for day {} of {}...".format(current_time, jj, ndays - 1)
            print_msg(msg)

            rel: list[NP_ARRAY[NP_REAL]] = calc_rel(rad_tran_infile, mnn_indices[jj], x_indices, zmax = zmax) # Cloud liquid water effective radius; [μm]; 3 * [lay, y]
            dei: list[NP_ARRAY[NP_REAL]] = calc_dei(rad_tran_infile, mnn_indices[jj], x_indices, zmax = zmax) # Cloud ice water effective diameter; [μm]; 3 * [lay, y]

            #-------------------------------------------------------------------
            # Obtain plotting bounds
            #-------------------------------------------------------------------
            msg: str = "Obtaining plotting bounds..."
            print_msg(msg)

            y_window_width: NP_REAL = 3. * zmax # Width of y window [km]
            y_window_indices: list[NP_ARRAY[NP_INT]] = [[] for _ in range(0, 3)]
            for ll in range(0, 3):
                y_window_indices = find_y_window_indices(y, cloud_wc[ll], y_window_width)
            y_windows = [y[y_window_indices[ll][0]:y_window_indices[ll][1]] for ll in range(0, 3)]

            #-------------------------------------------------------------------
            # Trim fields to the yz-slice
            #-------------------------------------------------------------------
            cloud_wc = [cloud_wc[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(0, 3)]   
            rel = [rel[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(0, 3)]
            dei = [dei[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(0, 3)]
            
            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            cloud_wc_max: list[NP_REAL] = [cloud_wc[ll].max() for ll in range(3)]
            cloud_wc_min: list[NP_REAL] = [cloud_wc[ll].min() for ll in range(3)]

            rel_max: list[NP_REAL] = [rel[ll].max() for ll in range(3)]
            rel_min: list[NP_REAL] = [rel[ll].min() for ll in range(3)]

            dei_max: list[NP_REAL] = [dei[ll].max() for ll in range(3)]
            dei_min: list[NP_REAL] = [dei[ll].min() for ll in range(3)]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting data...".format(current_time)
            print(msg, flush = True)

            nrows: NP_INT = NP_INT(3)
            ncols: NP_INT = NP_INT(3)
            fig_height: NP_REAL = NP_REAL(3.)
            fig_base_size = np.array([(y_window_width / zmax) * fig_height, fig_height])
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
                sharex = "col", sharey = True,
                constrained_layout = True,
                figsize = 3. * fig_base_size)

            # Row 0: Horizontal Water Path
            cloud_wc_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                cloud_wc_pcm[ll] = axs[0, ll].pcolormesh(y_slices[ll], z, cloud_wc[ll],
                    vmin = cloud_wc_min[ll], vmax = cloud_wc_max[ll],
                    cmap = cw_cmap)

            # Row 1: Cloud Liquid Water Effective Radius
            rel_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                rel_pcm[ll] = axs[1, ll].pcolormesh(y_slices[ll], z, rel[ll],
                    vmin = rel_min[ll], vmax = rel_max[ll],
                    cmap = lw_cmap)

            # Row 2: Cloud Ice Water Effective Diameter
            dei_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                dei_pcm[ll] = axs[2, ll].pcolormesh(y_slices[ll], z, dei[ll],
                    vmin = rel_min[ll], vmax = rel_max[ll],
                    cmap = iw_cmap)

            # Colorbars
            for ll in range(0, ncols):
                cloud_wc_cbar = fig.colorbar(cloud_wc_pcm[ll], ax = axs[0,ll])
                rel_cbar = fig.colorbar(rel_pcm[ll], ax = axs[1,ll])
                dei_cbar = fig.colorbar(dei_pcm[ll], ax = axs[2,ll])

            # Labels
            fig.suptitle("RTE-RRTMGP-CPP Atmospheric Radiative Transfer")
            fig.supxlabel(r"y $\left[ km \right]$")
            fig.supylabel(r"z $\left[ km \right]$")

            for ll in range(0, ncols):
                col_title: str = (r"{:.2f} Hours - ".format(mnn_times[jj,ll])
                    + r"Solar Zenith Angle {:.1f}$^{{\circ}}$ - ".format(mnn_szas[jj,ll])
                    + r"$x$ = {:.2f} $\left[ km \right]$".format(yz_slices_x[ll]))
                axs[0,ll].set_title(col_title)

            cloud_wc_cbar.ax.set_ylabel(r"Cloud Water Content $\left[ g\,m^{-3} \right]$")          
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

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_atm_day_{}.{}.png".format(jj, lr_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()