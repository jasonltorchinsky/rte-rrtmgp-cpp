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
from consts.visual import cw_cmap, plot_colors
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_vmr

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-atm-snapshot"
prog_desc: str = "Visualize atmospheric state for RTE-RRTMGP-CPP."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time: str = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line input...".format(current_time)
    print(msg, flush = True)

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

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, lr_str)
        print(msg, flush = True)

        #-----------------------------------------------------------------------
        # Obtain Morning-Noon-Night time indices, times, SZAs, zmax index
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Obtaining morning-noon-night information...".format(current_time)
        print(msg, flush = True)

        mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
        mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = mnn_indices.shape[0]

        #-----------------------------------------------------------------------
        # Calculate fields for each MNN of each day
        #-----------------------------------------------------------------------
        int: jj
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate cloud water content, x-indices for yz-slices for calculations
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating cloud water content for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj]) # Cloud water content; [g m^{-3}]; [3, lev, y, x]
            x_indices: NP_ARRAY[NP_INT] = np.array([np.unravel_index(np.argmax(cloud_wc.isel(time = ll).to_numpy()), cloud_wc.shape[1:])[2] for ll in range(3)]) # [3]

            # Sneak in getting grid information
            z: NP_ARRAY[NP_REAL] = cloud_wc["lay"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]

            #-------------------------------------------------------------------
            # Calculate volume mixing ratios
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating volume mixing ratios for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            vmr: dict = calc_vmr(rad_tran_infile, mnn_indices[jj], x_indices) # Volume mixing ratios; [kg kg^{-1}]; 3 * [lay, y] for each VMR

            #-------------------------------------------------------------------
            # Trim fields to the yz-slice
            #-------------------------------------------------------------------
            vmr_z: dict = {}
            for key in vmr.keys():
                vmr_z[key] = [np.mean(vmr[key][ll], axis = 1) for ll in range(0, 3)] # Averaged over the y-range

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting data...".format(current_time)
            print(msg, flush = True)

            nrows: NP_INT = NP_INT(1)
            fig_height: NP_REAL = NP_REAL(3.)
            fig_base_size = np.array([(y_window_width / zmax) * fig_height, fig_height])
            fig, axs = plt.subplots(nrows = nrows, ncols = 3,
                sharey = True,
                constrained_layout = True,
                figsize = 3. * fig_base_size)

            if nrows == 1:
                axs = axs[None,...]

            # Row 0: Volume Mixing Ratios
            ncolors: NP_INT = NP_INT(len(plot_colors))
            vmr_keys: list[str] = list(vmr_z.keys())
            vmr_z_plot: dict = {}
            ll: int
            for ll in range(0, len(vmr_keys)):
                key: str = vmr_keys[ll]
                vmr_z_plot[key] = [[] for _ in range(0, 3)]
                mm: int
                for mm in range(0, 3):
                    vmr_z_plot[key][mm] = axs[0, mm].plot(vmr_z[key][mm], z,
                        color = plot_colors[ll%ncolors], linewidth = 2.0, label = key,)

            # Labels
            fig.suptitle("RTE-RRTMGP-CPP Volume Mixing Ratios")
            fig.supxlabel(r"Volume Mixing Ratio $\left[ mol\,mol^{-1} \right]$")
            fig.supylabel(r"z $\left[ km \right]$")

            for ll in range(3):
                col_title: str = (r"{:.2f} Hours - ".format(mnn_times[jj,ll])
                    + r"Solar Zenith Angle {:.1f}$^{{\circ}}$ - ".format(mnn_szas[jj,ll])
                    + r"$x$ = {:.2f} $\left[ km \right]$".format(yz_slices_x[ll]))
                axs[0,ll].set_title(col_title)

            # Set legends
            axs[0,0].legend()

            # Set vertical limits
            ll: int
            for ll in range(0, 3):
                axs[0,ll].set_ylim((0.0, zmax))

            # Set horizontal scales
            ll: int
            for ll in range(0, 3):
                axs[0,ll].set_xscale("log")

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_vmr_day_{}.{}.png".format(jj, lr_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()