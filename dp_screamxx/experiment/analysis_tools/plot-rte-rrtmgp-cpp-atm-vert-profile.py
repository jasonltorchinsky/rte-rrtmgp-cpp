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
from consts.visual import plot_colors
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, \
    calc_cloud_top, calc_cloud_wc, calc_t, calc_vmr, calc_tropopause, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-atm-vert-profile"
prog_desc: str = "Visualize atmospheric vertical profile for RTE-RRTMGP-CPP."

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
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate

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
        # Calculate vertical profiles for desired atmospheric quantities
        #-----------------------------------------------------------------------
        msg: str = "Calculating vertical profiles for desired atmospheric quantities..."
        print_msg(msg)

        t_lev: XR_DATAARRAY = calc_t(rad_tran_infile) # Temperature at layer interfaces; [K]; [time, lev, y, x]
        t_vert_profile: XR_DATAARRAY = t_lev.mean(dim = ["time", "y", "x"]) # Horizontal mean temperature at layer interfaces; [K]; [lev]
        
        vmr: XR_DATAARRAY = calc_vmr(rad_tran_infile) # Trace gas molar dry mixing ration; [mol mol^{-1}]; [time, lay, y, x]
        vmr_vert_profile: XR_DATAARRAY = vmr.mean(dim = ["time", "y", "x"]) # Horizontal mean trace gas molar dry mixing ratio; [mol mol^{-1}]; [lay]

        cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile) # Clouc water content; [g m^{-3}]; [time, lay, y, x]
        cloud_wc_vert_profile: XR_DATAARRAY = cloud_wc.mean(dim = ["time", "y", "x"]) # Horizontal mean cloud water content; [g m^{-3}]; [lay]

        z_tropopause: NP_REAL = calc_tropopause(rad_tran_infile) # Tropopause height; [km]
        z_cloud_top: NP_REAL = calc_cloud_top(rad_tran_infile) # Cloud top height; [km]

        #-----------------------------------------------------------------------
        # Plot the data
        #-----------------------------------------------------------------------
        msg: str = "Plotting data..."
        print_msg(msg)

        nrows: NP_INT = NP_INT(1)
        ncols: NP_INT = NP_INT(3)
        fig_height: NP_REAL = NP_REAL(3.)
        fig_base_size = np.array([fig_height, fig_height])
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = False, sharey = True,
            constrained_layout = True,
            figsize = 3. * fig_base_size)

        # Plot 0: Temperature
        axs[0].plot(t_vert_profile, grid["zh"] * 1.e-3, color = "black")
        axs[0].axhline(z_tropopause, color = "black", linewidth = 1.0)
        axs[0].axhline(z_cloud_top, color = "red", linewidth = 1.0)

        # Plot 1: Volume Mixing Ratios
        ncolors: NP_INT = NP_INT(len(plot_colors))
        vmr_keys: list[str] = list(vmr_vert_profile.keys())
        ll: int
        for ll in range(0, len(vmr_keys)):
            vmr_key: str = vmr_keys[ll]
            axs[1].plot(vmr_vert_profile[vmr_key], grid["z"] * 1.e-3,
                color = plot_colors[ll%ncolors],
                linewidth = 2.0, label = vmr_key)
        axs[1].axhline(z_tropopause, color = "black", linewidth = 1.0)
        axs[1].axhline(z_cloud_top, color = "red", linewidth = 1.0)

        # Plot 1: Cloud Water Content
        axs[2].plot(cloud_wc_vert_profile, grid["z"] * 1.e-3, color = "black")
        axs[2].axhline(z_tropopause, color = "black", linewidth = 1.0)
        axs[2].axhline(z_cloud_top, color = "red", linewidth = 1.0)

        # Labels
        dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.1f} $km$".format(dx * 1.e-3)

        fig.suptitle(r"RTE-RRTMGP-CPP Atmospheric Vertical Profiles - {}".format(dx_str))
        fig.supylabel(r"z $\left[ km \right]$")

        axs[0].set_xlabel(r"Temperature $\left[ K \right ]$")
        axs[1].set_xlabel(r"Mixing Ratio $\left[ mol\,mol^{-1} \right ]$")
        axs[2].set_xlabel(r"Cloud Water Content $\left[ g\,m^{-3} \right ]$")

        # Set axis scales
        axs[1].set_xscale("log")

        # Set legends
        axs[1].legend()

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_atm_vert_profile.{}.png".format(lr_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()