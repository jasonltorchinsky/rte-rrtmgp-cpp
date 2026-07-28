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
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_FIGURE, MPL_AXES
from consts.numeric import NP_LARGE
from consts.visual import plot_colors, plot_markers
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_sw_flux_abs, calc_sw_flux_sfc_dn, calc_sw_reflectance, \
    calc_z_max_info, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-rad-tran-distributions"
prog_desc: str = "Visualize distributions of two-stream and ray-tracer solver differences for RTE-RRTMGP-CPP."

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
    parser.add_argument("--rad-tran-outdir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT combined output directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

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
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(rad_tran_indir,
        rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    #---------------------------------------------------------------------------
    # Calculate relevant quantities at each resolution for each day
    #---------------------------------------------------------------------------
    reflectance_rt: dict = {}
    reflectance_ts: dict = {}
    reflectance_diff: dict = {}
    flux_sfc_dn_rt: dict = {}
    flux_sfc_dn_ts: dict = {}
    flux_sfc_dn_diff: dict = {}
    heating_rt: dict = {}
    heating_ts: dict = {}
    heating_diff: dict = {}

    #---------------------------------------------------------------------------
    # Loop through resolutions
    #---------------------------------------------------------------------------
    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(coarse_factor_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain z_max information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining maximum altitude information..."
        print_msg(msg)

        z_max_info: dict = calc_z_max_info(rad_tran_infile, z_max = z_max)

        #-------------------------------------------------------------------
        # Calculate reflectance
        #-------------------------------------------------------------------
        msg: str = "Calculating reflectance..."
        print_msg(msg)

        reflectance_rt[coarse_factor_str] = calc_sw_reflectance(
            rad_tran_infile,
            rad_tran_outfile,
            solver = "rt") # Shortwave reflectance, ray-tracer; [N/A]; [time, y, x]

        reflectance_ts[coarse_factor_str] = calc_sw_reflectance(
            rad_tran_infile,
            rad_tran_outfile,
            solver = "ts") # Shortwave reflectance, two-stream; [N/A]; [time, y, x]

        reflectance_diff[coarse_factor_str] = (
            reflectance_rt[coarse_factor_str] 
            - reflectance_ts[coarse_factor_str])

        #-------------------------------------------------------------------
        # Calculate downwelling surface flux
        #-------------------------------------------------------------------
        msg: str = "Calculating downwelling surface fluxes..."
        print_msg(msg)

        flux_sfc_dn_rt[coarse_factor_str] = calc_sw_flux_sfc_dn(
            rad_tran_outfile,
            solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_ts[coarse_factor_str] = calc_sw_flux_sfc_dn(
            rad_tran_outfile,
            solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_diff[coarse_factor_str] = (
            flux_sfc_dn_rt[coarse_factor_str] 
            - flux_sfc_dn_ts[coarse_factor_str])

        #-------------------------------------------------------------------
        # Calculate heating rates
        #-------------------------------------------------------------------
        msg: str = "Calculating heating rates..."
        print_msg(msg)

        heating_rt[coarse_factor_str] = calc_sw_heating(
            rad_tran_infile,
            rad_tran_outfile,
            z_max_info = z_max_info,
            solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

        heating_ts[coarse_factor_str] = calc_sw_heating(
            rad_tran_infile,
            rad_tran_outfile,
            z_max_info = z_max_info,
            solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

        heating_diff[coarse_factor_str] = (
            heating_rt[coarse_factor_str] 
            - heating_ts[coarse_factor_str])

    #-----------------------------------------------------------------------
    # Set up figure for plotting
    #-----------------------------------------------------------------------
    msg: str = "Setting up figure..."
    print_msg(msg)

    nrows: NP_INT = NP_INT(1)
    ncols: NP_INT = NP_INT(3)
    fig_width: NP_REAL = NP_REAL(6.5)
    fig_height: NP_REAL = (NP_REAL(nrows) / NP_REAL(ncols)) * fig_width
    fig_size: list[NP_REAL] = [fig_width, fig_height]
    fig: MPL_FIGURE
    axs: MPL_AXES
    fig, axs = plt.subplots(
        nrows = nrows, ncols = ncols,
        sharex = False, sharey = False,
        constrained_layout = True,
        figsize = fig_size)

    if ncols == 1:
        axs = axs[...,None]
    elif nrows == 1:
        axs = axs[None,...]

    #-------------------------------------------------------------------
    # Obtain quantities common across plots
    #-------------------------------------------------------------------
    hres_str_list: list[str] = []
    coarse_factor_str: str
    for coarse_factor_str in flux_sfc_dn_rt.keys():
        x: XR_DATAARRAY = flux_sfc_dn_rt[coarse_factor_str]["x"] # [n_x]; [m]
        dx: NP_REAL = NP_REAL(x[1] - x[0]) # [m]
        if dx < 1.0e3:
            hres_str_list += [r"{:.0f} $m$".format(dx)]
        else:
            hres_str_list += [r"{:.2f} $km$".format(dx * 1.e-3)]
    time: XR_DATAARRAY = flux_sfc_dn_rt[coarse_factor_str]["time"] # [time]; [h]

    #-------------------------------------------------------------------
    # Plot distributions at each resolution
    #-------------------------------------------------------------------
    msg: str = "Plotting distributions..."
    print_msg(msg)

    nbins: NP_INT = NP_INT(64)
    nplot_colors: NP_INT = NP_INT(len(plot_colors))

    # Column 0 - Reflectance, Ray-Tracer - Two-Stream
    ii: int
    for ii in range(0, coarse_factors.size):
        coarse_factor: NP_INT = coarse_factors[ii]
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

        axs[0,0].hist(
            NP_REAL(reflectance_diff[coarse_factor_str].to_numpy().flatten()),
            bins = nbins,
            density = True,
            histtype = "step",
            linewidth = 1.0,
            color = plot_colors[ii % nplot_colors],
            label = hres_str_list[ii],
            zorder = -ii)
    
    # Column 1 - Downwelling Surface Flux, Ray-Tracer - Two-Stream
    ii: int
    for ii in range(0, coarse_factors.size):
        coarse_factor: NP_INT = coarse_factors[ii]
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

        axs[0,1].hist(
            NP_REAL(flux_sfc_dn_diff[coarse_factor_str].to_numpy().flatten()),
            bins = nbins,
            density = True,
            histtype = "step",
            linewidth = 1.0,
            color = plot_colors[ii % nplot_colors],
            label = hres_str_list[ii],
            zorder = -ii)
    
    # Column 2 - Heating, Ray-Tracer - Two-Stream
    ii: int
    for ii in range(0, coarse_factors.size):
        coarse_factor: NP_INT = coarse_factors[ii]
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

        axs[0,2].hist(
            NP_REAL(heating_diff[coarse_factor_str].to_numpy().flatten()),
            bins = nbins,
            density = True,
            histtype = "step",
            linewidth = 1.0,
            color = plot_colors[ii % nplot_colors],
            label = hres_str_list[ii],
            zorder = -ii)

    #---------------------------------------------------------------------------
    # Add plot elements
    #---------------------------------------------------------------------------
    fig.suptitle(r"$\left(\text{Ray-Tracer - Two-Stream}\right)$ Distribution")
    fig.supylabel("Probability Density")

    axs[0,0].set_xlabel(r"Reflectance")
    axs[0,1].set_xlabel(r"Absorbed Flux $\left[ W\,m^{-3} \right]$")
    axs[0,2].set_xlabel(r"Heating Rate $\left[ K\,d^{-1} \right]$")

    axs[0,0].legend(loc = "lower right")

    for ax in axs[...].flatten():
        ax.axvline(0,
            color = "gray", 
            linestyle = "solid", 
            linewidth = 0.5,
            zorder = -32)

    #---------------------------------------------------------------------------
    # Adjust vertical scales
    #---------------------------------------------------------------------------
    for ax in axs[...].flatten():
        ax.set_yscale("log")

    #---------------------------------------------------------------------------
    # Save the plot to file
    #---------------------------------------------------------------------------
    plt_filename = "rte_rrtmgp_cpp_error_distribution.png"
    plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
    fig.savefig(plt_filepath, dpi = 512)
    plt.close(fig)

if __name__ == "__main__":
    main()