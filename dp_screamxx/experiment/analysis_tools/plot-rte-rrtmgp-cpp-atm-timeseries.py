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
from scipy import ndimage

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_PCOLORMESH
from consts.visual import flux_cmap, heating_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, \
    calc_cloud_wc

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-rad-tran-snapshot"
prog_desc: str = "Visualize absorbed shortwave flux and atmospheric heating rates for RTE-RRTMGP-CPP."

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

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, coarse_factor_str)
        print(msg, flush = True)

        #-----------------------------------------------------------------------
        # Obtain daytime indices, times, SZAs
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Obtaining daytime information...".format(current_time)
        print(msg, flush = True)

        daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
        ndays: NP_INT = NP_INT(daytime_indices.shape[0])

        #-----------------------------------------------------------------------
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Setting up figure...".format(current_time)
        print(msg, flush = True)

        nrows: NP_INT = NP_INT(4)
        ncols: NP_INT = NP_INT(ndays)
        fig_height: NP_REAL = NP_REAL(6.)
        fig_base_size = np.array([(ncols / nrows) * fig_height, fig_height])
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = 3. * fig_base_size)

        if len(axs.shape) == 1:
            axs = axs[...,None]

        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate cloud water content
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating cloud water content for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, daytime_indices[jj,...], zmax = zmax) # Cloud water content; [g m^{-3}]; [ntime, lay, y, x]

            #-------------------------------------------------------------------
            # Calculate cloud quantities
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating cloud water statistics for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            # Obtain grid information
            x: XR_DATAARRAY = cloud_wc["x"] # x-coordinate of column-midpoints; [n_x]; [m]
            y: XR_DATAARRAY = cloud_wc["y"] # y-coordinate of column-midpoints; [n_y]; [m]
            lay: XR_DATAARRAY = cloud_wc["lay"] # z-coordinate of layer-midpoints; [n_lay]; [m]

            dx: NP_REAL = NP_REAL(x[1] - x[0]) # x-dimension spacing; [m]; # ASSUME UNIFORM
            dy: NP_REAL = NP_REAL(y[1] - y[0]) # y-dimension spacing; [m]; # ASSUME UNIFORM
            dz: NP_REAL = NP_REAL(lay[1] - lay[0]) # z-dimension spacing; [m]; # ASSUME UNIFORM

            # Thresholds
            cloud_threshold: NP_REAL = NP_REAL(1.0e-3) # [g m^{-3}]

            # Total, column, and layer cloud mass
            total_mass_cloud: XR_DATAARRAY = (dx * dy * dz) * cloud_wc.sum(dim = ["lay", "y", "x"]) * 1.e-3 # [ntime]; [g] => [kg]
            column_mass_cloud: XR_DATAARRAY = (dx * dy * dz) * cloud_wc.sum(dim = ["lay"]) * 1.e-3 # [ntime]; [g] => [kg]
            layer_mass_cloud: XR_DATAARRAY = (dx * dy * dz) * cloud_wc.sum(dim = ["y", "x"]) * 1.e-3 # [ntime]; [g] => [kg]
            
            # Number of clouds
            n_clouds: XR_DATAARRAY = xr.apply_ufunc(
                count_connected_components_3d,
                cloud_wc,
                input_core_dims = [["lay", "y", "x"]],
                output_core_dims = [[]],
                vectorize = True,
                kwargs = {
                    "threshold": cloud_threshold,
                    "connectivity": 3,
                },
                output_dtypes = [NP_INT],
            )

            #-------------------------------------------------------------------
            # Obtain statistics for cloud quantities
            #-------------------------------------------------------------------
            quantiles: NP_ARRAY[NP_REAL] = np.array([0.95, 0.8, 0.6, 0.5, 0.4, 0.2, 0.05], dtype = NP_REAL)
            nquantiles: NP_INT = NP_INT(quantiles.size)

            column_mass_cloud_stats: list[XR_DATAARRAY] = [column_mass_cloud.quantile(qq, dim = ["y", "x"]) for qq in quantiles]
            layer_mass_cloud_stats: list[XR_DATAARRAY] = [layer_mass_cloud.quantile(qq, dim = ["lay"]) for qq in quantiles]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting data...".format(current_time)
            print(msg, flush = True)

            # Row 0: Total Cloud Mass
            row: NP_INT = NP_INT(0)
            axs[row,jj].plot(total_mass_cloud["time"], total_mass_cloud, color = "black")
            axs[row,jj].set_yscale("log")

            # Row 1: Column Cloud Mass
            row: NP_INT = NP_INT(1)
            median_index: NP_INT = NP_INT(nquantiles // 2)
            axs[row,jj].plot(column_mass_cloud_stats[median_index]["time"],
                column_mass_cloud_stats[median_index],
                color = "blue")

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(column_mass_cloud_stats[ll]["time"], 
                    column_mass_cloud_stats[ll],
                    column_mass_cloud_stats[(nquantiles - 1) - ll],
                    color = "blue", edgecolor = None, alpha = 0.15)
            axs[row,jj].set_yscale("log")

            # Row 2: Layer Cloud Mass
            row: NP_INT = NP_INT(2)
            median_index: NP_INT = NP_INT(nquantiles // 2)
            axs[row,jj].plot(layer_mass_cloud_stats[median_index]["time"],
                layer_mass_cloud_stats[median_index],
                color = "blue")

            ll: int
            for ll in range(0, median_index):
                axs[row,jj].fill_between(layer_mass_cloud_stats[ll]["time"], 
                    layer_mass_cloud_stats[ll],
                    layer_mass_cloud_stats[(nquantiles - 1) - ll],
                    color = "blue", edgecolor = None, alpha = 0.15)
            axs[row,jj].set_yscale("log")

            # Row 3: Number of Clouds
            row: NP_INT = NP_INT(3)
            axs[row,jj].step(n_clouds["time"], n_clouds, where = "post", color = "black")
            axs[row,jj].set_ylim(ymin = 0)

            # Common column-wise plot elements
            # x-ticks
            time: XR_DATARRAY = cloud_wc["time"]
            xlim: list[NP_REAL] = np.array([time.min(), time.max()], dtype = NP_REAL)
            xticks: NP_ARRAY[NP_REAL] = np.array([xlim[0], (xlim[0] + xlim[1]) / 2., xlim[1]], dtype = NP_REAL)
            ll: int
            for ll in range(0, nrows):
                axs[ll,jj].set_xticks(xticks)
                axs[ll,jj].axvline(xticks[1], color = "gray", linestyle = "solid", linewidth = 0.5)

        # Labels
        fig.suptitle("RTE-RRTMGP-CPP Atmosphere Time Series")
        fig.supxlabel(r"Time $\left[ h \right]$")

        for jj in range(0, ndays):
            col_title: str = "Day {}".format(jj)
            axs[0,jj].set_title(col_title)
        axs[0,0].set_ylabel(r"Total Cloud Water Mass $\left[ kg \right]$")
        axs[1,0].set_ylabel(r"Column Cloud Water Mass $\left[ kg \right]$")
        axs[2,0].set_ylabel(r"Layer Cloud Water Mass $\left[ kg \right]$")
        axs[3,0].set_ylabel(r"Number of Clouds")

        #-------------------------------------------------------------------
        # Save the plot to file
        #-------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_atm_timeseries.{}.png".format(coarse_factor_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

def count_connected_components_3d(field_3d: XR_DATAARRAY,
    threshold: NP_REAL = NP_REAL(1.0e-3), connectivity = 3) -> NP_INT:
    field_mask: XR_DATAARRAY = field_3d > threshold
    structure: NP_ARRAY[NP_BOOL] = ndimage.generate_binary_structure(rank = 3, connectivity = connectivity)
    n_components: NP_INT
    n_components = NP_INT(ndimage.label(field_mask, structure = structure)[1])

    return n_components

if __name__ == "__main__":
    main()