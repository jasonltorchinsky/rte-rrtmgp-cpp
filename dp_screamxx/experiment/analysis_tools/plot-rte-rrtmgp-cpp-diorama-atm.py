#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
import re
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    XR_DATAARRAY, \
    MPL_AXES, MPL_FIGURE, MPL_LINEAR_SEGMENTED_COLORMAP, MPL_LOGNORM, MPL_PCOLORMESH, \
    MPL_COLORBAR
from consts.visual import cloud_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_z_max_info, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-diorama-atm"
prog_desc: str = "Create a diorama of the atmopsheric state for RTE-RRTMGP-CPP."

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
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
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
    [rad_tran_infiles, _] = find_inout_pairs(rad_tran_indir, None, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    #-----------------------------------------------------------------------
    # Obtain Morning-Noon-Night time indices, times, SZAs, z_max_info
    #-----------------------------------------------------------------------
    msg: str = "Obtaining time index and z-max info..."
    print_msg(msg)

    mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(
        rad_tran_infiles[0]
    ) # [ndays, 3]
    mnn_times: NP_ARRAY[NP_REAL] = find_times(
        rad_tran_infiles[0], 
        mnn_indices) # Time since simulation start; [h]; [ndays, 3]
    ndays: NP_INT = NP_INT(mnn_indices.shape[0])
    z_max_info: dict = calc_z_max_info(
        rad_tran_infiles[0],
        z_max = z_max)

    time_indices: NP_ARRAY[NP_INT] = mnn_indices[:,0:2].flatten()
    times: NP_ARRAY[NP_REAL] = mnn_times[:,0:2].flatten()

    n_t: NP_INT = NP_INT(time_indices.size)

    ii: int
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

        # Rescale grids to have correct units
        xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
        yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]
        zh: XR_DATAARRAY = (grid["zh"]
            .sel(zh = z_max_info["sel_indexers"]["zh"])) * 1.e-3 # [m] => [km]

        # Get number of grid points
        n_xh: NP_INT = NP_INT(xh.size)
        n_yh: NP_INT = NP_INT(yh.size)
        n_zh: NP_INT = NP_INT(zh.size)

        n_x: NP_INT = n_xh - 1
        n_y: NP_INT = n_yh - 1
        n_z: NP_INT = n_zh - 1

        # Combine into meshgrid for voxel plot
        xx: NP_ARRAY[NP_REAL]
        yy: NP_ARRAY[NP_REAL]
        zz: NP_ARRAY[NP_REAL]
        xx, yy, zz = np.meshgrid(
            NP_REAL(xh.to_numpy()),
            NP_REAL(yh.to_numpy()),
            NP_REAL(zh.to_numpy()),
            indexing = "ij"
        )
        
        #-----------------------------------------------------------------------
        # Calculate CWC
        #-----------------------------------------------------------------------
        msg: str = "Obtaining cloud water content info..."
        print_msg(msg)

        cwc: XR_DATAARRAY = calc_cloud_wc(
            rad_tran_infile, 
            time_indices = time_indices,
            z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [time, lay, y, x]

        # Discard values less than the tolerance
        cwc_tol: NP_REAL = NP_REAL(1.e-2)
        cwc = xr.where(cwc < cwc_tol, NP_REAL(0.), cwc)

        #-----------------------------------------------------------------------
        # Calculate the filled and facecolors for each voxel
        #-----------------------------------------------------------------------
        max_cwc: NP_REAL = NP_REAL(cwc.max())
        min_cwc: NP_REAL = cwc_tol

        cwc_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[cloud_cmap]
        cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_cwc, vmax = max_cwc)
        cwc_norm: NP_ARRAY[NP_REAL] = cwc_colormap_norm(
            NP_REAL(
                cwc
                .to_numpy()
                ).flatten()
                ).reshape([n_t, n_z, n_y, n_x])
        
        filled: NP_ARRAY[NP_BOOL] = NP_BOOL((cwc >= cwc_tol).to_numpy()) # [n_t, n_z, n_y, n_x]

        # Transpose to necessary shape
        filled: NP_ARRAY[NP_BOOL] = np.transpose(filled, axes = (0, 3, 2, 1)) # [n_t, n_x, n_y, n_z]
        cwc_norm: NP_ARRAY[NP_REAL] = np.transpose(cwc_norm, axes = (0, 3, 2, 1)) # [n_t, n_x, n_y, n_z]
        facecolors: NP_ARRAY[NP_REAL] = cwc_colormap(cwc_norm)

        # Set alpha
        cwc_min: NP_REAL = cwc.min()
        cwc_max: NP_REAL = cwc.max()
        alpha_min: NP_REAL = NP_REAL(0.1)
        alpha_max: NP_REAL = NP_REAL(0.7)
        alpha: NP_ARRAY[NP_REAL] = NP_REAL(
            (((alpha_max - alpha_min) * (cwc - cwc_min) / (cwc_max - cwc_min)) + alpha_min).to_numpy()) # [n_t, n_z, n_y, n_x]
        alpha: NP_ARRAY[NP_REAL] = np.transpose(alpha, axes = (0, 3, 2, 1)) # [n_t, n_x, n_y, n_z]
        facecolors[...,3] = alpha
        
        #-----------------------------------------------------------------------
        # Set up the figure
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        nrows: NP_INT = NP_INT(2)
        ncols: NP_INT = NP_INT(ndays)
        fig_height: NP_REAL = NP_REAL(5.5)
        fig_width: NP_REAL = NP_REAL(6.5)
        fig: MPL_FIGURE
        axs: NP_ARRAY[MPL_AXES]
        fig, axs = plt.subplots(
            nrows = nrows, ncols = ncols,
            sharex = False, sharey = False,
            constrained_layout = True,
            figsize = (fig_width, fig_height),
            subplot_kw = {"projection" : "3d"})
        
        if ncols == 1:
            axs = axs[...,None]
        elif nrows == 1:
            axs = axs[None,...]

        #-----------------------------------------------------------------------
        # Plot the data
        #-----------------------------------------------------------------------
        msg: str = "Plotting the data..."
        print_msg(msg)

        jj: int
        kk: int
        for jj in range(0, nrows):
            for kk in range(0, ncols):
                index: int = 2 * jj + kk

                axs[jj, kk].voxels(
                    xx,
                    yy,
                    zz,
                    filled[index,...],
                    facecolors = facecolors[index,...],
                    edgecolor = "none"
                )

        #-----------------------------------------------------------------------
        # Set up colorbar
        #-----------------------------------------------------------------------
        msg: str = "Setting up colorbar..."
        print_msg(msg)

        cwc_colorbar: MPL_COLORBAR = fig.colorbar(
            mappable = mpl.colorizer.ColorizingArtist(
                mpl.colorizer.Colorizer(
                    norm = cwc_colormap_norm,
                    cmap = cwc_colormap)
            ), 
            ax = axs,
            pad = 0.2)
        cwc_colorbar.ax.set_yscale("log")

        #-----------------------------------------------------------------------
        # Set style elements
        #-----------------------------------------------------------------------
        msg: str = "Setting style elements..."
        print_msg(msg)

        # Background Panes
        pane_color: list[float] = [0.0, 0.0, 0.0, 0.0]
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.xaxis.set_pane_color(pane_color)
            ax.yaxis.set_pane_color(pane_color)
            ax.zaxis.set_pane_color(pane_color)

        # Set grid linewidth
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.xaxis._axinfo["grid"]["linewidth"] = 0
            ax.yaxis._axinfo["grid"]["linewidth"] = 0
            ax.zaxis._axinfo["grid"]["linewidth"] = 0


        # Aspect Ratio
        xh_len: NP_REAL = xh.max() - xh.min()
        yh_len: NP_REAL = yh.max() - yh.min()
        zh_len: NP_REAL = zh.max() - zh.min()
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

        # Diorama Time Labels
        jj: int
        kk: int
        for jj in range(0, nrows):
            for kk in range(0, ncols):
                index: int = 2 * jj + kk
                axs[jj,kk].set_title(r"{} $h$".format(times[index]))

        # Tick Labels - Get rid of unnecessary ones and keep them uniform 
        # across all plots
        x_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(xh.min(), xh.max()))
        y_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(yh.min(), yh.max()))
        z_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 2).tick_values(0., np.floor(zh.max())))
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.xaxis.set_ticks(x_ticks)
            ax.yaxis.set_ticks(y_ticks)
            ax.zaxis.set_ticks(z_ticks)
        for ax in (axs[:-1,-1]).flatten():
            ax.xaxis.set_tick_params(labelcolor = "none")
            ax.yaxis.set_tick_params(labelcolor = "none")
        for ax in (axs[-1,:-1]).flatten():
            ax.yaxis.set_tick_params(labelcolor = "none")
            ax.zaxis.set_tick_params(labelcolor = "none")
        for ax in (axs[:-1,:-1]).flatten():
            ax.xaxis.set_tick_params(labelcolor = "none")
            ax.yaxis.set_tick_params(labelcolor = "none")
            ax.zaxis.set_tick_params(labelcolor = "none")

        # Axis labels
        axs[-1,-1].set_ylabel(r"y $\left[ km \right]$")
        for ax in (axs[:,-1]).flatten():
            ax.set_zlabel(r"z $\left[ km \right]$")
        for ax in (axs[-1,:]).flatten():
            ax.set_xlabel(r"x $\left[ km \right]$")

        # Suplabels
        dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.1f} $km$".format(dx * 1.e-3)

        fig.suptitle(r"Cloud Water Content $\left[ g\,m^{-3} \right]$" + " - {}".format(dx_str))
        fig.supxlabel("") # Add for padding at bottom

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        msg: str = "Saving plot to file..."
        print_msg(msg)

        plt_filename = "rte_rrtmgp_cpp_diorama_atm.{}.png".format(lr_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 200)
        plt.close(fig)

if __name__ == "__main__":
    main()