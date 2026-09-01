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
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    XR_DATAARRAY, \
    MPL_AXES, MPL_FIGURE, MPL_LINEAR_SEGMENTED_COLORMAP, MPL_LOGNORM, \
    MPL_COLORBAR
from consts.visual import cloud_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_times, \
    calc_cloud_wc, calc_z_max_info, find_grid, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-atm-snapshot"
prog_desc: str = "Visualize atmospheric state for RTE-RRTMGP-CPP."


def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    msg: str = "Parsing command-line input..."
    print_msg(msg)

    parser: ArgumentParser = ArgumentParser(
        prog = prog_name,
        description = prog_desc
    )
    parser.add_argument(
        "--rad-tran-indir",
        action = "store",
        nargs = "?",
        type = str,
        required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory."
    )
    parser.add_argument(
        "--rad-tran-vizdir",
        nargs = "?",
        required = True,
        type = str,
        help = "Radiative Transfer visualization file directory."
    )
    parser.add_argument(
        "--working-dir",
        nargs = "?",
        default = ".working",
        type = str,
        help = "Working directory to output calculated values."
    )
    parser.add_argument(
        "--recalculate",
        nargs = "?",
        default = False,
        type = bool,
        help = "Re-calculate surface heating rates."
    )
    parser.add_argument(
        "--z-max",
        nargs = "?",
        default = 0.,
        type = float,
        help = "Maximum height for calculations [km]."
    )
    parser.add_argument(
        "--coarse-factors",
        action = "store",
        nargs = "?",
        type = str,
        required = False,
        default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64."
    )

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

    coarse_factors: Optional[NP_ARRAY[NP_INT]] = None
    if args.coarse_factors is not None:
        coarse_factors = np.sort(
            np.array(args.coarse_factors.split(","), dtype = NP_INT)
        )[::-1]

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
    [rad_tran_infiles, _] = find_inout_pairs(
        rad_tran_indir,
        None,
        coarse_factors
    )

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]

        lr_match: Optional[re.Match] = lr_re.search(rad_tran_infile)
        if lr_match is None:
            raise ValueError("Could not find low-resolution string matching 'lr_..' in {}".format(rad_tran_infile))

        lr_str: str = lr_match.group()

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain Morning-Noon-Night time indices, times, z_max_info
        #-----------------------------------------------------------------------
        msg: str = "Obtaining time index and z-max info..."
        print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(
            rad_tran_infile
        ) # [ndays, 3]

        daytime_times: NP_ARRAY[NP_REAL] = find_times(
            rad_tran_infile,
            daytime_indices
        ) # Time since simulation start; [h]; [ndays, 3]

        ndays: NP_INT = NP_INT(daytime_indices.shape[0])

        z_max_info: dict = calc_z_max_info(
            rad_tran_infile,
            z_max = z_max
        )

        n_t: NP_INT = NP_INT(daytime_indices.size)
        n_t_perday: NP_INT = NP_INT(daytime_indices.shape[1])

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)

        grid: dict = find_grid(rad_tran_infile)

        # Rescale grids to have correct units
        xh: XR_DATAARRAY = grid["xh"] * 1.e-3 # [m] => [km]
        yh: XR_DATAARRAY = grid["yh"] * 1.e-3 # [m] => [km]
        zh: XR_DATAARRAY = (
            grid["zh"]
            .sel(zh = z_max_info["sel_indexers"]["zh"])
        ) * 1.e-3 # [m] => [km]

        # Convert to NumPy once
        xh_np: NP_ARRAY[NP_REAL] = NP_REAL(xh.to_numpy())
        yh_np: NP_ARRAY[NP_REAL] = NP_REAL(yh.to_numpy())
        zh_np: NP_ARRAY[NP_REAL] = NP_REAL(zh.to_numpy())

        # Get number of grid points
        n_xh: NP_INT = NP_INT(xh_np.size)
        n_yh: NP_INT = NP_INT(yh_np.size)
        n_zh: NP_INT = NP_INT(zh_np.size)

        n_x: NP_INT = n_xh - 1
        n_y: NP_INT = n_yh - 1
        n_z: NP_INT = n_zh - 1

        # Combine into meshgrid for voxel plot
        xx: NP_ARRAY[NP_REAL]
        yy: NP_ARRAY[NP_REAL]
        zz: NP_ARRAY[NP_REAL]
        xx, yy, zz = np.meshgrid(
            xh_np,
            yh_np,
            zh_np,
            indexing = "ij"
        )

        #-----------------------------------------------------------------------
        # Calculate CWC
        #-----------------------------------------------------------------------
        msg: str = "Obtaining cloud water content info..."
        print_msg(msg)

        cwc: XR_DATAARRAY = calc_cloud_wc(
            rad_tran_infile,
            time_indices = daytime_indices.flatten(),
            z_max_info = z_max_info
        ) # Cloud water content; [g m^{-3}]; [time, lay, y, x]

        # Discard values less than the tolerance
        cwc_tol: NP_REAL = NP_REAL(1.e-1)
        cwc = xr.where(cwc < cwc_tol, NP_REAL(0.), cwc)

        # Convert to NumPy once
        cwc_np: NP_ARRAY[NP_REAL] = NP_REAL(cwc.to_numpy())

        #-----------------------------------------------------------------------
        # Calculate the filled and facecolors for each voxel
        #-----------------------------------------------------------------------
        max_cwc: NP_REAL = NP_REAL(np.max(cwc_np))
        min_cwc: NP_REAL = cwc_tol

        cwc_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[cloud_cmap]
        cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(
            vmin = min_cwc,
            vmax = max_cwc
        )

        cwc_norm: NP_ARRAY[NP_REAL] = cwc_colormap_norm(
            cwc_np.flatten()
        ).reshape([n_t, n_z, n_y, n_x])

        filled: NP_ARRAY[NP_BOOL] = NP_BOOL(cwc_np >= cwc_tol)

        # Transpose to shape needed by matplotlib voxels:
        # [n_t, n_x, n_y, n_z]
        filled = np.transpose(filled, axes = (0, 3, 2, 1))
        cwc_norm = np.transpose(cwc_norm, axes = (0, 3, 2, 1))

        facecolors: NP_ARRAY[NP_REAL] = cwc_colormap(cwc_norm)

        # Set alpha
        cwc_min: NP_REAL = NP_REAL(np.min(cwc_np))
        cwc_max: NP_REAL = NP_REAL(np.max(cwc_np))

        alpha_min: NP_REAL = NP_REAL(0.1)
        alpha_max: NP_REAL = NP_REAL(0.6)

        if cwc_max > cwc_min:
            alpha: NP_ARRAY[NP_REAL] = NP_REAL(
                ((alpha_max - alpha_min) * (cwc_np - cwc_min) / (cwc_max - cwc_min)) + alpha_min
            )
        else:
            alpha: NP_ARRAY[NP_REAL] = NP_REAL(
                np.full_like(cwc_np, alpha_max)
            )

        alpha = np.transpose(alpha, axes = (0, 3, 2, 1))
        facecolors[..., 3] = alpha

        #-----------------------------------------------------------------------
        # Set up the figure
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        nrows: NP_INT = NP_INT(1)
        ncols: NP_INT = NP_INT(1)
        fig_height: NP_REAL = NP_REAL(3.5)
        fig_width: NP_REAL = NP_REAL(4.5)

        fig: MPL_FIGURE
        axs: NP_ARRAY[MPL_AXES]

        # Important animation fix:
        # Do not use constrained_layout = True for this animation. It can
        # recompute the layout every frame and cause the axes to visibly move.
        fig, axs = plt.subplots(
            nrows = nrows,
            ncols = ncols,
            sharex = False,
            sharey = False,
            constrained_layout = False,
            figsize = (fig_width, fig_height),
            subplot_kw = {"projection": "3d"}
        )

        if (ncols == 1) and (nrows == 1):
            axs = np.array([[axs]])
        elif ncols == 1:
            axs = axs[..., None]
        elif nrows == 1:
            axs = axs[None, ...]

        #-----------------------------------------------------------------------
        # Set up colorbar
        #-----------------------------------------------------------------------
        msg: str = "Setting up colorbar..."
        print_msg(msg)

        cwc_colorbar: MPL_COLORBAR = fig.colorbar(
            mappable = mpl.colorizer.ColorizingArtist(
                mpl.colorizer.Colorizer(
                    norm = cwc_colormap_norm,
                    cmap = cwc_colormap
                )
            ),
            ax = axs,
            pad = 0.1
        )
        cwc_colorbar.ax.set_yscale("log")

        #-----------------------------------------------------------------------
        # Set style elements
        #-----------------------------------------------------------------------
        msg: str = "Setting style elements..."
        print_msg(msg)

        # Background Panes
        pane_color: list[float] = [135. / 255, 206. / 255., 235. / 255., 1.0]
        for ax in axs.flatten():
            ax.xaxis.set_pane_color(pane_color)
            ax.yaxis.set_pane_color(pane_color)
            ax.zaxis.set_pane_color(pane_color)

        # Fixed limits
        xlim: tuple[float, float] = (float(np.min(xh_np)), float(np.max(xh_np)))
        ylim: tuple[float, float] = (float(np.min(yh_np)), float(np.max(yh_np)))
        zlim: tuple[float, float] = (float(np.min(zh_np)), float(np.max(zh_np)))

        # Aspect Ratio
        xh_len: NP_REAL = NP_REAL(xlim[1] - xlim[0])
        yh_len: NP_REAL = NP_REAL(ylim[1] - ylim[0])
        zh_len: NP_REAL = NP_REAL(zlim[1] - zlim[0])

        # Tick Labels - fixed across frames
        x_ticks: NP_ARRAY[NP_REAL] = NP_REAL(
            MaxNLocator(nbins = 4).tick_values(xlim[0], xlim[1])
        )
        y_ticks: NP_ARRAY[NP_REAL] = NP_REAL(
            MaxNLocator(nbins = 4).tick_values(ylim[0], ylim[1])
        )
        z_ticks: NP_ARRAY[NP_REAL] = NP_REAL(
            MaxNLocator(nbins = 3).tick_values(0., np.floor(zlim[1]))
        )

        for ax in axs.flatten():
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)

            ax.set_autoscale_on(False)

            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax.zaxis.set_major_locator(FixedLocator(z_ticks))

            ax.xaxis.set_ticks(x_ticks)
            ax.yaxis.set_ticks(y_ticks)
            ax.zaxis.set_ticks(z_ticks)

            ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

        for ax in (axs[:-1, -1]).flatten():
            ax.xaxis.set_tick_params(labelcolor = "none")
            ax.yaxis.set_tick_params(labelcolor = "none")

        for ax in (axs[-1, :-1]).flatten():
            ax.yaxis.set_tick_params(labelcolor = "none")
            ax.zaxis.set_tick_params(labelcolor = "none")

        for ax in (axs[:-1, :-1]).flatten():
            ax.xaxis.set_tick_params(labelcolor = "none")
            ax.yaxis.set_tick_params(labelcolor = "none")
            ax.zaxis.set_tick_params(labelcolor = "none")

        # Axis labels
        axs[-1, -1].set_ylabel(r"y $\left[ km \right]$")

        for ax in (axs[:, -1]).flatten():
            ax.set_zlabel(r"z $\left[ km \right]$")

        for ax in (axs[-1, :]).flatten():
            ax.set_xlabel(r"x $\left[ km \right]$")

        # Suplabels
        dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.1f} $km$".format(dx * 1.e-3)

        fig.suptitle(
            r"Cloud Water Content $\left[ g\,m^{-3} \right]$" +
            " - {}".format(dx_str)
        )
        fig.supxlabel("") # Add for padding at bottom

        # Manual layout for animation stability.
        # You may need to tune these values for your preferred output.
        fig.subplots_adjust(
            left = 0.02,
            right = 0.84,
            bottom = 0.08,
            top = 0.86
        )

        # Use a persistent text artist instead of changing the 3D axes title.
        # This avoids layout jitter from varying title bounding boxes.
        time_text = axs[0, 0].text2D(
            0.5,
            1.02,
            "",
            transform = axs[0, 0].transAxes,
            ha = "center",
            va = "bottom"
        )

        #-----------------------------------------------------------------------
        # Animate the data
        #-----------------------------------------------------------------------
        daytime_times_flat: NP_ARRAY[NP_REAL] = NP_REAL(daytime_times.flatten())

        n_frames: NP_INT = NP_INT(n_t)
        fps: NP_INT = NP_INT(6)
        interval: NP_REAL = NP_REAL(1.e3) / NP_REAL(fps)

        voxel_artists: list = []

        def freeze_3d_axes():
            """
            Re-apply fixed 3D axes properties after ax.voxels().

            ax.voxels() can trigger autoscaling/reprojection. Re-applying these
            settings prevents changing limits, changing tick counts, and axes
            jitter during animation saving.
            """

            for ax in axs.flatten():
                ax.set_xlim3d(xlim)
                ax.set_ylim3d(ylim)
                ax.set_zlim3d(zlim)

                ax.set_autoscale_on(False)

                ax.xaxis.set_major_locator(FixedLocator(x_ticks))
                ax.yaxis.set_major_locator(FixedLocator(y_ticks))
                ax.zaxis.set_major_locator(FixedLocator(z_ticks))

                ax.xaxis.set_ticks(x_ticks)
                ax.yaxis.set_ticks(y_ticks)
                ax.zaxis.set_ticks(z_ticks)

                ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

        def draw_voxels(frame: int):
            """
            Draw one voxel frame and return the artists created by ax.voxels().

            Matplotlib 3D voxels returns a dictionary mapping voxel coordinates
            to Poly3DCollection artists.
            """

            voxel_dict = axs[0, 0].voxels(
                xx,
                yy,
                zz,
                filled[frame, ...],
                facecolors = facecolors[frame, ...],
                edgecolor = "none"
            )

            freeze_3d_axes()

            return list(voxel_dict.values())

        def init():
            """
            Draw the first frame once.
            """

            nonlocal voxel_artists

            voxel_artists = draw_voxels(0)
            time_text.set_text(r"{:8.2f} $h$".format(daytime_times_flat[0]))
            freeze_3d_axes()

            return voxel_artists + [time_text]

        def update(frame: int):
            """
            Remove old voxel artists, draw the requested frame, and update time.
            """

            nonlocal voxel_artists

            for artist in voxel_artists:
                artist.remove()

            voxel_artists = draw_voxels(frame)
            time_text.set_text(r"{:8.2f} $h$".format(daytime_times_flat[frame]))
            freeze_3d_axes()

            return voxel_artists + [time_text]

        ani = animation.FuncAnimation(
            fig = fig,
            func = update,
            init_func = init,
            frames = range(n_frames),
            interval = interval,
            blit = False,
            cache_frame_data = False
        )

        #-----------------------------------------------------------------------
        # Save the animation to file
        #-----------------------------------------------------------------------
        msg: str = "Saving animation to file..."
        print_msg(msg)

        writer = animation.FFMpegWriter(
            fps = fps,
            bitrate = 2400,
            codec = "libx264",
            extra_args = [
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p"
            ]
        )

        ani_filename: str = "rte_rrtmgp_cpp_atm_diorama.{}.mp4".format(lr_str)
        ani_filepath: str = os.path.join(rad_tran_vizdir, ani_filename)

        ani.save(
            filename = ani_filepath,
            writer = writer,
            dpi = 200
        )

        plt.close(fig)


if __name__ == "__main__":
    main()