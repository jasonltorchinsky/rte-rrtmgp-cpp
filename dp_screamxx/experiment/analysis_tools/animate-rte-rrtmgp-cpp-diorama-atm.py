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
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    XR_DATAARRAY, \
    MPL_AXES, MPL_FIGURE, MPL_LINEAR_SEGMENTED_COLORMAP, MPL_LOGNORM, \
    MPL_COLORBAR
from consts.visual import cloud_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, \
    calc_cloud_wc, calc_z_max_info, find_grid, print_msg

# Script variables
prog_name: str = "animate-rte-rrtmgp-cpp-diorama-atm"
prog_desc: str = "Create an animation of the atmospheric state for RTE-RRTMGP-CPP."

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
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate all quantities needed for plotting.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
    parser.add_argument("--frames", nargs = "?", default = 0, type = int,
        help = "Number of frames per day to include in the animation. If <= 0, use all frames.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None
    nframes_day_req: NP_INT = NP_INT(args.frames)

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
    # Obtain time index and z-max info
    #-----------------------------------------------------------------------
    msg: str = "Obtaining time index and z-max info..."
    print_msg(msg)

    ds_time: xr.Dataset = xr.load_dataset(rad_tran_infiles[0])

    time_var_name: str
    if "time" in ds_time.variables:
        time_var_name = "time"
    elif "t" in ds_time.variables:
        time_var_name = "t"
    else:
        ds_time.close()
        raise ValueError("Could not determine time variable name in input file.")

    time_all: NP_ARRAY[NP_REAL] = NP_REAL(ds_time[time_var_name].to_numpy())
    nt_all: NP_INT = NP_INT(time_all.size)

    if nt_all < 1:
        ds_time.close()
        raise ValueError("No time points found in input file.")

    daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(
        rad_tran_infiles[0]
    ) # [ndays, ndaytime]
    ndays: NP_INT = NP_INT(daytime_indices.shape[0])
    z_max_info: dict = calc_z_max_info(
        rad_tran_infiles[0],
        z_max = z_max)

    time_indices_list: list[NP_ARRAY[NP_INT]] = []
    day_frame_counts: list[int] = []

    ii_day: int
    for ii_day in range(0, ndays):
        day_indices_full: NP_ARRAY[NP_INT] = NP_INT(daytime_indices[ii_day,:])

        if (nframes_day_req is not None) and (nframes_day_req > 0) and (day_indices_full.size > nframes_day_req):
            sample_indices: NP_ARRAY[NP_INT] = NP_INT(
                np.linspace(0, day_indices_full.size - 1, nframes_day_req, dtype = int)
            )
            day_indices: NP_ARRAY[NP_INT] = day_indices_full[sample_indices]
        else:
            day_indices = day_indices_full

        if day_indices.size < 1:
            day_indices = np.array([day_indices_full[0]], dtype = NP_INT)

        time_indices_list.append(day_indices.astype(NP_INT))
        day_frame_counts.append(int(day_indices.size))

    time_indices: NP_ARRAY[NP_INT] = np.concatenate(time_indices_list).astype(NP_INT)
    times: NP_ARRAY[NP_REAL] = time_all[time_indices]

    n_t: NP_INT = NP_INT(time_indices.size)

    ds_time.close()

    fps: NP_INT = NP_INT(8)
    titlecard_frames: NP_INT = NP_INT(fps)

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

        #-----------------------------------------------------------------------
        # Read cached plotting data or calculate and save them
        #-----------------------------------------------------------------------
        working_filename: str = "rte_rrtmgp_cpp_diorama_atm.animation.{}.nc".format(lr_str)
        working_filepath: str = os.path.join(working_dir, working_filename)

        need_recalculate: bool = recalculate or (not os.path.exists(working_filepath))

        if not need_recalculate:
            msg: str = "Checking cached plotting data..."
            print_msg(msg)

            ds_plot: Optional[xr.Dataset] = None
            try:
                ds_plot = xr.load_dataset(working_filepath)

                required_vars: list[str] = [
                    "filled",
                    "facecolors",
                    "time_plot",
                    "xh",
                    "yh",
                    "zh"
                ]
                required_dims: list[str] = [
                    "time_plot",
                    "x_edge",
                    "y_edge",
                    "z_edge",
                    "x",
                    "y",
                    "z",
                    "rgba"
                ]

                has_required_vars: bool = all(var_name in ds_plot.variables for var_name in required_vars)
                has_required_dims: bool = all(dim_name in ds_plot.dims for dim_name in required_dims)
                has_required_attrs: bool = all(attr_name in ds_plot.attrs for attr_name in ["cwc_vmin", "cwc_vmax"])

                valid_shapes: bool = True
                if has_required_vars and has_required_dims:
                    valid_shapes = valid_shapes and (ds_plot["filled"].shape == (n_t, n_x, n_y, n_z))
                    valid_shapes = valid_shapes and (ds_plot["facecolors"].shape == (n_t, n_x, n_y, n_z, 4))
                    valid_shapes = valid_shapes and (ds_plot["xh"].size == n_xh)
                    valid_shapes = valid_shapes and (ds_plot["yh"].size == n_yh)
                    valid_shapes = valid_shapes and (ds_plot["zh"].size == n_zh)
                    valid_shapes = valid_shapes and (ds_plot["time_plot"].size == n_t)
                else:
                    valid_shapes = False

                valid_times: bool = False
                if has_required_vars and ("time_plot" in ds_plot.variables):
                    valid_times = np.array_equal(
                        NP_REAL(ds_plot["time_plot"].to_numpy()),
                        NP_REAL(times)
                    )

                if has_required_vars and has_required_dims and valid_shapes and has_required_attrs and valid_times:
                    filled: NP_ARRAY[NP_BOOL] = NP_BOOL(ds_plot["filled"].to_numpy())
                    facecolors: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["facecolors"].to_numpy())
                    times_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["time_plot"].to_numpy())
                    xh_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["xh"].to_numpy())
                    yh_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["yh"].to_numpy())
                    zh_plot: NP_ARRAY[NP_REAL] = NP_REAL(ds_plot["zh"].to_numpy())

                    min_cwc: NP_REAL = NP_REAL(ds_plot.attrs["cwc_vmin"])
                    max_cwc: NP_REAL = NP_REAL(ds_plot.attrs["cwc_vmax"])
                    cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_cwc, vmax = max_cwc)

                    use_cached: bool = True
                else:
                    use_cached: bool = False

                ds_plot.close()

                if not use_cached:
                    need_recalculate = True

            except Exception:
                if ds_plot is not None:
                    ds_plot.close()
                need_recalculate = True

        if need_recalculate:
            msg: str = "Calculating plotting data..."
            print_msg(msg)

            #-------------------------------------------------------------------
            # Calculate CWC
            #-------------------------------------------------------------------
            msg: str = "Obtaining cloud water content info..."
            print_msg(msg)

            cwc: XR_DATAARRAY = calc_cloud_wc(
                rad_tran_infile, 
                time_indices = time_indices,
                z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [time, lay, y, x]

            # Discard values less than the tolerance
            cwc_tol: NP_REAL = NP_REAL(1.e-2)
            cwc = xr.where(cwc < cwc_tol, NP_REAL(0.), cwc)

            #-------------------------------------------------------------------
            # Calculate the filled and facecolors for each voxel
            #-------------------------------------------------------------------
            max_cwc: NP_REAL = NP_REAL(cwc.max())
            min_cwc: NP_REAL = cwc_tol

            if max_cwc <= min_cwc:
                max_cwc = NP_REAL(min_cwc * NP_REAL(10.))

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
            filled = np.transpose(filled, axes = (0, 3, 2, 1)) # [n_t, n_x, n_y, n_z]
            cwc_norm = np.transpose(cwc_norm, axes = (0, 3, 2, 1)) # [n_t, n_x, n_y, n_z]
            facecolors: NP_ARRAY[NP_REAL] = NP_REAL(cwc_colormap(cwc_norm))

            # Set alpha
            cwc_min: NP_REAL = NP_REAL(cwc.min())
            cwc_max: NP_REAL = NP_REAL(cwc.max())
            alpha_min: NP_REAL = NP_REAL(0.1)
            alpha_max: NP_REAL = NP_REAL(0.5)

            if cwc_max > cwc_min:
                alpha: NP_ARRAY[NP_REAL] = NP_REAL(
                    (((alpha_max - alpha_min) * (cwc - cwc_min) / (cwc_max - cwc_min)) + alpha_min).to_numpy()) # [n_t, n_z, n_y, x]
            else:
                alpha = NP_REAL((alpha_min * np.ones(cwc.shape, dtype = NP_REAL)))

            alpha = np.transpose(alpha, axes = (0, 3, 2, 1)) # [n_t, n_x, n_y, n_z]
            facecolors[...,3] = alpha

            xh_plot: NP_ARRAY[NP_REAL] = NP_REAL(xh.to_numpy())
            yh_plot: NP_ARRAY[NP_REAL] = NP_REAL(yh.to_numpy())
            zh_plot: NP_ARRAY[NP_REAL] = NP_REAL(zh.to_numpy())
            times_plot: NP_ARRAY[NP_REAL] = NP_REAL(times)

            msg: str = "Saving plotting data to file..."
            print_msg(msg)

            ds_plot: xr.Dataset = xr.Dataset(
                data_vars = {
                    "filled" : (
                        ("time_plot", "x", "y", "z"),
                        filled.astype(np.int8)
                    ),
                    "facecolors" : (
                        ("time_plot", "x", "y", "z", "rgba"),
                        facecolors.astype(NP_REAL)
                    )
                },
                coords = {
                    "time_plot" : times_plot,
                    "xh" : (("x_edge",), xh_plot),
                    "yh" : (("y_edge",), yh_plot),
                    "zh" : (("z_edge",), zh_plot),
                    "x" : np.arange(n_x, dtype = NP_INT),
                    "y" : np.arange(n_y, dtype = NP_INT),
                    "z" : np.arange(n_z, dtype = NP_INT),
                    "x_edge" : np.arange(n_xh, dtype = NP_INT),
                    "y_edge" : np.arange(n_yh, dtype = NP_INT),
                    "z_edge" : np.arange(n_zh, dtype = NP_INT),
                    "rgba" : np.arange(4, dtype = NP_INT)
                },
                attrs = {
                    "lr_str" : lr_str,
                    "cloud_cmap" : cloud_cmap,
                    "z_max_km" : float(zh_plot.max()) if zh_plot.size > 0 else 0.0,
                    "cwc_vmin" : float(min_cwc),
                    "cwc_vmax" : float(max_cwc)
                }
            )
            ds_plot["filled"].attrs["description"] = "Voxel occupancy mask"
            ds_plot["facecolors"].attrs["description"] = "RGBA voxel face colors"
            ds_plot["xh"].attrs["units"] = "km"
            ds_plot["yh"].attrs["units"] = "km"
            ds_plot["zh"].attrs["units"] = "km"
            ds_plot["time_plot"].attrs["units"] = "h"

            ds_plot.to_netcdf(working_filepath)
            ds_plot.close()

        if need_recalculate:
            filled = NP_BOOL(filled)
            facecolors = NP_REAL(facecolors)
            times_plot = NP_REAL(times_plot)
            xh_plot = NP_REAL(xh_plot)
            yh_plot = NP_REAL(yh_plot)
            zh_plot = NP_REAL(zh_plot)

        cwc_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[cloud_cmap]
        cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_cwc, vmax = max_cwc)

        #-----------------------------------------------------------------------
        # Build animation frame index with 1-second title cards before each day
        #-----------------------------------------------------------------------
        msg: str = "Building animation frame sequence..."
        print_msg(msg)

        animation_indices_list: list[int] = []
        animation_is_titlecard_list: list[bool] = []
        animation_day_list: list[int] = []

        i0: int = 0
        for ii_day in range(0, ndays):
            for _ in range(0, int(titlecard_frames)):
                animation_indices_list.append(i0)
                animation_is_titlecard_list.append(True)
                animation_day_list.append(ii_day)

            n_day_frames: int = day_frame_counts[ii_day]
            for jj in range(0, n_day_frames):
                animation_indices_list.append(i0 + jj)
                animation_is_titlecard_list.append(False)
                animation_day_list.append(ii_day)

            i0 += n_day_frames

        animation_indices: NP_ARRAY[NP_INT] = np.array(animation_indices_list, dtype = NP_INT)
        animation_is_titlecard: NP_ARRAY[NP_BOOL] = np.array(animation_is_titlecard_list, dtype = NP_BOOL)
        animation_day: NP_ARRAY[NP_INT] = np.array(animation_day_list, dtype = NP_INT)
        n_anim_frames: NP_INT = NP_INT(animation_indices.size)

        #-----------------------------------------------------------------------
        # Set up the figure
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        fig_height: NP_REAL = NP_REAL(4.0)
        fig_width: NP_REAL = NP_REAL(6.5)
        fig: MPL_FIGURE
        ax: MPL_AXES
        fig = plt.figure(figsize = (fig_width, fig_height))
        ax = fig.add_subplot(111, projection = "3d")

        fig.subplots_adjust(
            left = 0.04,
            right = 0.82,
            bottom = 0.00,
            top = 0.94
        )

        cax = fig.add_axes([0.87, 0.08, 0.025, 0.80])

        #-----------------------------------------------------------------------
        # Set up colorbar
        #-----------------------------------------------------------------------
        msg: str = "Setting up colorbar..."
        print_msg(msg)

        cwc_colorbar: MPL_COLORBAR = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm = cwc_colormap_norm,
                cmap = cwc_colormap
            ),
            cax = cax
        )
        cwc_colorbar.ax.set_yscale("log")

        #-----------------------------------------------------------------------
        # Set style elements
        #-----------------------------------------------------------------------
        msg: str = "Setting style elements..."
        print_msg(msg)

        # Background Panes
        pane_color: list[float] = [0.0, 0.0, 0.0, 0.0]
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)

        # Set grid linewidth
        ax.xaxis._axinfo["grid"]["linewidth"] = 0
        ax.yaxis._axinfo["grid"]["linewidth"] = 0
        ax.zaxis._axinfo["grid"]["linewidth"] = 0

        # Aspect Ratio
        xh_len: NP_REAL = NP_REAL(xh_plot.max() - xh_plot.min())
        yh_len: NP_REAL = NP_REAL(yh_plot.max() - yh_plot.min())
        zh_len: NP_REAL = NP_REAL(zh_plot.max() - zh_plot.min())
        ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

        # Tick Labels
        x_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(xh_plot.min(), xh_plot.max()))
        y_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 4).tick_values(yh_plot.min(), yh_plot.max()))
        z_ticks: NP_ARRAY[NP_REAL] = NP_REAL(MaxNLocator(nbins = 2).tick_values(0., np.floor(zh_plot.max())))
        ax.xaxis.set_ticks(x_ticks)
        ax.yaxis.set_ticks(y_ticks)
        ax.zaxis.set_ticks(z_ticks)

        # Axis labels
        ax.set_ylabel(r"y $\left[ km \right]$")
        ax.set_zlabel(r"z $\left[ km \right]$")
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

        time_text = fig.text(
            0.5, 0.86,
            "",
            ha = "center",
            va = "top"
        )

        titlecard_text = fig.text(
            0.5, 0.5,
            "",
            ha = "center",
            va = "center"
        )

        x0: NP_ARRAY[NP_REAL] = xh_plot[:-1]
        x1: NP_ARRAY[NP_REAL] = xh_plot[1:]
        y0: NP_ARRAY[NP_REAL] = yh_plot[:-1]
        y1: NP_ARRAY[NP_REAL] = yh_plot[1:]
        z0: NP_ARRAY[NP_REAL] = zh_plot[:-1]
        z1: NP_ARRAY[NP_REAL] = zh_plot[1:]

        current_poly: Optional[Poly3DCollection] = None

        #-----------------------------------------------------------------------
        # Plot the data
        #-----------------------------------------------------------------------
        msg: str = "Creating animation..."
        print_msg(msg)

        def update(i_anim: int):
            nonlocal current_poly

            index: int = int(animation_indices[i_anim])

            if current_poly is not None:
                current_poly.remove()
                current_poly = None

            if animation_is_titlecard[i_anim]:
                ax.set_axis_off()
                titlecard_text.set_text("Day {}".format(int(animation_day[i_anim])))
                time_text.set_text("")
                return [titlecard_text, time_text]

            ax.set_axis_on()
            titlecard_text.set_text("")

            # Background Panes
            pane_color: list[float] = [0.0, 0.0, 0.0, 0.0]
            ax.xaxis.set_pane_color(pane_color)
            ax.yaxis.set_pane_color(pane_color)
            ax.zaxis.set_pane_color(pane_color)

            # Set grid linewidth
            ax.xaxis._axinfo["grid"]["linewidth"] = 0
            ax.yaxis._axinfo["grid"]["linewidth"] = 0
            ax.zaxis._axinfo["grid"]["linewidth"] = 0

            # Tick Labels
            ax.xaxis.set_ticks(x_ticks)
            ax.yaxis.set_ticks(y_ticks)
            ax.zaxis.set_ticks(z_ticks)

            # Axis labels
            ax.set_ylabel(r"y $\left[ km \right]$")
            ax.set_zlabel(r"z $\left[ km \right]$")
            ax.set_xlabel(r"x $\left[ km \right]$")

            filled_i: NP_ARRAY[NP_BOOL] = filled[index,...]

            if np.any(filled_i):
                i_x: NP_ARRAY[NP_INT]
                i_y: NP_ARRAY[NP_INT]
                i_z: NP_ARRAY[NP_INT]
                i_x, i_y, i_z = np.where(filled_i)

                n_vox: int = int(i_x.size)

                xv0: NP_ARRAY[NP_REAL] = x0[i_x]
                xv1: NP_ARRAY[NP_REAL] = x1[i_x]
                yv0: NP_ARRAY[NP_REAL] = y0[i_y]
                yv1: NP_ARRAY[NP_REAL] = y1[i_y]
                zv0: NP_ARRAY[NP_REAL] = z0[i_z]
                zv1: NP_ARRAY[NP_REAL] = z1[i_z]

                verts: NP_ARRAY[NP_REAL] = np.empty((6 * n_vox, 4, 3), dtype = NP_REAL)

                verts[0::6,0,:] = np.stack((xv0, yv0, zv0), axis = 1)
                verts[0::6,1,:] = np.stack((xv1, yv0, zv0), axis = 1)
                verts[0::6,2,:] = np.stack((xv1, yv1, zv0), axis = 1)
                verts[0::6,3,:] = np.stack((xv0, yv1, zv0), axis = 1)

                verts[1::6,0,:] = np.stack((xv0, yv0, zv1), axis = 1)
                verts[1::6,1,:] = np.stack((xv1, yv0, zv1), axis = 1)
                verts[1::6,2,:] = np.stack((xv1, yv1, zv1), axis = 1)
                verts[1::6,3,:] = np.stack((xv0, yv1, zv1), axis = 1)

                verts[2::6,0,:] = np.stack((xv0, yv0, zv0), axis = 1)
                verts[2::6,1,:] = np.stack((xv1, yv0, zv0), axis = 1)
                verts[2::6,2,:] = np.stack((xv1, yv0, zv1), axis = 1)
                verts[2::6,3,:] = np.stack((xv0, yv0, zv1), axis = 1)

                verts[3::6,0,:] = np.stack((xv0, yv1, zv0), axis = 1)
                verts[3::6,1,:] = np.stack((xv1, yv1, zv0), axis = 1)
                verts[3::6,2,:] = np.stack((xv1, yv1, zv1), axis = 1)
                verts[3::6,3,:] = np.stack((xv0, yv1, zv1), axis = 1)

                verts[4::6,0,:] = np.stack((xv0, yv0, zv0), axis = 1)
                verts[4::6,1,:] = np.stack((xv0, yv1, zv0), axis = 1)
                verts[4::6,2,:] = np.stack((xv0, yv1, zv1), axis = 1)
                verts[4::6,3,:] = np.stack((xv0, yv0, zv1), axis = 1)

                verts[5::6,0,:] = np.stack((xv1, yv0, zv0), axis = 1)
                verts[5::6,1,:] = np.stack((xv1, yv1, zv0), axis = 1)
                verts[5::6,2,:] = np.stack((xv1, yv1, zv1), axis = 1)
                verts[5::6,3,:] = np.stack((xv1, yv0, zv1), axis = 1)

                voxel_facecolors: NP_ARRAY[NP_REAL] = facecolors[index, i_x, i_y, i_z, :]
                poly_facecolors: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors, 6, axis = 0)

                current_poly = Poly3DCollection(
                    verts,
                    facecolors = poly_facecolors,
                    edgecolors = poly_facecolors,
                    linewidths = 0
                )
                ax.add_collection3d(current_poly)

            ax.set_xlim(xh_plot.min(), xh_plot.max())
            ax.set_ylim(yh_plot.min(), yh_plot.max())
            ax.set_zlim(zh_plot.min(), zh_plot.max())

            time_text.set_text(
                r"{:.2f} $h$".format(times_plot[index])
            )

            artists: list = [time_text, titlecard_text]
            if current_poly is not None:
                artists.append(current_poly)

            return artists

        anim = animation.FuncAnimation(
            fig,
            update,
            frames = int(n_anim_frames),
            interval = float(1000.0 / fps),
            blit = False,
            repeat = False
        )

        #-----------------------------------------------------------------------
        # Save the animation to file
        #-----------------------------------------------------------------------
        msg: str = "Saving animation to file..."
        print_msg(msg)

        plt_filename = "rte_rrtmgp_cpp_diorama_atm.{}.mp4".format(lr_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        anim.save(
            plt_filepath,
            writer = animation.FFMpegWriter(fps = int(fps))
        )
        plt.close(fig)

if __name__ == "__main__":
    main()