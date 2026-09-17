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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    XR_DATAARRAY, \
    MPL_AXES, MPL_FIGURE, MPL_LINEAR_SEGMENTED_COLORMAP, MPL_LOGNORM, \
    MPL_COLORBAR
from consts.visual import cloud_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_times, \
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
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate all quantities needed for plotting.")
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
        z_max = z_max,
        method = "cloud_top")

    # Only plot morning and noon.
    # Flattened order is:
    # day 0 morning, day 0 noon, day 1 morning, day 1 noon, ...
    time_indices: NP_ARRAY[NP_INT] = mnn_indices[:,0:2].flatten()
    times: NP_ARRAY[NP_REAL] = mnn_times[:,0:2].flatten()

    n_t: NP_INT = NP_INT(time_indices.size)
    n_times_per_day: NP_INT = NP_INT(2)

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
        working_filename: str = "rte_rrtmgp_cpp_diorama_atm.{}.nc".format(lr_str)
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

                if has_required_vars and has_required_dims and valid_shapes and has_required_attrs:
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
                    (((alpha_max - alpha_min) * (cwc - cwc_min) / (cwc_max - cwc_min)) + alpha_min).to_numpy()) # [n_t, n_z, n_y, n_x]
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

        cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_cwc, vmax = max_cwc)

        #-----------------------------------------------------------------------
        # Set up the figure
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        # Layout:
        #   rows    = days
        #   columns = morning, noon
        nrows: NP_INT = NP_INT(ndays)
        ncols: NP_INT = NP_INT(n_times_per_day)

        fig_height_per_row: NP_REAL = NP_REAL(2.0)
        fig_height: NP_REAL = NP_REAL(fig_height_per_row * NP_REAL(nrows))
        fig_width: NP_REAL = NP_REAL(6.5)

        subplot_top_in: NP_REAL = NP_REAL(0.24)
        subplot_top: NP_REAL = NP_REAL((fig_height - subplot_top_in) / fig_height)

        cbar_bottom_in: NP_REAL = NP_REAL(0.16)
        cbar_top_in: NP_REAL = NP_REAL(0.40)
        cbar_bottom: NP_REAL = NP_REAL(cbar_bottom_in / fig_height)
        cbar_height: NP_REAL = NP_REAL((fig_height - cbar_bottom_in - cbar_top_in) / fig_height)

        fig: MPL_FIGURE
        axs: NP_ARRAY[MPL_AXES]
        fig, axs = plt.subplots(
            nrows = nrows, ncols = ncols,
            sharex = False, sharey = False,
            constrained_layout = False,
            figsize = (fig_width, fig_height),
            subplot_kw = {"projection" : "3d"})

        # Ensure axs is always indexed as axs[row, col].
        if ncols == 1:
            axs = axs[...,None]
        elif nrows == 1:
            axs = axs[None,...]

        fig.subplots_adjust(
            left = 0.04,
            right = 0.78,
            bottom = 0.02,
            top = subplot_top,
            wspace = 0.04,
            hspace = -0.16
        )

        cax = fig.add_axes([0.90, cbar_bottom, 0.02, cbar_height])

        #-----------------------------------------------------------------------
        # Plot the data
        #-----------------------------------------------------------------------
        msg: str = "Plotting the data..."
        print_msg(msg)

        cwc_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[cloud_cmap]

        rgba_valid: NP_ARRAY[NP_REAL] = facecolors[filled]
        if rgba_valid.shape[0] > 0:
            rgb_valid: NP_ARRAY[NP_REAL] = rgba_valid[:,:3]
            rgb_nonzero: NP_ARRAY[NP_BOOL] = np.any(rgb_valid > 0., axis = 1)
            if np.any(rgb_nonzero):
                rgba_nonzero: NP_ARRAY[NP_REAL] = rgba_valid[rgb_nonzero]
                cwc_norm_valid: NP_ARRAY[NP_REAL] = NP_REAL(
                    np.clip(
                        cwc_colormap_norm.inverse(
                            np.clip(
                                np.interp(
                                    rgba_nonzero[:,0],
                                    np.linspace(0., 1., cwc_colormap.N),
                                    cwc_colormap(np.linspace(0., 1., cwc_colormap.N))[:,0]
                                ),
                                0., 1.
                            )
                        ),
                        cwc_colormap_norm.vmin,
                        cwc_colormap_norm.vmax
                    )
                )
                max_cwc_colorbar: NP_REAL = NP_REAL(np.nanmax(cwc_norm_valid))
                min_cwc_colorbar: NP_REAL = NP_REAL(cwc_colormap_norm.vmin)
                if (not np.isfinite(max_cwc_colorbar)) or (max_cwc_colorbar <= min_cwc_colorbar):
                    max_cwc_colorbar = NP_REAL(min_cwc_colorbar * NP_REAL(10.))
            else:
                min_cwc_colorbar = NP_REAL(1.e-2)
                max_cwc_colorbar = NP_REAL(1.e-1)
        else:
            min_cwc_colorbar = NP_REAL(1.e-2)
            max_cwc_colorbar = NP_REAL(1.e-1)

        cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_cwc_colorbar, vmax = max_cwc_colorbar)

        x0: NP_ARRAY[NP_REAL] = xh_plot[:-1]
        x1: NP_ARRAY[NP_REAL] = xh_plot[1:]
        y0: NP_ARRAY[NP_REAL] = yh_plot[:-1]
        y1: NP_ARRAY[NP_REAL] = yh_plot[1:]
        z0: NP_ARRAY[NP_REAL] = zh_plot[:-1]
        z1: NP_ARRAY[NP_REAL] = zh_plot[1:]

        x_plot_min: NP_REAL = NP_REAL(xh_plot.min())
        x_plot_max: NP_REAL = NP_REAL(xh_plot.max())
        y_plot_min: NP_REAL = NP_REAL(yh_plot.min())
        y_plot_max: NP_REAL = NP_REAL(yh_plot.max())
        z_plot_min: NP_REAL = NP_REAL(0.)
        z_plot_max: NP_REAL = NP_REAL(np.ceil(zh_plot.max()))

        if x_plot_max <= x_plot_min:
            x_plot_max = NP_REAL(x_plot_min + NP_REAL(1.))
        if y_plot_max <= y_plot_min:
            y_plot_max = NP_REAL(y_plot_min + NP_REAL(1.))
        if z_plot_max <= z_plot_min:
            z_plot_max = NP_REAL(1.)

        jj: int
        kk: int
        for jj in range(0, nrows):
            for kk in range(0, ncols):
                index: int = n_times_per_day * jj + kk

                filled_i: NP_ARRAY[NP_BOOL] = filled[index,...]
                if not np.any(filled_i):
                    continue

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

                poly: Poly3DCollection = Poly3DCollection(
                    verts,
                    facecolors = poly_facecolors,
                    edgecolors = poly_facecolors,
                    linewidths = 0
                )
                axs[jj, kk].add_collection3d(poly)

        # Set bounds after plotting so they are determined by the simulation
        # domain, not by automatically generated tick locations.
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.set_xlim(x_plot_min, x_plot_max)
            ax.set_ylim(y_plot_min, y_plot_max)
            ax.set_zlim(z_plot_min, z_plot_max)

        #-----------------------------------------------------------------------
        # Set up colorbar
        #-----------------------------------------------------------------------
        msg: str = "Setting up colorbar..."
        print_msg(msg)

        cwc_colorbar = fig.colorbar(
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
        xh_len: NP_REAL = NP_REAL(x_plot_max - x_plot_min)
        yh_len: NP_REAL = NP_REAL(y_plot_max - y_plot_min)
        zh_len: NP_REAL = NP_REAL(z_plot_max - z_plot_min)
        ax: MPL_AXES
        for ax in axs.flatten():
            ax.set_box_aspect([xh_len, yh_len, 2.5 * zh_len])

        # Diorama Time Labels
        jj: int
        kk: int
        for jj in range(0, nrows):
            for kk in range(0, ncols):
                index: int = n_times_per_day * jj + kk
                axs[jj,kk].text2D(
                    0.5, 0.86,
                    r"{} $h$".format(times_plot[index]),
                    transform = axs[jj,kk].transAxes,
                    ha = "center",
                    va = "top"
                )

        # Tick Labels - Get rid of unnecessary ones and keep them uniform
        # across all plots.  Choose 3--5 evenly spaced horizontal ticks depending
        # on whether the domain size is evenly divisible by the number of tick
        # intervals.  This prevents ticks from extending beyond the simulation
        # bounds and avoids uneven endpoints, e.g., 0, 15, 30, 45, 50 for 50 km.
        x_n_ticks: NP_INT = NP_INT(3)
        y_n_ticks: NP_INT = NP_INT(3)

        xh_len_round: NP_REAL = NP_REAL(np.round(xh_len))
        yh_len_round: NP_REAL = NP_REAL(np.round(yh_len))

        if np.isclose(xh_len, xh_len_round, rtol = 1.e-10, atol = 1.e-8):
            n_ticks_i: NP_INT
            for n_ticks_i in [NP_INT(5), NP_INT(4), NP_INT(3)]:
                n_intervals_i: NP_INT = NP_INT(n_ticks_i - NP_INT(1))
                x_tick_spacing_i: NP_REAL = NP_REAL(xh_len_round / n_intervals_i)
                if np.isclose(x_tick_spacing_i, np.round(x_tick_spacing_i), rtol = 1.e-10, atol = 1.e-8):
                    x_n_ticks = n_ticks_i
                    break

        if np.isclose(yh_len, yh_len_round, rtol = 1.e-10, atol = 1.e-8):
            n_ticks_i: NP_INT
            for n_ticks_i in [NP_INT(5), NP_INT(4), NP_INT(3)]:
                n_intervals_i: NP_INT = NP_INT(n_ticks_i - NP_INT(1))
                y_tick_spacing_i: NP_REAL = NP_REAL(yh_len_round / n_intervals_i)
                if np.isclose(y_tick_spacing_i, np.round(y_tick_spacing_i), rtol = 1.e-10, atol = 1.e-8):
                    y_n_ticks = n_ticks_i
                    break

        x_ticks: NP_ARRAY[NP_REAL] = NP_REAL(np.linspace(x_plot_min, x_plot_max, x_n_ticks))
        y_ticks: NP_ARRAY[NP_REAL] = NP_REAL(np.linspace(y_plot_min, y_plot_max, y_n_ticks))
        z_ticks: NP_ARRAY[NP_REAL] = NP_REAL(np.array([z_plot_min, z_plot_max]))

        ax: MPL_AXES
        for ax in axs.flatten():
            ax.xaxis.set_ticks(x_ticks)
            ax.yaxis.set_ticks(y_ticks)
            ax.zaxis.set_ticks(z_ticks)

            # Setting ticks can expand the axis limits if any tick lies outside
            # the current view.  Re-apply the simulation bounds explicitly.
            ax.set_xlim(x_plot_min, x_plot_max)
            ax.set_ylim(y_plot_min, y_plot_max)
            ax.set_zlim(z_plot_min, z_plot_max)

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
        fig.supxlabel(" ") # Add for padding at bottom

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        msg: str = "Saving plot to file..."
        print_msg(msg)

        plt_filename = "rte_rrtmgp_cpp_diorama_atm.{}.png".format(lr_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 256)
        plt.close(fig)

if __name__ == "__main__":
    main()