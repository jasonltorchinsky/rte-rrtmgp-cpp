import os, sys
exp_hres_dir: str = os.path.normpath( \
    os.path.join(os.path.dirname(__file__), os.pardir))
if exp_hres_dir not in sys.path:
    sys.path.append(exp_hres_dir)

# Standard Library Imports
import argparse
import ast
import os
import re
import sys

from typing import Optional

# Third-Party Library Imports
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET
from convert_utils import get_sort_mask


# Script variables
prog_name: str = "animate_dpscream_output"
prog_desc: str = "Animates DP-SCREAM output."

def main(argv):
    # MPI Communicator info
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = prog_name,
        description = prog_desc
    )

    parser.add_argument("--rte_rrtmgp_cpp_input_file_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP input."
    )

    parser.add_argument("--rte_rrtmgp_cpp_output_file_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP output."
    )

    parser.add_argument("--rte_rrtmgp_cpp_viz_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["comparison"],
        help = "Path to RTE-RRTMGP-CPP viz directory."
    )

    args: argparse.Namespace = parser.parse_args()

    rte_rrtmgp_cpp_input_file_path: str = os.path.normpath(args.rte_rrtmgp_cpp_input_file_path[0])
    rte_rrtmgp_cpp_output_file_path: str = os.path.normpath(args.rte_rrtmgp_cpp_output_file_path[0])
    rte_rrtmgp_cpp_viz_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_viz_dir_path[0])

    file_ext: re.Pattern = re.compile("\\.in.nc")
    animation_file_name_root: str = file_ext.sub("", os.path.basename(rte_rrtmgp_cpp_input_file_path))

    xr_rte_rrtmgp_cpp_input: XR_DATASET = xr.open_dataset(rte_rrtmgp_cpp_input_file_path, engine="netcdf4")
    xr_rte_rrtmgp_cpp_output: XR_DATASET = xr.open_dataset(rte_rrtmgp_cpp_output_file_path, engine="netcdf4")

    # Recover grid for pcolormesh
    time: NP_ARRAY[np.datetime64] = xr_rte_rrtmgp_cpp_input["time"].values.astype(NP_REAL)  # Hours since simulation start
    nt: NP_INT = NP_INT(time.size)

    xh: NP_ARRAY[NP_REAL] = xr_rte_rrtmgp_cpp_input["xh"].values.astype(NP_REAL)  # Column x-interfaces
    yh: NP_ARRAY[NP_REAL] = xr_rte_rrtmgp_cpp_input["yh"].values.astype(NP_REAL)  # Column y-interfaces

    XX, YY = np.meshgrid(xh / 1000., yh / 1000., indexing="ij")

    nx: NP_INT = NP_INT(xh.size - 1)  # No. columns in x
    ny: NP_INT = NP_INT(yh.size - 1)  # No. columns in y

    # ----------------------------
    # Extract TWO fields to plot side-by-side
    # (Change these keys/labels as desired.)
    # ----------------------------
    field_key: str
    for field_key in ["flux_abs", "sfc_up", "sfc_dn", "tod_up"]:
        title_str: str
        if field_key == "flux_abs":
            title_str = r"Vertically-Integrated Absorbed Shortwave Flux [$W\,m^{-2}$]"
        elif field_key == "sfc_up":
            title_str = r"Upwelling Shortwave Surface Flux [$W\,m^{-2}$]"
        elif field_key == "sfc_dn":
            title_str = r"Downwelling Shortwave Surface Flux [$W\,m^{-2}$]"
        elif field_key == "tod_up":
            title_str = r"Upwelling Shortwave Top-of-Domain Flux [$W\,m^{-2}$]"

        ts_field: NP_ARRAY[NP_REAL]
        rt_field: NP_ARRAY[NP_REAL]

        [ts_field, rt_field] = get_fields(xr_rte_rrtmgp_cpp_input, xr_rte_rrtmgp_cpp_output, field_key)

        ts_field = np.transpose(ts_field, axes=(0, 2, 1))
        rt_field = np.transpose(rt_field, axes=(0, 2, 1))

        # Set up plot with two subplots (side-by-side)
        fig, (ax_l, ax_r) = plt.subplots(
            nrows=1, ncols=2, sharey = True,
            figsize=(12.8, 4.8),
            dpi=150,
            constrained_layout=True
        )

        # Colormap/norm for each subplot (separate scaling).
        # If you want identical color scaling across both, tell me and I’ll adjust.
        level_min = min(ts_field.min(), rt_field.min())
        level_max = max(ts_field.max(), rt_field.max())
        levels = MaxNLocator(nbins=64).tick_values(level_min, level_max)

        cmap = plt.colormaps["inferno"]
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        mesh_l = ax_l.pcolormesh(XX, YY, ts_field[0, ...], shading="flat", cmap=cmap, norm=norm)
        mesh_r = ax_r.pcolormesh(XX, YY, rt_field[0, ...], shading="flat", cmap=cmap, norm=norm)

        cbar = fig.colorbar(mesh_l, ax=[ax_l, ax_r], location="right", pad=0.02)
        cbar.set_label(title_str)

        ax_l.set_title("Two-Stream")
        ax_r.set_title("Ray-Tracer")

        for ax in (ax_l, ax_r):
            ax.set_xlabel(r"$x$ [$km$]")
        ax_l.set_ylabel(r"$y$ [$km$]")

        frame_text_l = ax_l.text(
            0.02, 0.98, "t_step = 0",
            transform=ax_l.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
        )
        frame_text_r = ax_r.text(
            0.02, 0.98, "t_step = 0",
            transform=ax_r.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
        )

        def update(frame):
            """
            Called for each frame of the animation.
            Updates both pcolormesh objects.
            """
            i = frame + 1

            Z_l = ts_field[i, ...]
            Z_r = rt_field[i, ...]

            # For pcolormesh, set_array expects the flattened cell array
            mesh_l.set_array(Z_l.ravel())
            mesh_r.set_array(Z_r.ravel())

            frame_text_l.set_text(f"t_step = {i}")
            frame_text_r.set_text(f"t_step = {i}")

            return [mesh_l, mesh_r, frame_text_l, frame_text_r]

        ani = FuncAnimation(
            fig,
            update,
            frames=nt - 1,
            blit=False
        )

        animation_file_name: str = animation_file_name_root + "." +  field_key + ".gif"
        animation_file_path: str = os.path.join(rte_rrtmgp_cpp_viz_dir_path, animation_file_name)
        ani.save(animation_file_path, writer=PillowWriter(fps=20))

def get_fields(xr_rte_rrtmgp_cpp_input: XR_DATASET, xr_rte_rrtmgp_cpp_output: XR_DATASET, key: str, **kwargs) -> NP_ARRAY[NP_REAL]:

    default_kwargs: dict = {
        "z_max" : 100000.0,  # Maximum altitude
    }

    l_kwargs: dict = {**default_kwargs, **kwargs}

    assert(key in ["flux_abs", "sfc_up", "sfc_dn", "tod_up"])

    ts_field: XR_DATAARRAY # Two-Stream solver quantity
    rt_field: XR_DATAARRAY # Ray-Tracer solver quantity
    if key == "flux_abs": # Absorbed shortwave flux [W m^(-3)]
        z_lev: XR_DATAARRAY = xr_rte_rrtmgp_cpp_input["z_lev"].sel(z_lev = slice(None, l_kwargs["z_max"])) # Level altitude - z-dimension [m]; (n_lev_z)
        z_lay: XR_DATAARRAY = xr_rte_rrtmgp_cpp_input["z_lay"].sel(z_lay = slice(None, z_lev.max())) # Layer altitude - z-dimension [m]; (n_lay_z)

        # Two-Stream
        dz = z_lev.diff("z_lev").rename({"z_lev": "lay"}).rename("dz").assign_coords(lay = z_lay.values) # Layer thickness lives on layer midpoints
        ts_flux_dn: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_dn"].sel(lev = slice(None, l_kwargs["z_max"])).rename("ts_flux_dn") # (z_lev, y, x); [W m^(-2)]
        ts_flux_up: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_up"].sel(lev = slice(None, l_kwargs["z_max"])).rename("ts_flux_up") # (z_lev, y, x); [W m^(-2)]
        ts_flux_diff: XR_DATAARRAY = \
            ( ts_flux_up.isel(lev = slice(None, -1)) - ts_flux_dn.isel(lev = slice(None, -1)) ).rename({"lev": "lay"}).assign_coords(lay = z_lay.values) \
            + ( ts_flux_dn.isel(lev = slice(1, None)) - ts_flux_up.isel(lev = slice(1, None))).rename({"lev": "lay"}).assign_coords(lay = z_lay.values)
        ts_flux_diff.attrs.update({"long_name" : "Difference in incoming and outgoing shortwave flux at each layer (TwoStrem solver)"})
        ts_field: XR_DATAARRAY = ts_flux_diff / dz # (z_lay, y, x); [W m^(-3)]
        ts_field = (ts_field * dz).sum(dim = "lay")
        ts_field.attrs.pop("description")
        ts_field.attrs.update({"long_name" : "Vertically-integrated absorbed shortwave fluxes (TwoStream solver)", "units" : "W m-2"})

        # Ray-Tracer
        rt_flux_abs_dif: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_abs_dif"].sel(z = slice(None, l_kwargs["z_max"])).rename({"z" : "lay"}) # (z_lay, y, x); [W m^(-3)]
        rt_flux_abs_dir: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_abs_dir"].sel(z = slice(None, l_kwargs["z_max"])).rename({"z" : "lay"}) # (z_lay, y, x); [W m^(-3)]
        rt_field: XR_DATAARRAY = rt_flux_abs_dif + rt_flux_abs_dir
        rt_field = (rt_field * dz).sum(dim = "lay")
        rt_field.attrs.update({"long_name" : "Vertically-integrated absorbed shortwave fluxes (Monte Carlo ray tracer)", "units" : "W m-2"})
    elif key == "sfc_up":
        ts_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_up"].isel(lev = 0) # (ny, nx)
        ts_field.attrs.update({"long_name" : "Upwelling shortwave surface fluxes (TwoStream solver)"})

        rt_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_sfc_up"] # (ny, nx)
    elif key == "sfc_dn":
        ts_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_dn"].isel(lev = 0) # (ny, nx)
        ts_field.attrs.update({"long_name" : "Downwelling shortwave surface fluxes (TwoStream solver)"})

        rt_flux_sfc_dir: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_sfc_dir"] # (ny, nx)
        rt_flux_sfc_dif: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_sfc_dif"] # (ny, nx)
        rt_field: XR_DATAARRAY = rt_flux_sfc_dir + rt_flux_sfc_dif
        rt_field.attrs.update({"long_name" : "Downwelling shortwave surface fluxes (Monte Carlo ray tracer)"})
    elif key == "tod_up":
        ts_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_up"].isel(lev = -1) # (ny, nx)
        ts_field.attrs.update({"long_name" : "Upwelling shortwave top-of-domain fluxes (TwoStream solver)"})

        rt_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_tod_up"] # (ny, nx)

    return [ts_field.values.astype(NP_REAL), rt_field.values.astype(NP_REAL)]

if __name__ == "__main__":
    main(sys.argv)