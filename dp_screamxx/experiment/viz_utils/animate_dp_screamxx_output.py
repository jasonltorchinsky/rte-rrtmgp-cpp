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
    
    parser.add_argument("--dpscream_file_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to DP-SCREAM output."
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

    dpscream_file_path: str = os.path.normpath(args.dpscream_file_path[0])
    plot_outdir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_viz_dir_path[0])

    file_ext: re.Pattern = re.compile("\\.nc")
    animation_file_name_root: str = file_ext.sub("", os.path.basename(dpscream_file_path))

    xr_dpscream: XR_DATASET = xr.open_dataset(dpscream_file_path, engine = "netcdf4")
    sort_mask: NP_ARRAY[NP_INT] = get_sort_mask(xr_dpscream)

    # Recover grid for pcolormesh
    nt: NP_INT = NP_INT(xr_dpscream["time"].size) - 1 # Going to skip first time step
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].isel(ncol = sort_mask).values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].isel(ncol = sort_mask).values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    x: NP_ARRAY[NP_REAL] = np.unique(lon) # Column x-midpoint
    dx: NP_REAL = x[1] - x[0]
    xh: NP_ARRAY[NP_REAL] = np.append(x - dx / 2., x[-1] + dx / 2.) # Column x-interfaces

    y: NP_ARRAY[NP_REAL] = np.unique(lat) # Column y-midpoint
    dy: NP_REAL = y[1] - y[0]
    yh: NP_ARRAY[NP_REAL] = np.append(y - dy / 2., y[-1] + dy / 2.) # Column y-interfaces

    XX, YY = np.meshgrid(xh / 1000., yh / 1000., indexing = "ij")

    nx: NP_INT = NP_INT(x.size) # No. columns in x
    ny: NP_INT = NP_INT(y.size) # No. columns in y

    # Extract field
    field_key: str = "LW_flux_up_at_model_top"
    field_str: str = "olr"
    title_str: str = r"ToA Outgoing Longwave Radiation [$W\,m^{-2}$]"
    field: NP_ARRAY[NP_REAL] = \
        xr_dpscream[field_key].isel(time = slice(1, None), ncol = sort_mask).values.astype(NP_REAL)
    field = field.reshape(nt, nx, ny)

    # Set up plot
    fig, ax = plt.subplots(figsize = (6.4, 4.8), dpi = 150)

    levels = MaxNLocator(nbins = 64).tick_values(field.min(), field.max())
    cmap = plt.colormaps["Blues"]
    norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)

    mesh = ax.pcolormesh(XX, YY, field[0,...], shading = "flat", 
        cmap = cmap, norm = norm)
    fig.colorbar(mesh, ax = ax)
    ax.set_title(title_str)
    ax.set_xlabel(r"$x$ [$km$]")
    ax.set_ylabel(r"$y$ [$km$]")
    frame_text = ax.text(
        0.02, 0.98, "t_step = 0",
        transform = ax.transAxes,   # axes-relative coordinates
        ha = "left",
        va = "top",
        fontsize = 12,
        bbox = dict(facecolor = "white", alpha = 0.8, edgecolor = "black")
    )

    def update(frame):
        """
        This function gets called for each frame of the animation.
        It updates the pcolormesh data.
        """
        # Evolve Z with frame index
        Z_new = field[frame + 1,...]

        # Update the mesh array
        mesh.set_array(Z_new.ravel())

        frame_text.set_text("t_step = {}".format(frame + 1))

        return [mesh]

    ani = FuncAnimation(
        fig,
        update,
        frames = nt - 1,
        blit = False
    )

    animation_file_name: str = animation_file_name_root + "_" + field_str + ".gif"
    animation_file_path: str = os.path.join(plot_outdir_path, animation_file_name)
    ani.save(animation_file_path, writer = PillowWriter(fps = 20))

if __name__ == "__main__":
    main(sys.argv)