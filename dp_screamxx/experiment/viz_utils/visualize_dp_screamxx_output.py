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
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET
from convert_utils import get_sort_mask


# Script variables
prog_name: str = "visualize_dpscream_output"
prog_desc: str = "Visualizes DP-SCREAM output."

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
    file_name_root: str = file_ext.sub("", os.path.basename(dpscream_file_path))

    xr_dpscream: XR_DATASET = xr.open_dataset(dpscream_file_path, engine = "netcdf4")
    sort_mask: NP_ARRAY[NP_INT] = get_sort_mask(xr_dpscream)

    # Recover hours since simulation start
    time: NP_ARRAY[np.datetime64] = xr_dpscream["time"].values
    time: NP_ARRAY[NP_REAL] = (time - time[0]).astype(NP_REAL) / 3.6e12 # Hours since simulation start, dtime is in ns

    # Recover time-step numbers
    t: NP_ARRAY[NP_INT] = np.arange(time.size, dtype = NP_INT)

    # Recover SZAs
    cosine_solar_zenith_angle: NP_ARRAY[NP_REAL] = \
        xr_dpscream["cosine_solar_zenith_angle"].isel(ncol = 0).values
    sza: NP_ARRAY[NP_REAL] = np.acos(cosine_solar_zenith_angle)

    # Set up plot
    fig, ax = plt.subplots(figsize = (6.4, 4.8), dpi = 150)

    sza_plot = ax.plot(t, sza)

    ax.set_title(file_name_root)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Solar Zenith Angle")

    visualization_file_name: str = file_name_root + "_sza.png"
    visualization_file_path: str = os.path.join(plot_outdir_path, visualization_file_name)
    plt.savefig(visualization_file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

if __name__ == "__main__":
    main(sys.argv)