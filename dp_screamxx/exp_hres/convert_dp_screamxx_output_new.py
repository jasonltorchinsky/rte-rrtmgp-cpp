"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import ast
import os
import sys

from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interpn
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, MPI_INT, MPI_REAL, NC_REAL, NC_INT, \
    MPI_COMM, NP_ARRAY, NC_VARIABLE, XR_DATASET, MPI_ROOT, g
from utils.rte_rrtmgp_cpp_fields import grid_dimensions, grid_descriptions, \
    grid_units, fields_dimensions, fields_descriptions, fields_units

# Script variables
prog_name: str = "convert_dpscream_output"
prog_desc: str = "Converts DP-SCREAM output to RTE-RRTMGP-CPP+RT input."

def main(argv):
    # MPI Communicator info
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = prog_name,
        description = prog_desc)
    
    parser.add_argument("--dpscream_file_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to DP-SCREAM output.")
    
    parser.add_argument("--rte_rrtmgp_cpp_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Directory path for RTE-RRTMGP-CPP+RT input.")
    
    args: argparse.Namespace = parser.parse_args()

    dpscream_file_path: str = os.path.normpath(args.dpscream_file[0])
    rte_rrtmgp_cpp_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_dir[0])

    # Root rank opens the file
    if l_rank == MPI_ROOT:
        xr_dpscream: XR_DATASET = xr.open_dataset(dpscream_file_path,
            engine = "netcdf4")

    ### Open the DP-SCREAM output file
    ### Get the high-resolution horizontal grid
    ### Get the low-resolution horizontal grid(s)
    ### Get the uniform vertical grid
    ### Combine into the new grids
    ### For each time-step
        ### For each new grid
            ### Interpolate values to the new grid
            ### For each SZA
                ### Output RTE-RRTMGP-CPP+RT input file

def get_hr_grid(xr_dpscream: XR_DATASET) -> dict:
    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)).astype(NP_INT) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    n_col_x: NP_INT = NP_INT(np.unique(lon).size) # No. columns in x
    n_col_y: NP_INT = NP_INT(np.unique(lat).size) # No. columns in y
    cols: NP_ARRAY[NP_REAL] = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(n_col_x, n_col_y, 2)

    ## Construct the horizontal grids
    ### NOTE: The names xh, yh seem to refer to the interfaces between columns,
    ### but in the original rcemip experiment, they just tack on an extra value.
    ### They don't seem to be directly used in the code, so we will use them
    ### to be interfaces between columns.
    ### NOTE: Assume that horizontal grids are regularly spaced.
    x: NP_ARRAY[NP_REAL] = (cols[:,:,0])[0,:] # x-midpoints of each column [m]; (n_col_x)
    dx: NP_REAL = x[1] - x[0]
    xh: NP_ARRAY[NP_REAL] = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-interfaces of each column [m]; (n_col_x + 1)

    y: NP_ARRAY[NP_REAL] = (cols[:,:,1])[:,0] # y-midpoints of each column [m]; (n_col_y)
    dy: NP_REAL = y[1] - y[0]
    yh: NP_ARRAY[NP_REAL] = np.append(y - (dy / 2.), x[-1] + (dy / 2.)) # y-interfaces of each column [m]; (n_col_y + 1)

    ## Dimension sizes - DP-SCREAM
    ntime_input: Optional[NP_INT] = NP_INT(xr_dpscream.sizes["time"]) # No. time-steps in input
    if times.size == 0:
        ntime: Optional[NP_INT] = ntime_input
        times = np.arange(ntime, dtype = NP_INT)
    else:
        ntime: Optional[NP_INT] = NP_INT(times.size)
    times: Optional[NP_ARRAY[NP_INT]] = times % ntime_input

    ncol: Optional[NP_INT] = NP_INT(xr_dpscream.sizes["ncol"]) # No. columns
    nlev: NP_INT = NP_INT(xr_dpscream.sizes["lev"]) # No. levels (layers)
    nilev: NP_INT = NP_INT(xr_dpscream.sizes["ilev"]) # No. level interfaces (levels)

    ## Dimension sizes - RTE-RRTMGP-CPP+RT - Only ones that need to be renamed
    n_lay_z: Optional[NP_INT] = nlev # DP-SCREAM "levels" = RTE-RRTMGP-CPP+RT "layers"
    n_lev_z: Optional[NP_INT] = nilev # DP-SCREAM "ilevels" = RTE-RRTMGP-CPP+RT "levels"

    ## Store spatial grid for outputting to RTE-RRTMGP-CPP input file
    grid: dict = {}

    ### NOTE: The number of points in the horizontal and vertical acceleration grids "should"
    ### be between 1/10 and 1/20 of n_col_x, n_col_y, n_col_z
    ### NOTE: These are the time-independent quantities
    ngrid_x: NP_INT = NP_INT(np.ceil(n_col_x / 10))
    ngrid_y: NP_INT = NP_INT(np.ceil(n_col_y / 10))
    ngrid_z: NP_INT = NP_INT(np.ceil(n_lay_z / 10))

    grid["x"] = x
    grid["xh"] = xh

    grid["y"] = y
    grid["yh"] = yh

    grid["ngrid_x"] = ngrid_x
    grid["ngrid_y"] = ngrid_y
    grid["ngrid_z"] = ngrid_z


if __name__ == "__main__":
    main(sys.argv)
