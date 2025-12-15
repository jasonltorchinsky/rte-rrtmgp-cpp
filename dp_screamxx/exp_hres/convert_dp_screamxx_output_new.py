"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import os
import sys

from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET
from utils.dp_screamxx_fields import dpscream_3dfield_keys, dpscream_2dfield_keys
from utils.rte_rrtmgp_cpp_fields import rte_3dfield_keys, rte_2dfield_keys

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

    dpscream_file_path: str = os.path.normpath(args.dpscream_file_path[0])
    rte_rrtmgp_cpp_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_dir_path[0])

    coarse_factors: NP_ARRAY[NP_INT] = np.array([2, 4], dtype = NP_INT)

    # Root rank gets original horizontal grid
    ntime: Optional[NP_INT] = None
    grids: Optional[dict]
    if l_rank == MPI_ROOT:
        xr_dpscream: XR_DATASET = xr.open_dataset(dpscream_file_path,
            engine = "netcdf4")

        # Get number of time-steps
        ntime = NP_INT(xr_dpscream.sizes["time"])

        sort_mask: NP_ARRAY[NP_INT] = get_sort_mask(xr_dpscream)
        grids = {}
        grids["01"] = get_grid_01(xr_dpscream, sort_mask)

        for coarse_factor in coarse_factors:
            grid_str: str = "{:02}".format(coarse_factor)
            grids[grid_str] = coarsen_grid(grids["01"], coarse_factor)
    else:
        grids = None

    ntime = comm.bcast(ntime, root = MPI_ROOT)
    l_grids: dict = bcast_grids(grids, comm)

    # NOTE: For now, go through all time-steps
    tt: int
    for tt in range(0, ntime):
        # MPI_ROOT Scattervs the fields
        if l_rank == MPI_ROOT:


    if l_rank == MPI_ROOT:
        print(l_rank, grids["02"]["x"])
    comm.barrier()

    ii: int
    for ii in range(0, comm_size):
        if ii == l_rank:
            print(ii, l_grids["02"]["x"])
        comm.barrier()


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

def get_sort_mask(xr_dpscream: XR_DATASET) -> NP_ARRAY[NP_INT]:
    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)).astype(NP_INT) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    return sort_mask

def get_grid_01(xr_dpscream: XR_DATASET, sort_mask: NP_ARRAY[NP_INT]) -> dict:
    # HORIZONTAL GRID

    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    nx: NP_INT = NP_INT(np.unique(lon).size) # No. columns in x
    ny: NP_INT = NP_INT(np.unique(lat).size) # No. columns in y
    cols: NP_ARRAY[NP_REAL] = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(nx, ny, 2)

    ## Construct the horizontal grids
    ### NOTE: Assume that horizontal grids are regularly-spaced.
    x: NP_ARRAY[NP_REAL] = (cols[:,:,0])[0,:] # x-midpoints of each column [m]; (nx)
    dx: NP_REAL = x[1] - x[0]
    xh: NP_ARRAY[NP_REAL] = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-interfaces of each column [m]; (nx + 1)

    y: NP_ARRAY[NP_REAL] = (cols[:,:,1])[:,0] # y-midpoints of each column [m]; (ny)
    dy: NP_REAL = y[1] - y[0]
    yh: NP_ARRAY[NP_REAL] = np.append(y - (dy / 2.), x[-1] + (dy / 2.)) # y-interfaces of each column [m]; (ny + 1)

    ### NOTE: The number of points in the horizontal acceleration grid "should"
    ### be between 1/10 and 1/20 of nx, ny
    ngrid_x: NP_INT = NP_INT(np.ceil(nx / 10))
    ngrid_y: NP_INT = NP_INT(np.ceil(ny / 10))

    # VERTICAL GRID
    # NOTE: Here we get the uniform, time-independent vertical grid that we will
    # remap values to
    nlay: NP_INT = NP_INT(xr_dpscream.sizes["lev"]) # No. DP-SCREAM levels (RTE layers)
    nlev: NP_INT = NP_INT(xr_dpscream.sizes["ilev"]) # No. DP-SCREAM level interfaces (RTE levels)

    z_min: NP_REAL = NP_REAL(xr_dpscream["z_mid"].min()) # Lowest RTE level altitude on regular grid; [m]
    z_max: NP_REAL = NP_REAL(xr_dpscream["z_mid"].max()) # Highest RTE level altitude on regular grid; [m]

    z_lev: NP_ARRAY[NP_REAL] = np.linspace(z_min, z_max, nlev, dtype = NP_REAL) # Regularly-spaced RTE levels [m]; (nlev)
    z_lay: NP_ARRAY[NP_REAL] = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced RTE layers [m]; (nlay)

    ### NOTE: The number of points in the vertical acceleration grid "should"
    ### be between 1/10 and 1/20 of nlay
    ngrid_z: NP_INT = NP_INT(np.ceil(nlay / 10))

    ## Spatial RTE-RRTMGP-CPP grid
    grid: dict = {}

    grid["nx"] = nx
    grid["x"] = x
    grid["xh"] = xh

    grid["ny"] = ny
    grid["y"] = y
    grid["yh"] = yh

    grid["lay"] = nlay
    grid["lev"] = nlev
    grid["z"] = z_lay
    grid["zh"] = z_lev
    grid["z_lay"] = z_lay
    grid["z_lev"] = z_lev

    grid["ngrid_x"] = ngrid_x
    grid["ngrid_y"] = ngrid_y
    grid["ngrid_z"] = ngrid_z

    return grid

def coarsen_grid(grid: dict, coarse_factor: NP_INT) -> dict:
    nx_coarse: NP_INT = grid["nx"] // coarse_factor
    ngrid_x_coarse: NP_INT = NP_INT(np.ceil(nx_coarse / 10))
    xh_min: NP_REAL = grid["xh"].min()
    xh_max: NP_REAL = grid["xh"].max()
    xh_coarse: NP_ARRAY[NP_REAL] = np.linspace(xh_min, xh_max, nx_coarse + 1,
        dtype = NP_REAL)
    x_coarse: NP_ARRAY[NP_REAL] = (xh_coarse[:-1] + xh_coarse[1:]) / 2.

    ny_coarse: NP_INT = grid["ny"] // coarse_factor
    ngrid_y_coarse: NP_INT = NP_INT(np.ceil(ny_coarse / 10))
    yh_min: NP_REAL = grid["yh"].min()
    yh_max: NP_REAL = grid["yh"].max()
    yh_coarse: NP_ARRAY[NP_REAL] = np.linspace(yh_min, yh_max, ny_coarse + 1,
        dtype = NP_REAL)
    y_coarse: NP_ARRAY[NP_REAL] = (yh_coarse[:-1] + yh_coarse[1:]) / 2.

    ## Spatial RTE-RRTMGP-CPP grid
    grid_coarse: dict = {}

    grid_coarse["nx"] = nx_coarse
    grid_coarse["x"] = x_coarse
    grid_coarse["xh"] = xh_coarse

    grid_coarse["ny"] = ny_coarse
    grid_coarse["y"] = y_coarse
    grid_coarse["yh"] = yh_coarse

    grid_coarse["lay"] = grid["lay"]
    grid_coarse["lev"] = grid["lev"]
    grid_coarse["z"] = grid["z"]
    grid_coarse["zh"] = grid["zh"]
    grid_coarse["z_lay"] = grid["z_lay"]
    grid_coarse["z_lev"] = grid["z_lev"]

    grid_coarse["ngrid_x"] = ngrid_x_coarse
    grid_coarse["ngrid_y"] = ngrid_y_coarse
    grid_coarse["ngrid_z"] = grid["ngrid_z"]

    return grid_coarse

def bcast_grids(grids: Optional[dict], comm: MPI_COMM) -> dict:
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    # Broadcast vertical grid info, which is the same across all grids
    lay: Optional[NP_INT] = None
    lev: Optional[NP_INT] = None
    z_lay: Optional[NP_ARRAY[NP_REAL]] = None
    z_lev: Optional[NP_ARRAY[NP_REAL]] = None
    ngrid_z: Optional[NP_INT] = None
    if l_rank == MPI_ROOT:
        lay = grids["01"]["lay"]
        lev = grids["01"]["lev"]
        z_lay = np.copy(grids["01"]["z_lay"])
        z_lev = np.copy(grids["01"]["z_lev"])
        ngrid_z = grids["01"]["ngrid_z"]
    lay = comm.bcast(lay, root = MPI_ROOT)
    lev = comm.bcast(lev, root = MPI_ROOT)
    z_lay = comm.bcast(z_lay, root = MPI_ROOT)
    z_lev = comm.bcast(z_lev, root = MPI_ROOT)
    ngrid_z = comm.bcast(ngrid_z, root = MPI_ROOT)

    # Broadcast coarse_strs to setup l_grids
    coarse_strs: Optional[list[str]] = None
    if l_rank == MPI_ROOT:
        coarse_strs = list(grids.keys())
    coarse_strs = comm.bcast(coarse_strs, root = MPI_ROOT)

    l_grids: dict = {}
    coarse_str: str
    for coarse_str in coarse_strs:
        l_grids[coarse_str] = {}

        # Scatterv the x-grid
        g_nx: Optional[NP_INT] = None
        if l_rank == MPI_ROOT:
            g_nx = grids[coarse_str]["nx"]
        g_nx = comm.bcast(g_nx, root = MPI_ROOT)
        
        l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
        l_counts[0] = (g_nx // comm_size + int(0 < (g_nx % comm_size)))

        l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)

        ii: int
        for ii in range(1, comm_size):
            l_counts[ii] = g_nx // comm_size + int(ii < (g_nx % comm_size))
            l_displs[ii] = l_counts[ii - 1] + l_displs[ii - 1] - 1

        g_x: NP_ARRAY[NP_REAL] = np.empty(g_nx, dtype = NP_REAL)
        l_x: NP_ARRAY[NP_REAL] = np.empty(l_counts[l_rank], dtype = NP_REAL)
        if l_rank == MPI_ROOT:
            g_x = np.copy(grids[coarse_str]["x"])
        comm.Scatterv([g_x, l_counts, l_displs, MPI_REAL], l_x, root = MPI_ROOT)
        dx: NP_REAL = l_x[1] - l_x[0]
        l_xh: NP_ARRAY[NP_REAL] = np.append(l_x - dx / 2., l_x[-1] + dx / 2.)

        l_grids[coarse_str]["nx"] = l_x.size
        l_grids[coarse_str]["x"] = l_x
        l_grids[coarse_str]["xh"] = l_xh

        # Broadcast the other values
        ny: Optional[NP_INT] = None
        y: Optional[NP_ARRAY[NP_REAL]] = None
        yh: Optional[NP_ARRAY[NP_REAL]] = None
        ngrid_y: Optional[NP_INT] = None
        if l_rank == MPI_ROOT:
            ny = grids[coarse_str]["ny"]
            y = np.copy(grids[coarse_str]["y"])
            yh = np.copy(grids[coarse_str]["yh"])
            ngrid_y = grids[coarse_str]["ngrid_y"]
        ny = comm.bcast(ny, root = MPI_ROOT)
        y = comm.bcast(y, root = MPI_ROOT)
        yh = comm.bcast(yh, root = MPI_ROOT)
        ngrid_y = comm.bcast(ngrid_y, root = MPI_ROOT)

        # Store the other values in l_grids
        l_grids[coarse_str]["ny"] = ny
        l_grids[coarse_str]["y"] = y
        l_grids[coarse_str]["yh"] = yh

        l_grids[coarse_str]["lay"] = lay
        l_grids[coarse_str]["lev"] = lev
        l_grids[coarse_str]["z"] = z_lay
        l_grids[coarse_str]["zh"] = z_lev
        l_grids[coarse_str]["z_lay"] = z_lay
        l_grids[coarse_str]["z_lev"] = z_lev

        l_grids[coarse_str]["ngrid_y"] = ngrid_y
        l_grids[coarse_str]["ngrid_z"] = ngrid_z

    return l_grids
        


if __name__ == "__main__":
    main(sys.argv)
