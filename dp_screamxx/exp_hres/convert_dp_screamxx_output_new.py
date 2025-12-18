"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import os
import re
import sys

from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, NC_REAL, NC_DATASET, NC_VARIABLE, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, \
    g
from utils.dp_screamxx_fields import dpscream_3dfield_keys, dpscream_2dfield_keys
from utils.rte_rrtmgp_cpp_fields import rte_3dfield_keys, rte_2dfield_keys, \
    grid_dimensions, grid_descriptions, grid_units, \
    fields_dimensions, fields_descriptions, fields_units

# Script variables
prog_name: str = "convert_dpscream_output"
prog_desc: str = "Converts DP-SCREAM output to RTE-RRTMGP-CPP+RT input."

def main(argv):
    # MPI Communicator info
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())

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
    file_ext: re.Pattern = re.compile("\\.nc")
    rte_rrtmgp_cpp_file_name_root: str = file_ext.sub("", os.path.basename(dpscream_file_path))
    rte_rrtmgp_cpp_file_path_root: str = os.path.join(rte_rrtmgp_cpp_dir_path, rte_rrtmgp_cpp_file_name_root)

    coarse_factors: NP_ARRAY[NP_INT] = np.array([2, 4], dtype = NP_INT)
    szas: NP_ARRAY[NP_REAL] = np.array([0., 85.], dtype = NP_REAL)

    # Root rank gets original horizontal grid
    sort_mask: Optional[NP_ARRAY[NP_INT]]
    grids: Optional[dict]
    ntime: Optional[NP_INT] = None
    if l_rank == MPI_ROOT:
        xr_dpscream: XR_DATASET = xr.open_dataset(dpscream_file_path,
            engine = "netcdf4")

        # Get number of time-steps
        ntime = NP_INT(xr_dpscream.sizes["time"])

        sort_mask: Optional[NP_ARRAY[NP_INT]] = get_sort_mask(xr_dpscream)
        grids = {}
        grids["01"] = get_grid_01(xr_dpscream, sort_mask)

        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            coarse_factor_str: str = "{:02}".format(coarse_factor)
            grids[coarse_factor_str] = coarsen_grid(grids["01"], coarse_factor)
    else:
        sort_mask = None
        grids = None

    ntime = comm.bcast(ntime, root = MPI_ROOT)
    l_grids: dict = bcast_grids(grids, comm)

    # NOTE: For now, go through all time-steps
    tt: int
    for tt in range(0, ntime):
        fields: Optional[dict]
        if l_rank == MPI_ROOT:
            fields = {"01" : {}}
            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "{:02}".format(coarse_factor)
                fields[coarse_factor_str] = {}

        ## Set unspecified fields
        unspecified_fields: dict = set_unspecified_fields(grids, comm)
        coarse_factor_str: str
        for coarse_factor_str in fields.keys():
            fields[coarse_factor_str] = {**fields[coarse_factor_str], **unspecified_fields[coarse_factor_str]}

        ## Interpolate 3D fields
        ii: int
        for ii in range(0, len(dpscream_3dfield_keys)):
            dpscream_field_key: str = dpscream_3dfield_keys[ii]
            rte_field_key: str = rte_3dfield_keys[ii]
            field: dict = interp_3dfield(xr_dpscream, dpscream_field_key, rte_field_key,
                sort_mask, grids, l_grids, tt, comm)
            coarse_factor_str: str
            for coarse_factor_str in field.keys():
                fields[coarse_factor_str] = {**fields[coarse_factor_str], **field[coarse_factor_str]}

        ## Interpolate 2D fields
        ii: int
        for ii in range(0, len(dpscream_2dfield_keys)):
            dpscream_field_key: str = dpscream_2dfield_keys[ii]
            rte_field_key: str = rte_2dfield_keys[ii]
            field: dict = interp_2dfield(xr_dpscream, dpscream_field_key, rte_field_key,
                sort_mask, grids, l_grids, tt, comm)
            coarse_factor_str: str
            for coarse_factor_str in field.keys():
                fields[coarse_factor_str] = {**fields[coarse_factor_str], **field[coarse_factor_str]}

        save_rte_rrtmgp_cpp_input(grids, fields, tt, rte_rrtmgp_cpp_file_path_root, comm, szas)

def get_sort_mask(xr_dpscream: XR_DATASET) -> NP_ARRAY[NP_INT]:
    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)).astype(NP_INT) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    return sort_mask

def get_grid_01(xr_dpscream: XR_DATASET, sort_mask: NP_ARRAY[NP_INT]) -> dict:
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

    z_min: NP_REAL = NP_REAL(xr_dpscream["z_mid"].isel(lev=[-1]).max()) # Lowest RTE level altitude on regular grid; [m]
    z_max: NP_REAL = NP_REAL(xr_dpscream["z_mid"].isel(lev=[0]).min()) # Highest RTE level altitude on regular grid; [m]

    z_lev: NP_ARRAY[NP_REAL] = np.linspace(z_min, z_max, nlev, dtype = NP_REAL) # Regularly-spaced RTE levels [m]; (nlev)
    z_lay: NP_ARRAY[NP_REAL] = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced RTE layers [m]; (nlay)

    ### NOTE: The number of points in the vertical acceleration grid "should"
    ### be between 1/10 and 1/20 of nlay
    ngrid_z: NP_INT = NP_INT(np.ceil(nlay / 10))

    ## Wavelength info
    n_bnd_sw: NP_INT = NP_INT(xr_dpscream.sizes["swband"])
    n_bnd_lw: NP_INT = NP_INT(xr_dpscream.sizes["lwband"])


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

    grid["n_bnd_sw"] = n_bnd_sw
    grid["n_bnd_lw"] = n_bnd_lw

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

    grid_coarse["n_bnd_sw"] = grid["n_bnd_sw"]
    grid_coarse["n_bnd_lw"] = grid["n_bnd_lw"]

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

        # Store communication values in each loal grid
        l_grids[coarse_str]["l_counts_x"] = l_counts
        l_grids[coarse_str]["l_displs_x"] = l_displs

    return l_grids

def interp_3dfield(xr_dpscream: XR_DATASET, dpscream_field_key: str,
    rte_field_key: str, sort_mask: NP_ARRAY[NP_INT], grids: dict, 
    l_grids: dict, tt: int, comm: MPI_COMM, interp_method: str = "nearest") -> NP_ARRAY[NP_REAL]:
    field_out: dict = {}

    l_rank: NP_INT = NP_INT(comm.Get_rank())

    z_src: Optional[NP_ARRAY[NP_REAL]]
    field_src: Optional[NP_ARRAY[NP_REAL]]
    field_min: Optional[NP_REAL]
    field_max: Optional[NP_REAL]
    # Root Rank reads input file, constructs full field and Scatterv
    if l_rank == MPI_ROOT:
        ## NOTE: Only using DP-SCREAM level interface (RTE-RRTMGP-CPP+RT layer) values
        if dpscream_field_key in xr_dpscream.keys(): # Only have values at midpoints
            field_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream[dpscream_field_key].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
            z_src = xr_dpscream["z_mid"].isel(time = tt, ncol = sort_mask, lev = slice(-1, None, -1)).values.astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)
                    
        else: # Should have values and midpoints and interfaces
            dpscream_field_key_mid: str = dpscream_field_key + "_mid"

            ## We should always have fields values at layer midpoints
            ## Unless we don't, then this needs to be fixed
            assert(dpscream_field_key_mid in xr_dpscream.keys())
            field_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream[dpscream_field_key_mid].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
            z_src = xr_dpscream["z_mid"].isel(time = tt, ncol = sort_mask, lev = slice(-1, None, -1)).values.astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)

        ## Exceptions - Do in serial for now
        if rte_field_key in ["dei"]: # DP-SCREAM has rei, RTE-RRTMGP-CPP has dei
            field_src = 2. * field_src
        elif rte_field_key in ["lwp", "iwp"]: # Derived from multiple quantities
            p_int: NP_ARRAY[NP_REAL] = \
                xr_dpscream["p_int"].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Pressure at layer interfaces [Pa]; (ncol, n_lev_z)
            dp: NP_ARRAY[NP_REAL] = p_int[:,1:] - p_int[:,:-1] # Layer pressure thickness [Pa]; (ncol, n_lay_z)

            field_src = field_src * dp / g

        ## Get field min and max
        ## Exceptions
        if rte_field_key in ["rel"]: # Between 2.5 μm and 21.5 μm
            field_min = NP_REAL(2.5)
            field_max = NP_REAL(21.5)
        elif rte_field_key in ["dei"]: # Between 10. μm and 180. μm
            field_min = NP_REAL(10.)
            field_max = NP_REAL(180.)
        else:
            field_min = field_src.min()
            field_max = field_src.max()

        nx: NP_INT = grids["01"]["nx"]
        ny: NP_INT = grids["01"]["ny"]
        nz: NP_INT = NP_INT(field_src.shape[1])

        z_src = z_src.reshape(nx, ny, nz)
        field_src = field_src.reshape(nx, ny, nz)
    else:
        z_src = None
        field_src = None
        field_min = None
        field_max = None

    # Scatterv the original field
    l_nx_src: NP_INT = l_grids["01"]["nx"]
    l_ny_src: NP_INT = l_grids["01"]["ny"]
    l_nlay_src: NP_INT = l_grids["01"]["lay"]

    l_counts_src: list[NP_INT] = l_grids["01"]["l_counts_x"] * l_ny_src * l_nlay_src
    l_displs_src: list[NP_INT] = l_grids["01"]["l_displs_x"] * l_ny_src * l_nlay_src

    l_field_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nlay_src], dtype = NP_REAL) # NOTE: ASSUME only using layer midpoint values for field

    field_min = comm.bcast(field_min, root = MPI_ROOT)
    field_max = comm.bcast(field_max, root = MPI_ROOT)
    comm.Scatterv([field_src, l_counts_src, l_displs_src, MPI_REAL], l_field_src, root = MPI_ROOT)

    # Get source grid - points to interpolate from
    l_x_src: NP_ARRAY[NP_REAL] = l_grids["01"]["x"]
    l_y_src: NP_ARRAY[NP_REAL] = l_grids["01"]["y"]
    l_z_src: NP_ARRAY[NP_REAL] = comm.bcast(z_src, root = MPI_ROOT)
    l_nz: NP_INT = l_z_src.shape[2]

    l_XX_src: NP_ARRAY[NP_REAL]
    l_YY_src: NP_ARRAY[NP_REAL]
    l_XX_src, l_YY_src = np.meshgrid(l_x_src, l_y_src, indexing = "ij")
    l_XX_src = np.tile(np.expand_dims(l_XX_src, axis = 2), (1, 1, l_nz))
    l_YY_src = np.tile(np.expand_dims(l_YY_src, axis = 2), (1, 1, l_nz))

    l_pts_src: NP_ARRAY[NP_REAL] = \
        np.stack([l_XX_src.flatten(), l_YY_src.flatten(), l_z_src.flatten()],
            axis = 1)

    # Coarsen the field as necessary
    for coarse_str in l_grids.keys():
        field_out[coarse_str]: dict = {}
        # Get target layer grid - points to interpolate to
        l_ny_tgt: NP_INT = l_grids[coarse_str]["ny"]
        l_nlay_tgt: NP_INT = l_grids[coarse_str]["lay"]

        l_counts_lay_tgt: list[NP_INT] = l_grids[coarse_str]["l_counts_x"] * l_ny_tgt * l_nlay_tgt
        l_displs_lay_tgt: list[NP_INT] = l_grids[coarse_str]["l_displs_x"] * l_ny_tgt * l_nlay_tgt

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids[coarse_str]["y"]
        l_z_lay_tgt: NP_ARRAY[NP_REAL] = l_grids[coarse_str]["z_lay"]

        l_XX_lay_tgt, l_YY_lay_tgt, l_ZZ_lay_tgt = \
            np.meshgrid(l_x_tgt, l_y_tgt, l_z_lay_tgt, indexing = "ij")
        l_pts_lay_tgt: NP_ARRAY[NP_REAL] = \
            np.stack([l_XX_lay_tgt.flatten(), l_YY_lay_tgt.flatten(), l_ZZ_lay_tgt.flatten()], 
                axis = 1)

        ## Interpolate the values to regular vertical layers, and limit them
        l_field_lay_tgt: NP_ARRAY[NP_REAL] = \
            griddata(l_pts_src, l_field_src.flatten(), l_pts_lay_tgt,
                method = interp_method)
        l_field_lay_tgt[l_field_lay_tgt < field_min] = field_min
        l_field_lay_tgt[l_field_lay_tgt > field_max] = field_max

        # Reconstruct the full field
        field_lay_tgt: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            nx_tgt: NP_INT = grids[coarse_str]["nx"]
            ny_tgt: NP_INT = grids[coarse_str]["ny"]
            nlay_tgt: NP_INT = grids[coarse_str]["lay"]

            field_lay_tgt = np.empty([nx_tgt, ny_tgt, nlay_tgt], dtype = NP_REAL)

        comm.Gatherv(l_field_lay_tgt,
            [field_lay_tgt, l_counts_lay_tgt, l_displs_lay_tgt, MPI_REAL],
            root = MPI_ROOT)

        if l_rank == MPI_ROOT:
            field_lay_tgt = np.reshape(field_lay_tgt, (nx_tgt, ny_tgt, nlay_tgt)) # (nx, ny, nlay)
            field_lay_tgt = np.transpose(field_lay_tgt, axes = (2, 1, 0)) # (nlay, ny, nx)

        ## Some fields need to be interpolated to regular vertical levels, too
        if dpscream_field_key in ["p", "T"]:
            # Get target level grid - points to interpolate to
            l_nlev_tgt: NP_INT = l_grids[coarse_str]["lev"]
            l_z_lev_tgt: NP_ARRAY[NP_REAL] = l_grids[coarse_str]["z_lev"]
            l_counts_lev_tgt: list[NP_INT] = l_grids[coarse_str]["l_counts_x"] * l_ny_tgt * l_nlev_tgt
            l_displs_lev_tgt: list[NP_INT] = l_grids[coarse_str]["l_displs_x"] * l_ny_tgt * l_nlev_tgt

            l_XX_lev_tgt, l_YY_lev_tgt, l_ZZ_lev_tgt = \
                np.meshgrid(l_x_tgt, l_y_tgt, l_z_lev_tgt, indexing = "ij")
            l_pts_lev_tgt: NP_ARRAY[NP_REAL] = \
                np.stack([l_XX_lev_tgt.flatten(), l_YY_lev_tgt.flatten(), l_ZZ_lev_tgt.flatten()], 
                    axis = 1)
            
            ## Interpolate the values to regular vertical levels, and limit them
            l_field_lev_tgt: NP_ARRAY[NP_REAL] = \
                griddata(l_pts_src, l_field_src.flatten(), l_pts_lev_tgt,
                    method = interp_method)
            l_field_lev_tgt[l_field_lev_tgt < field_min] = field_min
            l_field_lev_tgt[l_field_lev_tgt > field_max] = field_max
        
            # Reconstruct the full field
            field_lev_tgt: Optional[NP_ARRAY[NP_REAL]] = None
            if l_rank == MPI_ROOT:
                nlev_tgt: NP_INT = grids[coarse_str]["lev"]
                field_lev_tgt = np.empty([nx_tgt, ny_tgt, nlev_tgt], dtype = NP_REAL)
            
            comm.Gatherv(l_field_lev_tgt,
                [field_lev_tgt, l_counts_lev_tgt, l_displs_lev_tgt, MPI_REAL],
                root = MPI_ROOT)

            if l_rank == MPI_ROOT:
                field_lev_tgt = np.reshape(field_lev_tgt, (nx_tgt, ny_tgt, nlev_tgt)) # (nx, ny, nlev)
                field_lev_tgt = np.transpose(field_lev_tgt, axes = (2, 1, 0)) # (nlev, ny, nx)

        if l_rank == MPI_ROOT:
            ## Exceptions
            if rte_field_key in ["rh", "q", "lwp", "iwp", "rel", "dei",
                "vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o",
                "vmr_o2", "vmr_o3"]:
                field_out[coarse_str][rte_field_key] = field_lay_tgt
            else:
                rte_field_key_lay: str = rte_field_key + "_lay"
                field_out[coarse_str][rte_field_key_lay] = field_lay_tgt

                if dpscream_field_key in ["p", "T"]:
                    rte_field_key_lev: str = rte_field_key + "_lev"
                    field_out[coarse_str][rte_field_key_lev] = field_lev_tgt

    return field_out

def interp_2dfield(xr_dpscream: XR_DATASET, dpscream_field_key: str,
    rte_field_key: str, sort_mask: NP_ARRAY[NP_INT], grids: dict, 
    l_grids: dict, tt: int, comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    field_out: dict = {}

    l_rank: NP_INT = NP_INT(comm.Get_rank())

    field_src: Optional[NP_ARRAY[NP_REAL]]
    field_min: Optional[NP_REAL]
    field_max: Optional[NP_REAL]
    # Root Rank reads input file, constructs full field and Scatterv
    if l_rank == MPI_ROOT:
        assert(dpscream_field_key in xr_dpscream.keys())
        field_src: NP_ARRAY[NP_REAL] = \
            xr_dpscream[dpscream_field_key].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field; (ncol)
            
        ## Exceptions - Do in serial for now
        if rte_field_key in ["t_sfc", "mu0"]: # In case fill values are unreasonable
            if rte_field_key in ["t_sfc"]: # Between 0 K and 2300 K (max natural temperature on Earth)
                field_min: NP_REAL = NP_REAL(0.0)
                field_max: NP_REAL = NP_REAL(2300.0)
            elif rte_field_key in ["mu0"]: # Between -1.0 and 1.0
                field_min: NP_REAL = NP_REAL(-1.0)
                field_max: NP_REAL = NP_REAL(1.0)
            
            field_src[field_src > field_max] = field_max
            field_src[field_src < field_min] = field_min

        nx: NP_INT = grids["01"]["nx"]
        ny: NP_INT = grids["01"]["ny"]

        field_src = field_src.reshape(nx, ny)
    else:
        field_src = None
        field_min = None
        field_max = None

    # Scatterv the original field
    l_nx_src: NP_INT = l_grids["01"]["nx"]
    l_ny_src: NP_INT = l_grids["01"]["ny"]

    l_counts_src: list[NP_INT] = l_grids["01"]["l_counts_x"] * l_ny_src
    l_displs_src: list[NP_INT] = l_grids["01"]["l_displs_x"] * l_ny_src

    l_field_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src], dtype = NP_REAL)

    field_min = comm.bcast(field_min, root = MPI_ROOT)
    field_max = comm.bcast(field_max, root = MPI_ROOT)
    comm.Scatterv([field_src, l_counts_src, l_displs_src, MPI_REAL], l_field_src, root = MPI_ROOT)

    # Get source grid - points to interpolate from
    l_x_src: NP_ARRAY[NP_REAL] = l_grids["01"]["x"]
    l_y_src: NP_ARRAY[NP_REAL] = l_grids["01"]["y"]

    l_XX_src: NP_ARRAY[NP_REAL]
    l_YY_src: NP_ARRAY[NP_REAL]
    l_XX_src, l_YY_src = np.meshgrid(l_x_src, l_y_src, indexing = "ij")

    l_pts_src: NP_ARRAY[NP_REAL] = \
        np.stack([l_XX_src.flatten(), l_YY_src.flatten()], axis = 1)

    # Coarsen the field as necessary
    for coarse_str in l_grids.keys():
        field_out[coarse_str]: dict = {}
        # Get target layer grid - points to interpolate to
        l_ny_tgt: NP_INT = l_grids[coarse_str]["ny"]

        l_counts_tgt: list[NP_INT] = l_grids[coarse_str]["l_counts_x"] * l_ny_tgt
        l_displs_tgt: list[NP_INT] = l_grids[coarse_str]["l_displs_x"] * l_ny_tgt

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids[coarse_str]["y"]

        l_XX_tgt, l_YY_tgt = np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
        l_pts_tgt: NP_ARRAY[NP_REAL] = \
            np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], axis = 1)

        ## Interpolate the values to regular vertical layers, and limit them
        l_field_tgt: NP_ARRAY[NP_REAL] = \
            griddata(l_pts_src, l_field_src.flatten(), l_pts_tgt,
                method = interp_method)
        l_field_tgt[l_field_tgt < field_min] = field_min
        l_field_tgt[l_field_tgt > field_max] = field_max

        # Reconstruct the full field
        field_tgt: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            nx_tgt: NP_INT = grids[coarse_str]["nx"]
            ny_tgt: NP_INT = grids[coarse_str]["ny"]

            field_tgt = np.empty([nx_tgt, ny_tgt], dtype = NP_REAL)

        comm.Gatherv(l_field_tgt, [field_tgt, l_counts_tgt, l_displs_tgt, MPI_REAL],
            root = MPI_ROOT)

        if l_rank == MPI_ROOT:
            field_tgt = np.reshape(field_tgt, (nx_tgt, ny_tgt)) # (nx, ny)
            field_tgt = np.transpose(field_tgt, axes = (1, 0)) # (ny, nx)

        if l_rank == MPI_ROOT:
            field_out[coarse_str][rte_field_key] = field_tgt

    return field_out

def set_unspecified_fields(grids: dict, comm: MPI_COMM) -> dict:
    """
    Set fields not specified by the DP-SCREAM output.
    """
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    fields_out: dict = {}

    if l_rank == MPI_ROOT:
        coarse_factor_str: str
        for coarse_factor_str in grids.keys():
            fields_out[coarse_factor_str]: dict = {}

            nx: NP_INT = grids[coarse_factor_str]["nx"]
            ny: NP_INT = grids[coarse_factor_str]["ny"]
            nlay: NP_INT = grids[coarse_factor_str]["lay"]
            n_bnd_sw: NP_INT = grids[coarse_factor_str]["n_bnd_sw"]
            n_bnd_lw: NP_INT = grids[coarse_factor_str]["n_bnd_lw"]

            ## Longwave boundary conditions
            fields_out[coarse_factor_str]["emis_sfc"]: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_lw), dtype = NP_REAL)
            
            fields_out[coarse_factor_str]["sfc_alb_dir"]: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07 
            fields_out[coarse_factor_str]["sfc_alb_dif"]: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07

            fields_out[coarse_factor_str]["tsi"]: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx), dtype = NP_REAL) * 551.58

            fields_out[coarse_factor_str]["azi"]: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx), dtype = NP_REAL) * 0.0 

            ## Set quantities not expected to be set in the DP-SCREAM output
            unexpected_keys: list[str] = ["vmr_ccl4", "vmr_cfc11", "vmr_cfc12",
                "vmr_cfc22", "vmr_hfc143a", "vmr_hfc125", "vmr_hfc32", "vmr_hfc23",
                "vmr_hfc134a", "vmr_cf4", "vmr_no2", "aermr01", "aermr02",
                "aermr03", "aermr04", "aermr05", "aermr06", "aermr07", "aermr08",
                "aermr09", "aermr10", "aermr11"]
            
            for key in unexpected_keys:
                fields_out[coarse_factor_str][key]: NP_ARRAY[NP_REAL] = \
                    np.zeros((nlay, ny, nx), dtype = NP_REAL)
                
    return fields_out

def save_rte_rrtmgp_cpp_input(grids: dict, fields: dict, tt: NP_INT,
    file_path_root: str, comm: MPI_COMM, szas: Optional[NP_ARRAY[NP_REAL]] = None):

    l_rank: NP_INT = NP_INT(comm.Get_rank())

    if l_rank == MPI_ROOT:
        time_str: str = ".{:03d}".format(tt)

        coarse_factor_str: str
        for coarse_factor_str in grids.keys():
            lr_str: str = ".lr_" + coarse_factor_str
            nx: NP_INT = grids[coarse_factor_str]["nx"]
            ny: NP_INT = grids[coarse_factor_str]["ny"]
            if szas is not None:
                sza: NP_REAL
                for sza in szas:
                    sza_str: str = ".{:03.0f}".format(sza)

                    sza_rad: NP_REAL = np.deg2rad(sza)
                    fields[coarse_factor_str]["mu0"]: NP_ARRAY[NP_REAL] = \
                        np.zeros([ny, nx], dtype = NP_REAL) + np.cos(sza_rad)
                    
                    file_path: str = file_path_root + time_str + sza_str + lr_str + ".in.nc"
                    
                    write_rte_input(grids, fields, coarse_factor_str, file_path)
            else:
                file_path: str = file_path_root + time_str + lr_str + ".in.nc"

                write_rte_input(grids, fields, coarse_factor_str, file_path)
                    
def write_rte_input(grids: dict, fields: dict, coarse_factor_str: str,
    file_path: str):

    l_grid: dict = grids[coarse_factor_str]
    l_fields: dict = fields[coarse_factor_str]

    nc_file: NC_DATASET = nc.Dataset(file_path, mode = "w",
        datamodel = "NETCDF4", clobber = True)

    nc_file.createDimension("x", l_grid["nx"])
    nc_file.createDimension("y", l_grid["ny"])
    nc_file.createDimension("lay", l_grid["lay"])
    nc_file.createDimension("lev", l_grid["lev"])
    nc_file.createDimension("z", l_grid["lay"])
    nc_file.createDimension("xh", l_grid["nx"] + 1)
    nc_file.createDimension("yh", l_grid["ny"] + 1)
    nc_file.createDimension("zh", l_grid["lev"])
    nc_file.createDimension("band_sw", l_grid["n_bnd_sw"])
    nc_file.createDimension("band_lw", l_grid["n_bnd_lw"])

    ## Spatial grid
    for rte_grid_key in grid_dimensions.keys():
        field: NP_ARRAY[NP_REAL] = l_grid[rte_grid_key]
        field_dimensions: str | tuple[Optional[str]] = grid_dimensions[rte_grid_key]
        field_description: str = grid_descriptions[rte_grid_key]
        field_units: str = grid_units[rte_grid_key]

        nc_field: NC_VARIABLE = nc_file.createVariable(rte_grid_key, NC_REAL, field_dimensions)
        nc_field.description: str = field_description
        nc_field.units: str = field_units
        nc_field[...]: NP_ARRAY[NP_REAL] = field

    ## Fields
    for rte_field_key in fields_dimensions:
        field: NP_ARRAY[NP_REAL] = l_fields[rte_field_key][:]

        field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
        field_description: str = fields_descriptions[rte_field_key]
        field_units: str = fields_units[rte_field_key]

        nc_field: NC_VARIABLE = nc_file.createVariable(rte_field_key, NC_REAL, field_dimensions)
        nc_field.description: str = field_description
        nc_field.units: str = field_units
        nc_field[...]: NP_ARRAY[NP_REAL] = field

    nc_file.close()

if __name__ == "__main__":
    main(sys.argv)
