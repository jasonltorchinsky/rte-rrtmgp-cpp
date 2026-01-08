"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import ast
import os
import re
import sys

from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
from scipy.interpolate import griddata
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, \
    g
from utils.dp_screamxx_fields import dpscream_3dfield_keys, dpscream_2dfield_keys
from utils.rte_rrtmgp_cpp_fields import rte_3dfield_keys, rte_2dfield_keys, \
    grid_dimensions, grid_descriptions, grid_units, grid_dtypes, \
    fields_dimensions, fields_descriptions, fields_units, field_dtypes

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
        description = prog_desc
    )
    
    parser.add_argument("--dpscream_file_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to DP-SCREAM output."
    )
    
    parser.add_argument("--rte_rrtmgp_cpp_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Directory path for RTE-RRTMGP-CPP+RT input."
    )

    parser.add_argument("--coarse_factors",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = [None],
        help = "Factors by which to coarsen the horizontal grid."
    )

    parser.add_argument("--szas",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = None,
        help = "Solar zenith angles to create RTE-RRTMGP-CPP input for [degrees]."
    )

    parser.add_argument("--t0",
        action = "store",
        nargs = 1,
        type = int,
        required = False,
        default = None,
        help = "Initial time-step index to begin conversion at."
    )

    parser.add_argument("--tf",
        action = "store",
        nargs = 1,
        type = int,
        required = False,
        default = None,
        help = "Final time-step index to end at."
    )

    parser.add_argument("--times",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = None,
        help = "Time indexes to create RTE-RRTMGP-CPP input for [overwritten by t0, tf]."
    )
    
    args: argparse.Namespace = parser.parse_args()

    dpscream_file_path: str = os.path.normpath(args.dpscream_file_path[0])
    rte_rrtmgp_cpp_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_dir_path[0])
    coarse_factors: NP_ARRAY[NP_INT]
    if args.coarse_factors[0] is None:
        coarse_factors = np.array([2], dtype = NP_INT)
    else:
        coarse_factors = np.array(ast.literal_eval(args.coarse_factors[0]), dtype = NP_INT).flatten()
    szas: Optional[NP_ARRAY[NP_REAL]]
    if args.szas[0] is None:
        szas = None
    else:
        szas = np.array(ast.literal_eval(args.szas[0]), dtype = NP_REAL).flatten()
    t0: Optional[NP_INT]
    if args.t0 is None:
        t0 = None
    else:
        t0 = NP_INT(args.t0[0])
    tf: Optional[NP_INT]
    if args.tf is None:
        tf = None
    else:
        tf = NP_INT(args.tf[0])
    times: Optional[NP_ARRAY[NP_INT]]
    if args.times is not None:
        if ((args.times[0] is None) or ((t0 is not None) or (tf is not None))):
            times = None
        else:
            times = np.array(ast.literal_eval(args.times[0]), dtype = NP_INT).flatten()
    else:
        times = None

    interp_method: str = "nearest"

    file_ext: re.Pattern = re.compile("\\.nc")
    rte_rrtmgp_cpp_file_name_root: str = file_ext.sub("", os.path.basename(dpscream_file_path))
    rte_rrtmgp_cpp_file_path_root: str = os.path.join(rte_rrtmgp_cpp_dir_path, rte_rrtmgp_cpp_file_name_root)

    # Root rank gets original horizontal grid
    xr_dpscream: Optional[XR_DATASET]
    sort_mask: Optional[NP_ARRAY[NP_INT]]
    coords: Optional[dict]
    if l_rank == MPI_ROOT:
        xr_dpscream = xr.open_dataset(dpscream_file_path, engine = "netcdf4")

        # Get time-steps
        ntime_dpscream: NP_INT = NP_INT(xr_dpscream.sizes["time"])
        if t0 is not None:
            t0 = t0 % ntime_dpscream
        else:
            t0 = NP_INT(0)
        if tf is not None:
            tf = tf % ntime_dpscream
        else:
            tf = ntime_dpscream - 1

        assert(tf >= t0)

        if times is not None:
            times = np.sort(times % ntime_dpscream)
        else:
            times = np.arange(t0, tf + 1, dtype = NP_INT)

        sort_mask: Optional[NP_ARRAY[NP_INT]] = get_sort_mask(xr_dpscream)
        coords = {}
        coords["01"] = get_coords_01(xr_dpscream, sort_mask)

        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            coarse_factor_str: str = "{:02}".format(coarse_factor)
            coords[coarse_factor_str] = coarsen_coords(coords["01"], coarse_factor)
    else:
        xr_dpscream = None
        sort_mask = None
        coords = None
        times = None

    times = comm.bcast(times, root = MPI_ROOT)
    l_grids: dict = bcast_coords(coords, comm)

    # NOTE: For now, go through all time-steps
    tt: NP_INT
    for tt in times:
        fields: Optional[dict]
        if l_rank == MPI_ROOT:
            fields = {"01" : {}}
            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "{:02}".format(coarse_factor)
                fields[coarse_factor_str] = {}
        else:
            fields = None

        ## Set unspecified fields
        unspecified_fields: dict = set_unspecified_fields(coords, comm)
        if l_rank == MPI_ROOT:
            coarse_factor_str: str
            for coarse_factor_str in fields.keys():
                fields[coarse_factor_str] = {**fields[coarse_factor_str], **unspecified_fields[coarse_factor_str]}

        ## Interpolate 3D fields
        ii: int
        for ii in range(0, len(dpscream_3dfield_keys)):
            dpscream_field_key: str = dpscream_3dfield_keys[ii]
            rte_field_key: str = rte_3dfield_keys[ii]
            field_val_dict: dict = interp_3dfield(xr_dpscream, dpscream_field_key, rte_field_key,
                sort_mask, coords, l_grids, tt, comm, interp_method = interp_method)

            if l_rank == MPI_ROOT:
                coarse_factor_str: str
                for coarse_factor_str in field_val_dict.keys():
                    fields_val: dict = field_val_dict[coarse_factor_str]
                    for field_key in fields_val.keys():
                        field: list = (
                            fields_dimensions[field_key],
                            fields_val[field_key],
                            dict(description = fields_descriptions[field_key], units = fields_units[field_key])
                        )
                        fields[coarse_factor_str][field_key] = field

        ## Interpolate 2D fields
        ii: int
        for ii in range(0, len(dpscream_2dfield_keys)):
            dpscream_field_key: str = dpscream_2dfield_keys[ii]
            rte_field_key: str = rte_2dfield_keys[ii]
            field_val_dict: dict = interp_2dfield(xr_dpscream, dpscream_field_key, rte_field_key,
                sort_mask, coords, l_grids, tt, comm, interp_method = interp_method)

            if l_rank == MPI_ROOT:
                coarse_factor_str: str
                for coarse_factor_str in field_val_dict.keys():
                    fields_val: dict = field_val_dict[coarse_factor_str]
                    for field_key in fields_val.keys():
                        field: list = (
                            fields_dimensions[field_key],
                            fields_val[field_key],
                            dict(description = fields_descriptions[field_key], units = fields_units[field_key])
                        )
                        fields[coarse_factor_str][field_key] = field

        save_rte_rrtmgp_cpp_input(coords, fields, tt, rte_rrtmgp_cpp_file_path_root, comm, szas)

def get_sort_mask(xr_dpscream: XR_DATASET) -> NP_ARRAY[NP_INT]:
    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)).astype(NP_INT) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    return sort_mask

def get_coords_01(xr_dpscream: XR_DATASET, sort_mask: NP_ARRAY[NP_INT]) -> dict:
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

    ## Spatial RTE-RRTMGP-CPP coords
    coords: dict = dict(
        x = ("x", x, dict(description = grid_descriptions["x"], units = grid_units["x"])),
        xh = ("xh", xh, dict(description = grid_descriptions["xh"], units = grid_units["xh"])),
        y = ("y", y, dict(description = grid_descriptions["y"], units = grid_units["y"])),
        yh = ("yh", yh, dict(description = grid_descriptions["yh"], units = grid_units["yh"])),
        z = ("z", z_lay, dict(description = grid_descriptions["z"], units = grid_units["z"])),
        zh = ("zh", z_lev, dict(description = grid_descriptions["zh"], units = grid_units["zh"])),
        z_lay = ("z_lay", z_lay, dict(description = grid_descriptions["z_lay"], units = grid_units["z_lay"])),
        z_lev = ("z_lev", z_lev, dict(description = grid_descriptions["z_lev"], units = grid_units["z_lev"])),
        ngrid_x = ((), ngrid_x, dict(description = grid_descriptions["ngrid_x"], units = grid_units["ngrid_x"])),
        ngrid_y = ((), ngrid_y, dict(description = grid_descriptions["ngrid_y"], units = grid_units["ngrid_y"])),
        ngrid_z = ((), ngrid_z, dict(description = grid_descriptions["ngrid_z"], units = grid_units["ngrid_z"])),
        n_bnd_sw = ((), n_bnd_sw, dict(description = grid_descriptions["n_bnd_sw"], units = grid_units["n_bnd_sw"])),
        n_bnd_lw = ((), n_bnd_lw, dict(description = grid_descriptions["n_bnd_lw"], units = grid_units["n_bnd_lw"])),
    )

    return coords

def coarsen_coords(coords: dict, coarse_factor: NP_INT) -> dict:
    nx_fine: NP_INT = NP_INT(coords["x"][1].size)
    nx_coarse: NP_INT = nx_fine // coarse_factor
    ngrid_x_coarse: NP_INT = NP_INT(np.ceil(nx_coarse / 10))
    xh_min: NP_REAL = coords["xh"][1].min()
    xh_max: NP_REAL = coords["xh"][1].max()
    xh_coarse: NP_ARRAY[NP_REAL] = np.linspace(xh_min, xh_max, nx_coarse + 1,
        dtype = NP_REAL)
    x_coarse: NP_ARRAY[NP_REAL] = (xh_coarse[:-1] + xh_coarse[1:]) / 2.

    ny_fine: NP_INT = NP_INT(coords["y"][1].size)
    ny_coarse: NP_INT = ny_fine // coarse_factor
    ngrid_y_coarse: NP_INT = NP_INT(np.ceil(ny_coarse / 10))
    yh_min: NP_REAL = coords["yh"][1].min()
    yh_max: NP_REAL = coords["yh"][1].max()
    yh_coarse: NP_ARRAY[NP_REAL] = np.linspace(yh_min, yh_max, ny_coarse + 1,
        dtype = NP_REAL)
    y_coarse: NP_ARRAY[NP_REAL] = (yh_coarse[:-1] + yh_coarse[1:]) / 2.

    ## Spatial RTE-RRTMGP-CPP coords
    coords_coarse: dict = dict(
        x = ("x", x_coarse, dict(description = grid_descriptions["x"], units = grid_units["x"])),
        xh = ("xh", xh_coarse, dict(description = grid_descriptions["xh"], units = grid_units["xh"])),
        y = ("y", y_coarse, dict(description = grid_descriptions["y"], units = grid_units["y"])),
        yh = ("yh", yh_coarse, dict(description = grid_descriptions["yh"], units = grid_units["yh"])),
        ngrid_x = ((), ngrid_x_coarse, dict(description = grid_descriptions["ngrid_x"], units = grid_units["ngrid_x"])),
        ngrid_y = ((), ngrid_y_coarse, dict(description = grid_descriptions["ngrid_y"], units = grid_units["ngrid_y"])),
    )

    return {**coords, **coords_coarse}

def bcast_coords(coords: Optional[dict], comm: MPI_COMM) -> dict:
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    # Broadcast vertical grid info, which is the same across all grids
    lay: Optional[NP_INT] = None
    lev: Optional[NP_INT] = None
    z_lay: Optional[NP_ARRAY[NP_REAL]] = None
    z_lev: Optional[NP_ARRAY[NP_REAL]] = None
    ngrid_z: Optional[NP_INT] = None
    if l_rank == MPI_ROOT:
        lay = NP_INT(coords["01"]["z_lay"][1].size)
        lev = NP_INT(coords["01"]["z_lev"][1].size)
        z_lay = np.copy(coords["01"]["z_lay"][1])
        z_lev = np.copy(coords["01"]["z_lev"][1])
        ngrid_z = coords["01"]["ngrid_z"][1]
    lay = comm.bcast(lay, root = MPI_ROOT)
    lev = comm.bcast(lev, root = MPI_ROOT)
    z_lay = comm.bcast(z_lay, root = MPI_ROOT)
    z_lev = comm.bcast(z_lev, root = MPI_ROOT)
    ngrid_z = comm.bcast(ngrid_z, root = MPI_ROOT)

    # Broadcast coarse_strs to setup l_coords
    coarse_strs: Optional[list[str]] = None
    if l_rank == MPI_ROOT:
        coarse_strs = list(coords.keys())
    coarse_strs = comm.bcast(coarse_strs, root = MPI_ROOT)

    l_grids: dict = {}
    coarse_str: str
    for coarse_str in coarse_strs:
        l_grids[coarse_str] = {}

        # Scatterv the x-grid
        g_nx: Optional[NP_INT] = None
        if l_rank == MPI_ROOT:
            g_nx = NP_INT(coords[coarse_str]["x"][1].size)
        g_nx = comm.bcast(g_nx, root = MPI_ROOT)
        
        l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
        l_counts[0] = (g_nx // comm_size + int(0 < (g_nx % comm_size)))

        l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)

        ii: int
        for ii in range(1, comm_size):
            l_counts[ii] = g_nx // comm_size + int(ii < (g_nx % comm_size))
            l_displs[ii] = l_counts[ii - 1] + l_displs[ii - 1] - 1 ## NOTE: I think this might be wrong?

        dx: Optional[NP_REAL]
        g_x: NP_ARRAY[NP_REAL] = np.empty(g_nx, dtype = NP_REAL)
        l_x: NP_ARRAY[NP_REAL] = np.empty(l_counts[l_rank], dtype = NP_REAL)
        if l_rank == MPI_ROOT:
            g_x = np.copy(coords[coarse_str]["x"][1])
            dx = g_x[1] - g_x[0]
        else:
            dx = None

        comm.Scatterv([g_x, l_counts, l_displs, MPI_REAL], l_x, root = MPI_ROOT)
        dx = comm.bcast(dx, root = MPI_ROOT)
        l_nx: NP_INT
        l_xh: NP_ARRAY[NP_REAL]
        if l_x.size > 0:
            l_nx = NP_INT(l_x.size)
            l_xh = np.append(l_x - dx / 2., l_x[-1] + dx / 2.)
        else:
            l_nx = NP_INT(0)
            l_xh = np.array([], dtype = NP_REAL)

        # Broadcast the other values
        ny: Optional[NP_INT] = None
        y: Optional[NP_ARRAY[NP_REAL]] = None
        yh: Optional[NP_ARRAY[NP_REAL]] = None
        ngrid_y: Optional[NP_INT] = None
        if l_rank == MPI_ROOT:
            y = np.copy(coords[coarse_str]["y"][1])
            yh = np.copy(coords[coarse_str]["yh"][1])
            ny = NP_INT(y.size)
            ngrid_y = coords[coarse_str]["ngrid_y"][1]
        ny = comm.bcast(ny, root = MPI_ROOT)
        y = comm.bcast(y, root = MPI_ROOT)
        yh = comm.bcast(yh, root = MPI_ROOT)
        ngrid_y = comm.bcast(ngrid_y, root = MPI_ROOT)

        # Store the other values in l_grids
        l_grids[coarse_str]["nx"] = l_nx
        l_grids[coarse_str]["x"] = l_x
        l_grids[coarse_str]["xh"] = l_xh

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
        l_grids[coarse_str]["ngrid_y"] = ngrid_y
        l_grids[coarse_str]["ngrid_z"] = ngrid_z

        # Store communication values in each local grid
        l_grids[coarse_str]["l_counts_x"] = l_counts
        l_grids[coarse_str]["l_displs_x"] = l_displs

    return l_grids

def interp_3dfield(xr_dpscream: XR_DATASET, dpscream_field_key: str,
    rte_field_key: str, sort_mask: NP_ARRAY[NP_INT], coords: dict, 
    l_grids: dict, tt: NP_INT, comm: MPI_COMM, interp_method: str = "nearest") -> NP_ARRAY[NP_REAL]:
    field_out: dict = {}

    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

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
                    
        else: # Should have values and midpoints and interfaces
            dpscream_field_key_mid: str = dpscream_field_key + "_mid"

            ## We should always have fields values at layer midpoints
            ## Unless we don't, then this needs to be fixed
            assert(dpscream_field_key_mid in xr_dpscream.keys())
            field_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream[dpscream_field_key_mid].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
        
        z_src = xr_dpscream["z_mid"].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)

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

        nx: NP_INT = NP_INT(coords["01"]["x"][1].size)
        ny: NP_INT = NP_INT(coords["01"]["y"][1].size)
        nz: NP_INT = NP_INT(z_src.shape[1])

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

    l_counts_src: NP_ARRAY[NP_INT] = l_grids["01"]["l_counts_x"] * l_ny_src * l_nlay_src
    l_displs_src: NP_ARRAY[NP_INT] = l_grids["01"]["l_displs_x"] * l_ny_src * l_nlay_src

    l_field_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nlay_src], dtype = NP_REAL) # NOTE: ASSUME only using layer midpoint values for field
    l_z_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nlay_src], dtype = NP_REAL) # NOTE: ASSUME only using layer midpoint values for field

    field_min = comm.bcast(field_min, root = MPI_ROOT)
    field_max = comm.bcast(field_max, root = MPI_ROOT)
    comm.Scatterv([field_src, l_counts_src, l_displs_src, MPI_REAL], l_field_src, root = MPI_ROOT)
    comm.Scatterv([z_src, l_counts_src, l_displs_src, MPI_REAL], l_z_src, root = MPI_ROOT)

    # Get source grid - points to interpolate from
    l_x_src: NP_ARRAY[NP_REAL] = l_grids["01"]["x"]
    l_y_src: NP_ARRAY[NP_REAL] = l_grids["01"]["y"]
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

        l_counts_x: NP_ARRAY[NP_INT] = l_grids[coarse_str]["l_counts_x"]
        l_displs_x: NP_ARRAY[NP_INT] = l_grids[coarse_str]["l_displs_x"] \
            + np.arange(0, comm_size, dtype = NP_INT) # NOTE: Requires an offset from the x-grids meeting
            
        l_counts_lay_tgt: NP_ARRAY[NP_INT] = l_counts_x * l_ny_tgt * l_nlay_tgt
        l_displs_lay_tgt: NP_ARRAY[NP_INT] = l_displs_x * l_ny_tgt * l_nlay_tgt

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
            nx_tgt: NP_INT = NP_INT(coords[coarse_str]["x"][1].size)
            ny_tgt: NP_INT = NP_INT(coords[coarse_str]["y"][1].size)
            nlay_tgt: NP_INT = NP_INT(coords[coarse_str]["z_lay"][1].size)

            field_lay_tgt = np.empty(nx_tgt * ny_tgt * nlay_tgt, dtype = NP_REAL)

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
            l_counts_lev_tgt: list[NP_INT] = l_counts_x * l_ny_tgt * l_nlev_tgt
            l_displs_lev_tgt: list[NP_INT] = l_displs_x * l_ny_tgt * l_nlev_tgt

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
                nlev_tgt: NP_INT = NP_INT(coords[coarse_str]["z_lev"][1].size)
                field_lev_tgt = np.empty(nx_tgt * ny_tgt * nlev_tgt, dtype = NP_REAL)
            
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
    rte_field_key: str, sort_mask: NP_ARRAY[NP_INT], coords: dict, 
    l_grids: dict, tt: int, comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    field_out: dict = {}

    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

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

        nx: NP_INT = NP_INT(coords["01"]["x"][1].size)
        ny: NP_INT = NP_INT(coords["01"]["y"][1].size)

        field_src = field_src.reshape(nx, ny)
    else:
        field_src = None
        field_min = None
        field_max = None

    # Scatterv the original field
    l_nx_src: NP_INT = l_grids["01"]["nx"]
    l_ny_src: NP_INT = l_grids["01"]["ny"]

    l_counts_src: NP_ARRAY[NP_INT] = l_grids["01"]["l_counts_x"] * l_ny_src
    l_displs_src: NP_ARRAY[NP_INT] = l_grids["01"]["l_displs_x"] * l_ny_src

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

        l_counts_x: NP_ARRAY[NP_INT] = l_grids[coarse_str]["l_counts_x"]
        l_displs_x: NP_ARRAY[NP_INT] = l_grids[coarse_str]["l_displs_x"] \
            + np.arange(0, comm_size, dtype = NP_INT) # NOTE: Requires an offset from the x-grids meeting
            
        l_counts_tgt: list[NP_INT] = l_counts_x * l_ny_tgt
        l_displs_tgt: list[NP_INT] = l_displs_x * l_ny_tgt

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
            nx_tgt: NP_INT = NP_INT(coords[coarse_str]["x"][1].size)
            ny_tgt: NP_INT = NP_INT(coords[coarse_str]["y"][1].size)

            field_tgt = np.empty([nx_tgt, ny_tgt], dtype = NP_REAL)

        comm.Gatherv(l_field_tgt, 
            [field_tgt, l_counts_tgt, l_displs_tgt, MPI_REAL],
            root = MPI_ROOT)

        if l_rank == MPI_ROOT:
            field_tgt = np.reshape(field_tgt, (nx_tgt, ny_tgt)) # (nx, ny)
            field_tgt = np.transpose(field_tgt, axes = (1, 0)) # (ny, nx)

        if l_rank == MPI_ROOT:
            field_out[coarse_str][rte_field_key] = field_tgt

    return field_out

def set_unspecified_fields(coords: dict, comm: MPI_COMM) -> dict:
    """
    Set fields not specified by the DP-SCREAM output.
    """
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    fields_out: dict = {}

    if l_rank == MPI_ROOT:
        coarse_factor_str: str
        for coarse_factor_str in coords.keys():
            nx: NP_INT = NP_INT(coords[coarse_factor_str]["x"][1].size)
            ny: NP_INT = NP_INT(coords[coarse_factor_str]["y"][1].size)
            nlay: NP_INT = NP_INT(coords[coarse_factor_str]["z_lay"][1].size)
            n_bnd_sw: NP_INT = NP_INT(coords[coarse_factor_str]["n_bnd_sw"][1])
            n_bnd_lw: NP_INT = NP_INT(coords[coarse_factor_str]["n_bnd_lw"][1])

            ## Longwave boundary conditions
            emis_sfc: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_lw), dtype = NP_REAL)
            
            sfc_alb_dir: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07 
            sfc_alb_dif: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07

            tsi: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx), dtype = NP_REAL) * 551.58

            azi: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx), dtype = NP_REAL) * 0.0 
            
            fields_out[coarse_factor_str]: dict = dict(
                emis_sfc = (fields_dimensions["emis_sfc"], emis_sfc, dict(description = fields_descriptions["emis_sfc"], units = fields_units["emis_sfc"])),
                sfc_alb_dir = (fields_dimensions["sfc_alb_dir"], sfc_alb_dir, dict(description = fields_descriptions["sfc_alb_dir"], units = fields_units["sfc_alb_dir"])),
                sfc_alb_dif = (fields_dimensions["sfc_alb_dif"], sfc_alb_dif, dict(description = fields_descriptions["sfc_alb_dif"], units = fields_units["sfc_alb_dif"])),
                tsi = (fields_dimensions["tsi"], tsi, dict(description = fields_descriptions["tsi"], units = fields_units["tsi"])),
                azi = (fields_dimensions["azi"], azi, dict(description = fields_descriptions["azi"], units = fields_units["azi"]))
            )

            ## Set quantities not expected to be set in the DP-SCREAM output
            unexpected_keys: list[str] = ["vmr_ccl4", "vmr_cfc11", "vmr_cfc12",
                "vmr_cfc22", "vmr_hfc143a", "vmr_hfc125", "vmr_hfc32", "vmr_hfc23",
                "vmr_hfc134a", "vmr_cf4", "vmr_no2", "aermr01", "aermr02",
                "aermr03", "aermr04", "aermr05", "aermr06", "aermr07", "aermr08",
                "aermr09", "aermr10", "aermr11"]
            
            field_vals: NP_ARRAY[NP_REAL] = np.zeros((nlay, ny, nx), dtype = NP_REAL)
            for key in unexpected_keys:
                fields_out[coarse_factor_str][key] = \
                    (fields_dimensions[key], field_vals, 
                        dict(description = fields_descriptions[key], units = fields_units[key])
                    )
                
    return fields_out

def save_rte_rrtmgp_cpp_input(coords: dict, fields: dict, tt: NP_INT,
    file_path_root: str, comm: MPI_COMM, szas: Optional[NP_ARRAY[NP_REAL]] = None):

    l_rank: NP_INT = NP_INT(comm.Get_rank())

    if l_rank == MPI_ROOT:
        time_str: str = ".t_{:03d}".format(tt)

        coarse_factor_str: str
        for coarse_factor_str in coords.keys():
            lr_str: str = ".lr_" + coarse_factor_str
            nx: NP_INT = NP_INT(coords[coarse_factor_str]["x"][1].size)
            ny: NP_INT = NP_INT(coords[coarse_factor_str]["y"][1].size)
            if szas is not None:
                sza: NP_REAL
                for sza in szas:
                    sza_str: str = ".sza_{:03.0f}".format(sza)

                    sza_rad: NP_REAL = np.deg2rad(sza)
                    mu0: NP_ARRAY[NP_REAL] = np.zeros([ny, nx], dtype = NP_REAL) + np.cos(sza_rad)
                    fields[coarse_factor_str]["mu0"]: list = (
                        fields_dimensions["mu0"],
                        mu0,
                        dict(description = fields_descriptions["mu0"], units = fields_units["mu0"])
                    )
                    
                    file_path: str = file_path_root + time_str + sza_str + lr_str + ".in.nc"
                    
                    write_rte_input(coords, fields, coarse_factor_str, file_path)
            else:
                file_path: str = file_path_root + time_str + lr_str + ".in.nc"

                write_rte_input(coords, fields, coarse_factor_str, file_path)
                    
def write_rte_input(coords: dict, fields: dict, coarse_factor_str: str,
    file_path: str):

    out_coords: dict = coords[coarse_factor_str]
    out_fields: dict = fields[coarse_factor_str]

    ds: XR_DATASET = xr.Dataset(
        data_vars = out_fields,
        coords = out_coords
    )

    for v in ds.data_vars:
        ds[v].attrs.pop("coordinates", None)
    ds.to_netcdf(file_path)

if __name__ == "__main__":
    main(sys.argv)
