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
    grid_descriptions, grid_units, fields_dimensions, fields_descriptions, fields_units
from convert_utils import bcast_coords, coarsen_coords, get_coords_01, get_sort_mask, \
    interp_2dfield, interp_3dfield, save_rte_rrtmgp_cpp_input, set_unspecified_fields


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

if __name__ == "__main__":
    main(sys.argv)
