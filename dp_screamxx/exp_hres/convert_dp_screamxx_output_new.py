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
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_COMM, MPI_ROOT, XR_DATASET
from utils.dp_screamxx_fields import dpscream_3dfield_keys, dpscream_2dfield_keys
from utils.rte_rrtmgp_cpp_fields import rte_3dfield_keys, rte_2dfield_keys
from convert_utils import coarsen_g_grid, get_g_grid_01, get_sort_mask, grids_to_coords, \
    interp_2dfield, interp_3dfield, save_rte_rrtmgp_cpp_input, scatterv_g_grids, set_unspecified_vals, \
    vals_to_fields

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
        coarse_factors = np.array([1], dtype = NP_INT)
    else:
        coarse_factors = np.array(ast.literal_eval(args.coarse_factors[0]), dtype = NP_INT).flatten()
        if not np.any(coarse_factors == NP_INT(1)):
            coarse_factors = np.append(coarse_factors, NP_INT(1))
    coarse_factors = np.sort(coarse_factors)
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

    interp_method: str = "linear"

    file_ext: re.Pattern = re.compile("\\.nc")
    rte_rrtmgp_cpp_file_name_root: str = file_ext.sub("", os.path.basename(dpscream_file_path))
    rte_rrtmgp_cpp_file_path_root: str = os.path.join(rte_rrtmgp_cpp_dir_path, rte_rrtmgp_cpp_file_name_root)

    # Root rank gets original horizontal grid
    xr_dpscream: Optional[XR_DATASET]
    sort_mask: Optional[NP_ARRAY[NP_INT]]
    g_grids: Optional[dict]
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
        g_grids = {}
        g_grids["01"] = get_g_grid_01(xr_dpscream, sort_mask)

        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            coarse_factor_str: str = "{:02}".format(coarse_factor)
            g_grids[coarse_factor_str] = coarsen_g_grid(g_grids["01"], coarse_factor)
    else:
        xr_dpscream = None
        sort_mask = None
        g_grids = None
        times = None

    g_coords: Optional[list] = grids_to_coords(xr_dpscream, g_grids, comm)

    times = comm.bcast(times, root = MPI_ROOT)
    l_grid_src: dict
    l_grids_tgt: dict
    [l_grid_src, l_grids_tgt] = scatterv_g_grids(g_grids, comm)

    tt: NP_INT
    for tt in times:
        g_vals: Optional[dict]
        if l_rank == MPI_ROOT:
            g_vals = dict()
            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "{:02}".format(coarse_factor)
                g_vals[coarse_factor_str] = {}
        else:
            g_vals = None

        ## Set unspecified fields
        g_unspecified_vals: dict = set_unspecified_vals(xr_dpscream, g_grids, comm)
        if l_rank == MPI_ROOT:
            coarse_factor_str: str
            for coarse_factor_str in g_vals.keys():
                g_vals[coarse_factor_str] = {**g_vals[coarse_factor_str], **g_unspecified_vals[coarse_factor_str]}

        ## Interpolate 3D fields
        ii: int
        for ii in range(0, len(dpscream_3dfield_keys)):
            dpscream_field_key: str = dpscream_3dfield_keys[ii]
            rte_field_key: str = rte_3dfield_keys[ii]
            val_dict: dict = interp_3dfield(xr_dpscream, dpscream_field_key, rte_field_key,
                sort_mask, g_grids, l_grid_src, l_grids_tgt, tt, comm, interp_method = interp_method)

            if l_rank == MPI_ROOT:
                coarse_factor_str: str
                for coarse_factor_str in val_dict.keys():
                    vals: dict = val_dict[coarse_factor_str]
                    for val_key in vals.keys():
                        g_vals[coarse_factor_str][val_key] = vals[val_key]

        ## Interpolate 2D fields
        ii: int
        for ii in range(0, len(dpscream_2dfield_keys)):
            dpscream_field_key: str = dpscream_2dfield_keys[ii]
            rte_field_key: str = rte_2dfield_keys[ii]
            val_dict: dict = interp_2dfield(xr_dpscream, dpscream_field_key, rte_field_key,
                sort_mask, g_grids, l_grid_src, l_grids_tgt, tt, comm, interp_method = interp_method)

            if l_rank == MPI_ROOT:
                coarse_factor_str: str
                for coarse_factor_str in val_dict.keys():
                    vals: dict = val_dict[coarse_factor_str]
                    for val_key in vals.keys():
                        g_vals[coarse_factor_str][val_key] = vals[val_key]

        g_fields: dict = vals_to_fields(g_vals, comm)
        save_rte_rrtmgp_cpp_input(g_coords, g_fields, tt, rte_rrtmgp_cpp_file_path_root, comm, szas)

if __name__ == "__main__":
    main(sys.argv)
