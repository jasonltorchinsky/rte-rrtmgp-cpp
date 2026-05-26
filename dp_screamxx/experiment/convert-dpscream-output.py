"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import ast
import os
import re

from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_COMM, MPI_ROOT, XR_DATASET
from consts.dp_screamxx_fields import dpscream_3dfield_keys, dpscream_2dfield_keys
from consts.rte_rrtmgp_cpp_fields import rte_3dfield_keys, rte_2dfield_keys
from convert_utils import coarsen_g_grid, get_g_grid_01, get_sort_mask, grids_to_coords, \
    interp_2dfield, interp_3dfield, save_rte_rrtmgp_cpp_input, scatterv_g_grids, set_unspecified_vals, \
    vals_to_fields
from analyze_utils import find_daytime_slices

# Script variables
prog_name: str = "convert-dpscream-output"
prog_desc: str = "Converts DP-SCREAM output to RTE-RRTMGP-CPP+RT input."

def main():
    #---------------------------------------------------------------------------
    # Set up MPI communicator
    #---------------------------------------------------------------------------
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    
    parser.add_argument("--dp-scream-file", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path to DP-SCREAM output file."
    )
    
    parser.add_argument("--rad-tran-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory."
    )

    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Factors by which to coarsen the horizontal grid."
    )

    parser.add_argument("--szas", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Solar zenith angle(s) to create RTE-RRTMGP-CPP+RT input for [degrees]."
    )

    parser.add_argument("--time-interval", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Interval of times to convert to RTE-RRTMGP-CPP+RT input [timesteps]."
    )

    parser.add_argument("--timesteps", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "List of times to convert to RTE-RRTMGP-CPP+RT input - overwritten by --time-interval [timesteps]."
    )

    parser.add_argument("--day-only", action = "store_true",
        required = False,
        help = "Include times only when sun is present - overwritten by --time-interval, --timsteps."
    )
    
    args: Namespace = parser.parse_args()

    dp_scream_file: str = os.path.normpath(args.dp_scream_file)
    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)

    coarse_factors: NP_ARRAY[NP_INT]
    if args.coarse_factors is None:
        coarse_factors = np.array([1], dtype = NP_INT)
    else:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]
        if not np.any(coarse_factors == NP_INT(1)):
            coarse_factors = np.append(coarse_factors, NP_INT(1))

    szas: Optional[NP_ARRAY[NP_REAL]] = None
    if args.szas is not None:
        szas = np.sort(np.array(args.szas.split(","), dtype = NP_REAL))

    if args.time_interval is not None:
        time_interval = np.sort(np.array(args.time_interval.split(","), dtype = NP_INT))
        time_idxs = np.arange(time_interval[0], time_interval[1])
    elif args.timesteps is not None:
        time_idxs = np.sort(np.array(args.timesteps.split(","), dtype = NP_INT))
    elif args.day_only:
        daytime_slices = find_daytime_slices(dp_scream_file, mode = "dp-scream")
        daytime_intervals = []
        for daytime_slice in daytime_slices:
            daytime_intervals += [np.arange(daytime_slice.start, daytime_slice.stop)]
        time_idxs = np.concatenate(daytime_intervals, axis = 0)
    else:
        end_time_idx = xr.open_dataset(dp_scream_file, engine = "netcdf4")["time"].size
        time_idxs = np.arange(end_time_idx)

    #---------------------------------------------------------------------------
    # Set variables used throughout the script
    #---------------------------------------------------------------------------
    interp_method: str = "linear"

    file_ext: re.Pattern = re.compile("\\.nc")
    rad_tran_file_name_root: str = file_ext.sub("", os.path.basename(dp_scream_file))
    rad_tran_file_path_root: str = os.path.join(rad_tran_indir, rad_tran_file_name_root)

    #---------------------------------------------------------------------------
    # Root rank gets original horizontal grid
    #---------------------------------------------------------------------------
    xr_dp_scream: Optional[XR_DATASET] = None
    sort_mask: Optional[NP_ARRAY[NP_INT]] = None
    g_grids: Optional[dict] = None
    if l_rank == MPI_ROOT:
        msg: str = "Opening DP-SCREAM file: {}...".format(dp_scream_file)
        print(msg, flush = True)

        xr_dp_scream = xr.open_dataset(dp_scream_file, engine = "netcdf4",
            decode_timedelta = False).isel(time = time_idxs)
        
        #-----------------------------------------------------------------------
        # Get hours since first time
        #-----------------------------------------------------------------------
        times = (xr_dp_scream["time"] - xr_dp_scream["time"][0]).values.astype(NP_REAL) / (3600.e9) # [ns] => [h]

        #-----------------------------------------------------------------------
        # Coarsen original horizontal grid - force us to keep finest (original) grid
        #-----------------------------------------------------------------------
        sort_mask: Optional[NP_ARRAY[NP_INT]] = get_sort_mask(xr_dp_scream)
        g_grids = {}
        g_grids["01"] = get_g_grid_01(xr_dp_scream, sort_mask)

        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            coarse_factor_str: str = "{:02}".format(coarse_factor)
            g_grids[coarse_factor_str] = coarsen_g_grid(g_grids["01"], coarse_factor)

    #---------------------------------------------------------------------------
    # Prepare grid information for RTE-RRTMGP-CPP+RT input (coord)
    #---------------------------------------------------------------------------
    g_coords: Optional[list] = grids_to_coords(xr_dp_scream, g_grids, comm)

    #---------------------------------------------------------------------------
    # Scatter grids to other processes
    #---------------------------------------------------------------------------
    l_grid_src: dict
    l_grids_tgt: dict
    [l_grid_src, l_grids_tgt] = scatterv_g_grids(g_grids, comm)

    #---------------------------------------------------------------------------
    # Loop through time-steps
    #---------------------------------------------------------------------------
    tt: NP_INT
    for tt in time_idxs:
        #-----------------------------------------------------------------------
        # Set up global (g_) values for xarray to write to netcdf file
        #-----------------------------------------------------------------------
        g_vals: Optional[dict] = None
        if l_rank == MPI_ROOT:
            msg: str = "Converting time-step {}".format(tt)
            print(msg, flush = True)

            time: NP_REAL = times[tt]

            g_vals = dict()
            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "{:02}".format(coarse_factor)
                g_vals[coarse_factor_str] = {"time" : time}

        #-----------------------------------------------------------------------
        # Root process sets default values for those not specified in DP-SCREAM output
        #-----------------------------------------------------------------------
        g_unspecified_vals: dict = set_unspecified_vals(xr_dp_scream, g_grids, comm)
        if l_rank == MPI_ROOT:
            msg: str = "  Setting unspecified fields..."
            print(msg, flush = True)

            coarse_factor_str: str
            for coarse_factor_str in g_vals.keys():
                g_vals[coarse_factor_str] = {**g_vals[coarse_factor_str], **g_unspecified_vals[coarse_factor_str]}

        #-----------------------------------------------------------------------
        # Processes cooperate to interpolate (x,y,z) values to target grids
        #-----------------------------------------------------------------------
        ii: int
        for ii in range(0, len(dpscream_3dfield_keys)):
            dpscream_field_key: str = dpscream_3dfield_keys[ii]
            rte_field_key: str = rte_3dfield_keys[ii]

            if l_rank == MPI_ROOT:
                msg: str = "  Interpolating {}...".format(rte_field_key)
                print(msg, flush = True)

            val_dict: dict = interp_3dfield(xr_dp_scream, dpscream_field_key, rte_field_key,
                sort_mask, g_grids, l_grid_src, l_grids_tgt, tt, comm, interp_method = interp_method)

            if l_rank == MPI_ROOT:
                coarse_factor_str: str
                for coarse_factor_str in val_dict.keys():
                    vals: dict = val_dict[coarse_factor_str]
                    for val_key in vals.keys():
                        g_vals[coarse_factor_str][val_key] = vals[val_key]

        #-----------------------------------------------------------------------
        # Processes cooperate to interpolate (x,y) values to target grids
        #-----------------------------------------------------------------------
        ii: int
        for ii in range(0, len(dpscream_2dfield_keys)):
            dpscream_field_key: str = dpscream_2dfield_keys[ii]
            rte_field_key: str = rte_2dfield_keys[ii]

            if l_rank == MPI_ROOT:
                msg: str = "  Interpolating {}...".format(rte_field_key)
                print(msg, flush = True)

            val_dict: dict = interp_2dfield(xr_dp_scream, dpscream_field_key, rte_field_key,
                sort_mask, g_grids, l_grid_src, l_grids_tgt, tt, comm, interp_method = interp_method)

            if l_rank == MPI_ROOT:
                coarse_factor_str: str
                for coarse_factor_str in val_dict.keys():
                    vals: dict = val_dict[coarse_factor_str]
                    for val_key in vals.keys():
                        g_vals[coarse_factor_str][val_key] = vals[val_key]

        #-----------------------------------------------------------------------
        # Root process saves values to file
        #-----------------------------------------------------------------------
        if l_rank == MPI_ROOT:
            msg: str = "Writing RTE_RRTMGP_CPP input...".format(rte_field_key)
            print(msg, flush = True)
        g_fields: dict = vals_to_fields(g_vals, comm)
        save_rte_rrtmgp_cpp_input(g_coords, g_fields, tt, rad_tran_file_path_root, comm, szas)

if __name__ == "__main__":
    main()
