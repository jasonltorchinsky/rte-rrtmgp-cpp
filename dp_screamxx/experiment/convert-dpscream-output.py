"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import os
import re

from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM, XR_DATASET, XR_DATAARRAY
from consts.numeric import MPI_ROOT
from consts.physical import sec_per_hour
from convert_utils import coarsen_dp_scream, find_daytime_slices, get_sort_mask, \
    get_rad_tran_src_grid, get_rad_tran_tgt_grids, print_msg, \
    remap_dp_scream, save_rte_rrtmgp_cpp_input

# Script variables
prog_name: str = "convert-dpscream-output"
prog_desc: str = "Converts DP-SCREAM output to RTE-RRTMGP-CPP+RT input."

def main():
    #---------------------------------------------------------------------------
    # Set up MPI communicator
    #---------------------------------------------------------------------------
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    msg: str = "Parsing command-line input..."
    print_msg(msg, l_rank)

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

    szas: Optional[NP_ARRAY[NP_REAL]] = None
    if args.szas is not None:
        szas = np.sort(np.array(args.szas.split(","), dtype = NP_REAL))

    g_time_idxs: NP_ARRAY[NP_INT]
    if args.time_interval is not None:
        time_interval: NP_ARRAY[NP_INT] = np.sort(np.array(args.time_interval.split(","), dtype = NP_INT))
        g_time_idxs = np.arange(time_interval[0], time_interval[1], dtype = NP_INT)
    elif args.timesteps is not None:
        g_time_idxs = np.sort(np.array(args.timesteps.split(","), dtype = NP_INT))
    elif args.day_only:
        daytime_slices: list[slice] = find_daytime_slices(dp_scream_file, mode = "dp-scream")
        daytime_intervals: list[NP_ARRAY[NP_INT]] = []
        for daytime_slice in daytime_slices:
            daytime_intervals += [np.arange(daytime_slice.start, daytime_slice.stop, dtype = NP_INT)]
        g_time_idxs = np.concatenate(daytime_intervals, axis = 0)
    else:
        end_time_idx: NP_INT = NP_INT(xr.open_dataset(dp_scream_file, engine = "netcdf4")["time"].size)
        g_time_idxs = np.arange(end_time_idx)

    #---------------------------------------------------------------------------
    # Set variables used throughout the script
    #---------------------------------------------------------------------------
    interp_method: str = "linear"

    file_ext: re.Pattern = re.compile("\\.nc")
    rad_tran_file_name_root: str = file_ext.sub("", os.path.basename(dp_scream_file))
    rad_tran_file_path_root: str = os.path.join(rad_tran_indir, rad_tran_file_name_root)

    #---------------------------------------------------------------------------
    # Each rank gets a subset of time-steps to convert
    #---------------------------------------------------------------------------
    msg: str = "Obtaining local time indexes..."
    print_msg(msg, l_rank)

    n_time_idxs: NP_INT = NP_INT(g_time_idxs.size)
    base: NP_INT = n_time_idxs // comm_size
    remainder: NP_INT = n_time_idxs % comm_size

    start_idx: NP_INT = l_rank * base + min(l_rank, remainder)
    end_idx: NP_INT = start_idx + base + NP_INT(l_rank < remainder)

    l_time_idxs: NP_ARRAY[NP_INT] = g_time_idxs[start_idx:end_idx]

    #---------------------------------------------------------------------------
    # Each rank gets times of local time indexes
    #---------------------------------------------------------------------------
    msg: str = "Extracting time since simulation start..."
    print_msg(msg, l_rank)

    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        times: NP_ARRAY[NP_REAL] = (xr_dp_scream["time"] - xr_dp_scream["time"][0]).to_numpy().astype(NP_REAL) / (3600.e9) # [ns] => [h]
    times = times[l_time_idxs]

    #---------------------------------------------------------------------------
    # Each rank gets the mask needed to sort DP-SCREAM output into x- and y-
    #---------------------------------------------------------------------------
    msg: str = "Getting col to x-/y- sorting mask..."
    print_msg(msg, l_rank)

    sort_mask: NP_ARRAY[NP_INT] = get_sort_mask(dp_scream_file)

    #---------------------------------------------------------------------------
    # Get the RTE-RRTMGP-CPP source grid, which has the same horizontal grid
    # as DP-SCREAM, and a uniform vertical grid that is always within
    # the bounds of the DP-SCREAM vertical grid (which moves)
    #---------------------------------------------------------------------------
    msg: str = "Getting RTE-RRTMGP-CPP source grid..."
    print_msg(msg, l_rank)

    rad_tran_src_grid: dict = get_rad_tran_src_grid(dp_scream_file, l_time_idxs, comm)

    #---------------------------------------------------------------------------
    # Generate RTE-RRTMGP-CPP target grids (horizontally coarsened)
    #---------------------------------------------------------------------------
    msg: str = "Generating RTE-RRTMGP-CPP target grids..."
    print_msg(msg, l_rank)

    rad_tran_tgt_grids: dict = get_rad_tran_tgt_grids(rad_tran_src_grid, coarse_factors, comm)

    #---------------------------------------------------------------------------
    # Loop through local time indexes
    #---------------------------------------------------------------------------
    time_idx: NP_INT
    for time_idx in l_time_idxs:
        #-----------------------------------------------------------------------
        # Map relevant values from DP-SCREAM to the RTE-RRTMGP-CPP source grid
        #-----------------------------------------------------------------------
        dp_scream_remap: dict = remap_dp_scream(dp_scream_file, time_idx, rad_tran_src_grid, sort_mask, comm)

        #-----------------------------------------------------------------------
        # Horizontally coarsen relevant DP-SCREAM values to RTE-RRTMGP-CPP target grids
        #-----------------------------------------------------------------------
        dp_scream_coarsen: dict = coarsen_dp_scream(dp_scream_remap, rad_tran_src_grid, rad_tran_tgt_grids, comm)

        #-----------------------------------------------------------------------
        # Obtain time since simulation start
        #-----------------------------------------------------------------------
        xr_dp_scream: XR_DATASET
        with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
            times: XR_DATAARRAY = xr_dp_scream["time"] # Time since simulation start; [time]; [ns]
        time_data: NP_REAL = NP_REAL(times[time_idx] - times[0]) / (sec_per_hour * 1.0e9) # Time since simulation start; [ns] => [h]
        time: XR_DATAARRAY = XR_DATAARRAY(data = np.array([time_data], dtype = NP_REAL),
            dims = ("time"),
            coords = {"time" : np.array([time_data], dtype = NP_REAL)},
            name = "time",
            attrs = {
                "units" : "h",
                "long_name": "time_since_simulation_start",
                "standard_name": "time_since_simulation_start",
            }
        )
        coarse_factor_str: str
        for coarse_factor_str in dp_scream_coarsen.keys():
            dp_scream_coarsen[coarse_factor_str]["time"] = time

        #-----------------------------------------------------------------------
        # Save to RTE-RRTMGP-CPP input file
        #-----------------------------------------------------------------------
        save_rte_rrtmgp_cpp_input(rad_tran_tgt_grids, dp_scream_coarsen, rad_tran_indir,
            dp_scream_file, time_idx, comm)


if __name__ == "__main__":
    main()