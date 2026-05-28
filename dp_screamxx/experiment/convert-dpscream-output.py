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
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_COMM, MPI_ROOT, XR_DATASET
from consts.dp_screamxx_fields import dpscream_3d_field_keys, dpscream_2d_field_keys
from consts.rte_rrtmgp_cpp_fields import rte_3d_field_keys, rte_2d_field_keys
from convert_utils import coarsen_g_grid, get_g_grid_01, get_sort_mask, \
    coarsen_2d_fields, coarsen_3d_fields, save_rte_rrtmgp_cpp_input, scatterv_g_grids, set_unspecified_fields, \
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
    nt: NP_INT = NP_INT(time_idxs.size)

    #---------------------------------------------------------------------------
    # Set variables used throughout the script
    #---------------------------------------------------------------------------
    interp_method: str = "linear"

    file_ext: re.Pattern = re.compile("\\.nc")
    rad_tran_file_name_root: str = file_ext.sub("", os.path.basename(dp_scream_file))
    rad_tran_file_path_root: str = os.path.join(rad_tran_indir, rad_tran_file_name_root)

    g_fields_tgt: Optional[dict] = None
    if l_rank == MPI_ROOT:
        g_fields_tgt = {}
        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            coarse_factor_str: str = "{:02}".format(coarse_factor)
            g_fields_tgt[coarse_factor_str] = {}


    #---------------------------------------------------------------------------
    # Root rank gets original horizontal grid
    #---------------------------------------------------------------------------
    xr_dp_scream: Optional[XR_DATASET] = None
    sort_mask: Optional[NP_ARRAY[NP_INT]] = None
    g_grids: Optional[dict] = None
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Initial parsing of DP-SCREAM file: {}...".format(current_time, dp_scream_file)
        print(msg, flush = True)

        #-------------------------------------------------------------------
        # Extract hours since simulation start
        #-------------------------------------------------------------------
        xr_dp_scream: XR_DATASET
        with xr.open_dataset(dp_scream_file, engine = "netcdf4",
            decode_timedelta = False).isel(time = time_idxs) as xr_dp_scream:
        
            times = (xr_dp_scream["time"] - xr_dp_scream["time"][0]).to_numpy() / (3600.e9) # [ns] => [h]

        #-----------------------------------------------------------------------
        # Coarsen original horizontal grid - force us to keep finest (original) grid
        #-----------------------------------------------------------------------
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Coarsening horizontal grid...".format(current_time)
        print(msg, flush = True)

        sort_mask: Optional[NP_ARRAY[NP_INT]] = get_sort_mask(dp_scream_file)
        g_grids = {}
        g_grids["01"] = get_g_grid_01(dp_scream_file, sort_mask)

        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            if coarse_factor != 1: # Covered above
                coarse_factor_str: str = "{:02}".format(coarse_factor)
                g_grids[coarse_factor_str] = coarsen_g_grid(g_grids["01"], coarse_factor)

    #---------------------------------------------------------------------------
    # Prepare grid information for RTE-RRTMGP-CPP+RT input (coord)
    #---------------------------------------------------------------------------
    #g_coords: Optional[list] = grids_to_coords(xr_dp_scream, g_grids, comm)

    #---------------------------------------------------------------------------
    # Scatter sort_mask, grids to other processes
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Scattering horizontal grid...".format(current_time)
        print(msg, flush = True)

    sort_mask = comm.bcast(sort_mask, root = MPI_ROOT)
    l_grid_src: dict
    l_grids_tgt: dict
    [l_grid_src, l_grids_tgt] = scatterv_g_grids(g_grids, coarse_factors, comm)

    #-----------------------------------------------------------------------
    # Set fields that tend to be unspecified in the DP-SCREAM output file
    #-----------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Setting unspecified fields...".format(current_time)
        print(msg, flush = True)
    
    with (xr.open_dataset(dp_scream_file, engine = "netcdf4", 
        decode_timedelta = False)) as xr_dp_scream:
        g_unspecified_fields_tgt: dict = set_unspecified_fields(xr_dp_scream,
            g_grids, comm)

        if l_rank == MPI_ROOT:
            current_time = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Storing unspecified fields...".format(current_time)
            print(msg, flush = True)

            coarse_factor: NP_INT
            for coarse_factor in coarse_factors:
                coarse_factor_str: str = "{:02}".format(coarse_factor)
                g_fields_tgt[coarse_factor_str] = {**g_fields_tgt[coarse_factor_str], **g_unspecified_fields_tgt[coarse_factor_str]}

        if l_rank == MPI_ROOT:
            breakpoint()
        comm.barrier()

    #---------------------------------------------------------------------------
    # Each rank opens the DP-SCREAM file and extract their relevant part
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Opening and reformatting DP-SCREAM dataset...".format(current_time)
        print(msg, flush = True)

    l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
#    with (xr.open_dataset(dp_scream_file, engine = "netcdf4", 
#        decode_timedelta = False)
#        .isel(time = time_idxs, ncol = sort_mask)
#        .rename({"lat": "y", "lon": "x"})
#        .set_index(ncol = ["y", "x"])
#        .unstack("ncol")
#        .transpose(..., "y", "x")
#        .sel(x = slice(l_x_src.min(), l_x_src.max()))) as xr_dp_scream:

        #-----------------------------------------------------------------------
        # Coarsen 3-D fields from DP-SCREAM to RTE-RRTMGP-CPP
        #-----------------------------------------------------------------------
#        if l_rank == MPI_ROOT:
#            current_time = datetime.now().strftime("%H:%M:%S")
#            msg: str = "[{}]: Initiating 3-D field coarsening...".format(current_time)
#            print(msg, flush = True)
#
#        g_3d_fields_tgt: dict = coarsen_3d_fields(xr_dp_scream, g_grids,
#            l_grid_src, l_grids_tgt, comm, interp_method = interp_method)
#
#        if l_rank == MPI_ROOT:
#            current_time = datetime.now().strftime("%H:%M:%S")
#            msg: str = "[{}]: Storing coarsened 3-D fields...".format(current_time)
#            print(msg, flush = True)
#
#            coarse_factor: NP_INT
#            for coarse_factor in coarse_factors:
#                coarse_factor_str: str = "{:02}".format(coarse_factor)
#                g_fields_tgt[coarse_factor_str] = {**g_fields_tgt[coarse_factor_str], **g_3d_fields_tgt[coarse_factor_str]}

        #-----------------------------------------------------------------------
        # Coarsen 2-D fields from DP-SCREAM to RTE-RRTMGP-CPP
        #-----------------------------------------------------------------------
#        if l_rank == MPI_ROOT:
#            current_time = datetime.now().strftime("%H:%M:%S")
#            msg: str = "[{}]: Initiating 2-D field coarsening...".format(current_time)
#            print(msg, flush = True)
#
#        g_2d_fields_tgt: dict = coarsen_2d_fields(xr_dp_scream, g_grids,
#            l_grid_src, l_grids_tgt, comm, interp_method = interp_method)
#
#        if l_rank == MPI_ROOT:
#            current_time = datetime.now().strftime("%H:%M:%S")
#            msg: str = "[{}]: Storing coarsened 2-D fields...".format(current_time)
#            print(msg, flush = True)
#
#            coarse_factor: NP_INT
#            for coarse_factor in coarse_factors:
#                coarse_factor_str: str = "{:02}".format(coarse_factor)
#                g_fields_tgt[coarse_factor_str] = {**g_fields_tgt[coarse_factor_str], **g_2d_fields_tgt[coarse_factor_str]}

    #-----------------------------------------------------------------------
    # Root process saves values to file
    #-----------------------------------------------------------------------
#    if l_rank == MPI_ROOT:
#        msg: str = "Writing RTE_RRTMGP_CPP input...".format(rte_field_key)
#        print(msg, flush = True)
#    g_fields: dict = vals_to_fields(g_vals, comm)
#    save_rte_rrtmgp_cpp_input(g_coords, g_fields, tt, rad_tran_file_path_root, comm, szas)

if __name__ == "__main__":
    main()
