# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM
from consts.numeric import MPI_ROOT

from .print_msg import print_msg

def get_rad_tran_src_grid(dp_scream_file: str, l_time_idxs: NP_ARRAY[NP_INT], comm: MPI_COMM) -> dict:
    #---------------------------------------------------------------------------
    # Obtain MPI information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Each rank obtains vertical bounds for the uniform vertical grid for 
    # RTE-RRTMGP-CPP, and creates the uniform vertical grid.
    #---------------------------------------------------------------------------
    msg: str = "Generating RTE-RRTMGP-CPP vertical grid..."
    print_msg(msg, l_rank)

    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        l_z_int: XR_DATAARRAY = xr_dp_scream["z_int"].isel(time = l_time_idxs).load() # Geometric height at level interfaces [top -> bot]; [l_time, ncol, ilev]; [m]
        lon: XR_DATAARRAY = xr_dp_scream["lon"].load() # x-position at column midpoints; [ncol]; [m]
        lat: XR_DATAARRAY = xr_dp_scream["lat"].load() # y-position at column midpoints; [ncol]; [m]
        
    l_z_bot: NP_REAL = NP_REAL(l_z_int.isel(ilev = -1).max()) # [m]
    l_z_top: NP_REAL = NP_REAL(l_z_int.isel(ilev = 0).min()) # [m]

    sendbuf: NP_ARRA[NP_REAL] = np.array([l_z_bot, -l_z_top], dtype = NP_REAL)
    recvbuf: NP_ARRA[NP_REAL] = np.empty_like(sendbuf)

    print_msg("Before Allreduce vertical bounds...", l_rank)
    comm.Allreduce(sendbuf, recvbuf, op = MPI.MAX)
    print_msg("After Allreduce vertical bounds...", l_rank)

    g_z_bot: NP_REAL = NP_REAL(recvbuf[0])
    g_z_top: NP_REAL = NP_REAL(-recvbuf[1])

    n_zh: NP_INT = NP_INT(l_z_int["ilev"].size)

    zh: NP_ARRAY[NP_REAL] = np.linspace(g_z_bot, g_z_top, n_zh, dtype = NP_REAL) # [n_zh]; [m]
    z: NP_ARRAY[NP_REAL] = 0.5 * (zh[1:] + zh[:-1])

    #---------------------------------------------------------------------------
    # Each rank obtains the x- and y- grids from the DP-SCREAM file directly
    #---------------------------------------------------------------------------
    msg: str = "Generating RTE-RRTMGP-CPP horizontal grid..."
    print_msg(msg, l_rank)
        
    x: NP_ARRAY[NP_REAL] = NP_REAL(np.unique(lon)) # [n_x]; [m]
    y: NP_ARRAY[NP_REAL] = NP_REAL(np.unique(lat)) # [n_y]; [m]

    dx: NP_REAL = x[1] - x[0] # x-spacing; [m]
    xh: NP_ARRAY[NP_REAL] = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-position at column interfaces; [n_xh]; [m]

    dy: NP_REAL = y[1] - y[0] # y-spacing; [m]
    yh: NP_ARRAY[NP_REAL] = np.append(y - (dy / 2.), y[-1] + (dy / 2.)) # y-position at column interfaces; [n_yh]; [m]

    #---------------------------------------------------------------------------
    # Store RTE-RRTMGP-CPP source grid in a dict
    #---------------------------------------------------------------------------
    msg: str = "Storing RTE-RRTMGP-CPP source grid..."
    print_msg(msg, l_rank)

    rad_tran_src_grid: dict = {"x" : x,
        "xh" : xh,
        "y" : y,
        "yh" : yh,
        "z" : z,
        "zh" : zh}
        
    return rad_tran_src_grid