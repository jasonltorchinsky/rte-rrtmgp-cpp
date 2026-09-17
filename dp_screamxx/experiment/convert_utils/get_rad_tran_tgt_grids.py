# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY
from consts.numeric import MPI_ROOT

from .print_msg import print_msg

def get_rad_tran_tgt_grids(rad_tran_src_grid: dict, coarse_factors: NP_ARRAY[NP_INT], l_rank: NP_INT) -> dict:
    #---------------------------------------------------------------------------
    # Obtain source grid information
    #---------------------------------------------------------------------------
    n_xh: NP_INT = NP_INT(rad_tran_src_grid["xh"].size)
    xh_min: NP_REAL = NP_REAL(rad_tran_src_grid["xh"].min())
    xh_max: NP_REAL = NP_REAL(rad_tran_src_grid["xh"].max())

    n_yh: NP_INT = NP_INT(rad_tran_src_grid["yh"].size)
    yh_min: NP_REAL = NP_REAL(rad_tran_src_grid["yh"].min())
    yh_max: NP_REAL = NP_REAL(rad_tran_src_grid["yh"].max())

    #---------------------------------------------------------------------------
    # Loop through coarsening factors
    #---------------------------------------------------------------------------
    msg: str = "Coarsening RTE-RRTMGP-CPP source grid..."
    print_msg(msg, l_rank)

    rad_tran_tgt_grids: dict = {}
    coarse_factor: NP_INT
    for coarse_factor in coarse_factors:
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

        msg: str = "Coarsening RTE-RRTMGP-CPP source grid to {}...".format(coarse_factor_str)
        print_msg(msg, l_rank)

        n_x_coarse: NP_INT = (n_xh - 1) // coarse_factor
        n_xh_coarse: NP_INT = n_x_coarse + 1
        xh_coarse: NP_ARRAY[NP_REAL] = np.linspace(xh_min, xh_max, n_xh_coarse, dtype = NP_REAL)
        x_coarse: NP_ARRAY[NP_REAL] = 0.5 * (xh_coarse[1:] + xh_coarse[:-1])

        n_y_coarse: NP_INT = (n_yh - 1) // coarse_factor
        n_yh_coarse: NP_INT = n_y_coarse + 1
        yh_coarse: NP_ARRAY[NP_REAL] = np.linspace(yh_min, yh_max, n_yh_coarse, dtype = NP_REAL)
        y_coarse: NP_ARRAY[NP_REAL] = 0.5 * (yh_coarse[1:] + yh_coarse[:-1])

        #-----------------------------------------------------------------------
        # Store coarsened grid in dict
        #-----------------------------------------------------------------------
        rad_tran_tgt_grids[coarse_factor_str] = {"x" : x_coarse,
            "xh" : xh_coarse,
            "y" : y_coarse,
            "yh" : yh_coarse,
            "z" : rad_tran_src_grid["z"],
            "zh" : rad_tran_src_grid["zh"]}
        
    return rad_tran_tgt_grids