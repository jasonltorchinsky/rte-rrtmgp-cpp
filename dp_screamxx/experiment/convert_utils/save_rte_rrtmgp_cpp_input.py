# Standard Library Imports
from datetime import datetime
import os
import re
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM, XR_DATASET, XR_DATAARRAY
from consts.numeric import MPI_ROOT
from consts.rte_rrtmgp_cpp_fields import fields_dimensions, fields_descriptions, fields_units

def save_rte_rrtmgp_cpp_input(rad_tran_tgt_grids: dict, rad_tran_tgt_vars_dict: dict,
    rad_tran_indir: str, dp_scream_file: str, time_idx: NP_INT, comm: MPI_COMM):
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Make tweaks to xarray data arrays to match necessary format, and write to file
    #---------------------------------------------------------------------------
    time_str: str = "t_{:03}".format(time_idx)
    file_name_base: str = re.sub(".nc", "", os.path.basename(dp_scream_file))
    coarse_factor_str: str
    for coarse_factor_str in rad_tran_tgt_grids:
        file_name: str = file_name_base + "." + coarse_factor_str + "." + time_str + ".in.nc"
        file_path: str = os.path.join(rad_tran_indir, file_name)

        rad_tran_tgt_grid: dict = rad_tran_tgt_grids[coarse_factor_str]
        rad_tran_tgt_vars: dict = rad_tran_tgt_vars_dict[coarse_factor_str]

        var_key: str
        var: XR_DATAARRAY
        for var_key, var in rad_tran_tgt_vars.items():
            if "z" in var.dims:
                var = (var
                    .rename({"z" : "lay"}))
            elif "zh" in var.dims:
                var = (var
                    .rename({"zh" : "lev"}))
            rad_tran_tgt_vars[var_key] = var

        xr_rte_rrtmgp_cpp: XR_DATASET = XR_DATASET(
            data_vars = rad_tran_tgt_vars, 
            coords = rad_tran_tgt_grid)

        xr_rte_rrtmgp_cpp.to_netcdf(file_path)

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = ("[{}]: [Rank {}]: Writing to {}...").format(current_time, l_rank, file_path)
        print(msg, flush = True)