# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM, XR_DATASET
from consts.numeric import MPI_ROOT
from consts.rte_rrtmgp_cpp_fields import fields_dimensions, fields_descriptions, fields_units

def save_rte_rrtmgp_cpp_input(coords: dict, xr_rrtmgp_cpp_dict: dict,
    file_path_root: str, comm: MPI_COMM, szas: Optional[NP_ARRAY[NP_REAL]] = None):
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Only root process writes to file
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        coarse_str: str
        for coarse_str in xr_rrtmgp_cpp_dict.keys():
            lr_str: str = ".lr_" + coarse_str
            nx: NP_INT = NP_INT(coords[coarse_str]["x"][1].size)
            ny: NP_INT = NP_INT(coords[coarse_str]["y"][1].size)

            if szas is not None:
                sza: NP_REAL
                for sza in szas:
                    sza_str: str = ".sza_{:03.0f}".format(sza)

                    sza_rad: NP_REAL = np.deg2rad(sza)
                    mu0: NP_ARRAY[NP_REAL] = np.zeros([ny, nx], dtype = NP_REAL) + np.cos(sza_rad)
                    xr_rrtmgp_cpp_dict[coarse_str]["mu0"]: list = (
                        fields_dimensions["mu0"],
                        mu0,
                        dict(description = fields_descriptions["mu0"], units = fields_units["mu0"])
                    )
                    
                    file_path: str = file_path_root + sza_str + lr_str + ".in.nc"

                    write_rte_input(coords, xr_rrtmgp_cpp_dict, coarse_str, file_path)
            else:
                file_path: str = file_path_root + lr_str + ".in.nc"
                write_rte_input(coords, xr_rrtmgp_cpp_dict, coarse_str, file_path)
    comm.barrier()

def write_rte_input(coords: dict, fields: dict, coarse_str: str,
    file_path: str):

    out_coords: dict = coords[coarse_str]
    out_fields: dict = fields[coarse_str]

    ds: XR_DATASET = xr.Dataset(
        data_vars = out_fields,
        coords = out_coords
    )

    for v in ds.data_vars:
        ds[v].attrs.pop("coordinates", None)
    ds.to_netcdf(file_path)