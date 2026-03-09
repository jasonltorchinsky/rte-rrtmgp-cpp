# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_COMM, MPI_ROOT, XR_DATASET
from consts.rte_rrtmgp_cpp_fields import fields_dimensions, fields_descriptions, fields_units

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