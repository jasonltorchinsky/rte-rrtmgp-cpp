# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Find hours since simulation start for given time indices.
"""
def find_times(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT]) -> NP_ARRAY[NP_REAL]:
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        time: XR_DATARRAY = xr_rad_tran["time"] # Time since simulation start; [nt]

    time: NP_ARRAY[NP_REAL] = (time[time_indices.flatten()]).to_numpy().astype(NP_REAL) # Time since simulation start; [h]

    return time.reshape(time_indices.shape)