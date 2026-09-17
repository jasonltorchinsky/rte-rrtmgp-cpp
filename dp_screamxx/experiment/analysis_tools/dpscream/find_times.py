# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.physical import sec_per_hour
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Find hours since simulation start for given time indices.
"""
def find_times(dp_scream_file: str, time_indices: NP_ARRAY[NP_INT]) -> NP_ARRAY[NP_REAL]:
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        time: XR_DATARRAY = xr_dp_scream["time"] # Time since simulation start; [nt]

    time: NP_ARRAY[NP_REAL] = ((time[time_indices.flatten()] - time[0]) / (sec_per_hour * 1.e9)).to_numpy().astype(NP_REAL) # Time since simulation start; [h]

    return time.reshape(time_indices.shape)