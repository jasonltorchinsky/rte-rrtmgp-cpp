# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.physical import sec_per_hour
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Find solar zenith angles for given time indices.
"""
def find_szas(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT]) -> NP_ARRAY[NP_REAL]:
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        cosine_sza: XR_DATARRAY = xr_rad_tran["mu0"].isel(x= 0, y = 0) # Cosine SZA; [nt]; ASSUME- Constant throughout domain

    sza: NP_ARRAY[NP_REAL] = np.rad2deg(np.acos((cosine_sza[time_indices.flatten()]).to_numpy().astype(NP_REAL))) # SZA; [degrees]; [ndays, 3]

    return sza.reshape(time_indices.shape)