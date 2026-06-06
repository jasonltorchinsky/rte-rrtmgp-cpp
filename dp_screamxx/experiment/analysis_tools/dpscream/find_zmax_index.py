# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def find_zmax_index(dp_scream_file: str, zmax: NP_REAL):
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        cosine_sza: XR_DATARRAY = xr_dp_scream["cosine_solar_zenith_angle"].isel(ncol = 0) # Cosine SZA; [nt]; ASSUME- Constant throughout domain
    return int(np.argmin(np.abs(z - zmax)))