# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Find time indices for morning, noon, and night for each day of the simulation
"""
def find_mnn_indices(dp_scream_file: str, tol: NP_REAL = NP_REAL(1.e-3)) -> NP_ARRAY[NP_INT]:
    # tol: Tolerance of cosine solar zenith angle (SZA) to mark daytime
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        cosine_sza: XR_DATARRAY = xr_dp_scream["cosine_solar_zenith_angle"].isel(ncol = 0) # Cosine SZA; [nt]; ASSUME- Constant throughout domain
    
    daytime_mask: NP_ARRAY[NP_BOOL] = np.isfinite(cosine_sza) & (cosine_sza > tol) # Mask for daytime time-steps
    daystart_indices: NP_ARRAY[NP_INT] = (np.where(~daytime_mask.shift(time = 1, fill_value = False) & daytime_mask)[0]).astype(NP_INT) # Time indices for the start of each day
    dayend_indices: NP_ARRAY[NP_INT] = (np.where(~daytime_mask.shift(time = -1, fill_value = False) & daytime_mask)[0]).astype(NP_INT) # Time indices for the end of each day

    ndays: NP_INT = NP_INT(daystart_indices.size)

    mnn_indices: NP_ARRAY[NP_INT] = np.zeros((ndays, 3), dtype = NP_INT)
    ii: int
    for ii in range(0, ndays):
        daystart_index: NP_INT = daystart_indices[ii]
        dayend_index: NP_INT = dayend_indices[ii]
        index_range: NP_INT = dayend_index - daystart_index
        mnn_indices[ii,:] = np.array([NP_INT(np.round(0.15 * index_range + daystart_index)), # Morning
            NP_INT(np.round(0.5 * index_range + daystart_index)), # Noon
            NP_INT(np.round(0.67 * index_range + daystart_index))]) # Night

    return mnn_indices