# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Find time indices for morning, noon, and night for each day of the simulation
"""
def find_mnn_indices(rad_tran_infile: str, tol: NP_REAL = NP_REAL(1.e-3)) -> NP_ARRAY[NP_INT]:
    # tol: Tolerance of cosine solar zenith angle (SZA) to mark daytime
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        cosine_sza: XR_DATARRAY = (xr_rad_tran["mu0"]
            .isel(x = 0, y = 0)
            .load()) # Cosine SZA; [nt]; ASSUME- Constant throughout domain

    day_only: NP_BOOL = NP_BOOL(np.all(cosine_sza > tol))

    if day_only:
        with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            time: XR_DATARRAY = (xr_rad_tran["time"]
                .load()) # Cosine SZA; [nt]; ASSUME- Constant throughout domain
        dt: XR_DATAARRAY = time.diff(dim = "time")
        daystart_mask: NP_ARRAY[NP_BOOL] = np.append(np.array(True, dtype = NP_BOOL), (dt > dt.median()).to_numpy().astype(NP_BOOL))
        dayend_mask: NP_ARRAY[NP_BOOL] = np.append((dt > dt.median()).to_numpy().astype(NP_BOOL), NP_BOOL(True))

        daystart_indices: NP_ARRAY[NP_INT] = (np.where(daystart_mask)[0]).astype(NP_INT) # Time indices for the start of each day
        dayend_indices: NP_ARRAY[NP_INT] = (np.where(dayend_mask)[0]).astype(NP_INT) # Time indices for the end of each day
    else: # Not day_only
        with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            cosine_sza: XR_DATARRAY = (xr_rad_tran["mu0"]
                .isel(x = 0, y = 0)
                .load()) # Cosine SZA; [nt]; ASSUME- Constant throughout domain
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

        mnn_indices[ii,:] = np.array([NP_INT(np.floor(0.15 * index_range + daystart_index)), # Morning
            NP_INT(np.round(0.5 * index_range + daystart_index)), # Noon
            NP_INT(np.ceil(0.67 * index_range + daystart_index))]) # Night

    return mnn_indices