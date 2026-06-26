# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Find time indices for each day of the simulation.
"""
def find_daytime_indices(rad_tran_infile: str, tol: NP_REAL = NP_REAL(1.e-3)) -> NP_ARRAY[NP_INT]:
    # tol: Tolerance of cosine solar zenith angle (SZA) to mark daytime
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        mu0: XR_DATARRAY = xr_rad_tran["mu0"].isel(x = 0, y = 0) # Cosine SZA; [nt]; ASSUME- Constant throughout domain

    sunrising_mask: NP_ARRAY[NP_BOOL] = mu0.diff(dim = "time") > NP_REAL(0.)
    sunsetting_mask: NP_ARRAY[NP_BOOL] = mu0.diff(dim = "time") < NP_REAL(0.)
    daystart_indices: NP_ARRAY[NP_INT] = NP_INT(np.where(~sunrising_mask.shift(time = 1, fill_value = False) & sunrising_mask)[0])
    dayend_indices: NP_ARRAY[NP_INT] = NP_INT(np.where(~sunsetting_mask.shift(time = -1, fill_value = False) & sunsetting_mask)[0] + 1)

    ndays: NP_INT = NP_INT(daystart_indices.size)

    daytime_indices_list: list[NP_ARRAY[NP_INT]] = []
    ii: int
    for ii in range(0, ndays):
        daytime_indices_list += [np.arange(daystart_indices[ii], dayend_indices[ii] + 1, dtype = NP_INT)]

    return np.stack(daytime_indices_list) # ASSUME: Days have same number of time-steps