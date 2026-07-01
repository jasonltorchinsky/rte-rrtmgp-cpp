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

    mu0_diff: NP_ARRAY[NP_REAL] = NP_REAL(mu0.diff(dim = "time").to_numpy())
    dayends: NP_ARRAY[NP_INT] = np.where((mu0_diff[:-1] < 0) & (mu0_diff[1:] > 0))[0] + 1
    daystart_indices: NP_ARRAY[NP_INT] = np.r_[0, dayends + 1] # ASSUME: Day start at 0, days start right after days end
    dayend_indices: NP_ARRAY[NP_INT] = np.r_[dayends, mu0.size - 1] # ASSUME: Day ends at last index

    ndays: NP_INT = NP_INT(daystart_indices.size)

    daytime_indices_list: list[NP_ARRAY[NP_INT]] = []
    ii: int
    for ii in range(0, ndays):
        daytime_indices_list += [np.arange(daystart_indices[ii], dayend_indices[ii] + 1, dtype = NP_INT)]

    return np.stack(daytime_indices_list) # ASSUME: Days have same number of time-steps