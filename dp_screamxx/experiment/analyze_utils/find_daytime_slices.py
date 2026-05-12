import numpy as np
import xarray as xr

def find_daytime_slices(rad_tran_infile, tol = 1.e-3):
    mu0 = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0)
    daytime_mask = np.isfinite(mu0) & (mu0 > tol)
    daystart_indices = np.where(~daytime_mask.shift(time = 1, fill_value = False) & daytime_mask)[0]
    dayend_indices = np.where(~daytime_mask.shift(time = -1, fill_value = False) & daytime_mask)[0]

    ndays = daystart_indices.size

    daytime_slices = []
    for ii in range(ndays):
        daytime_slices += [slice(daystart_indices[ii], dayend_indices[ii] + 1)]

    return daytime_slices