import numpy as np
import xarray as xr

def find_daytime_slices(infile, tol = 1.e-3, mode = "rte-rrtmgp-cpp"):
    assert(mode in ["rte-rrtmgp-cpp", "dp-scream"])
    if mode == "rte-rrtmgp-cpp":
        key = "mu0"
        mu0 = xr.open_dataset(infile, engine = "netcdf4", decode_timedelta = False)[key].isel(x = 0, y = 0)
    elif mode == "dp-scream":
        key = "cosine_solar_zenith_angle"
        mu0 = xr.open_dataset(infile, engine = "netcdf4", decode_timedelta = False)[key].isel(ncol = 0)

    
    daytime_mask = np.isfinite(mu0) & (mu0 > tol)
    daystart_indices = np.where(~daytime_mask.shift(time = 1, fill_value = False) & daytime_mask)[0]
    dayend_indices = np.where(~daytime_mask.shift(time = -1, fill_value = False) & daytime_mask)[0]

    ndays = daystart_indices.size

    daytime_slices = []
    for ii in range(ndays):
        daytime_slices += [slice(daystart_indices[ii], dayend_indices[ii] + 1)]

    return daytime_slices