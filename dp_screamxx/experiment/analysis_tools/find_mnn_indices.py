import numpy as np
import xarray as xr

def find_mnn_indices(rad_tran_infile, tol = 1.e-3): # Morning, Noon, Night indices for a given day
    mu0 = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0)
    daytime_mask = np.isfinite(mu0) & (mu0 > tol)
    daystart_indices = np.where(~daytime_mask.shift(time = 1, fill_value = False) & daytime_mask)[0]
    dayend_indices = np.where(~daytime_mask.shift(time = -1, fill_value = False) & daytime_mask)[0]

    ndays = daystart_indices.size

    mnn_indices = np.zeros((ndays, 3), dtype = np.int32)
    for ii in range(ndays):
        daystart_index = daystart_indices[ii]
        dayend_index = dayend_indices[ii]
        index_range = dayend_index - daystart_index
        mnn_indices[ii,:] = np.array([int(np.round(0.15 * index_range + daystart_index)), # Morning
            int(np.round(0.5 * index_range + daystart_index)), # Noon
            int(np.round(0.67 * index_range + daystart_index))]) # Night

    return mnn_indices