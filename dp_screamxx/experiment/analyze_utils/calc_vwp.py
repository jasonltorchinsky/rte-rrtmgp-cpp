import xarray as xr

from calc_wc import calc_wc

def calc_vwp(rad_tran_infile, in_time_index, detailed_calc = False):
    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    lev = rad_tran_inds["lev"].isel(lev = [0, 1]).values # [m]

    rad_tran_inds.close()

    dz = lev[1] - lev[0] # [m]

    wc = calc_wc(rad_tran_infile, in_time_index, detailed_calc = detailed_calc) # [g m^{-3}], [time, lay, y, x]

    vwp = dz * wc.sum(dim = "lay") # [g m^{-2}], [time, y, x]

    return vwp