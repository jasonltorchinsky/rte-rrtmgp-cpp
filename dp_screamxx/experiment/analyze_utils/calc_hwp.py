import xarray as xr

from calc_wc import calc_wc

def calc_hwp(rad_tran_infile, in_time_index, y_index = slice(0, None), zmax_index = None, detailed_calc = False):
    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    y = rad_tran_inds["y"].isel(y = [0, 1]).values # [m]

    rad_tran_inds.close()

    dy = y[1] - y[0] # [m]

    wc = calc_wc(rad_tran_infile, in_time_index, y_index, zmax_index, detailed_calc) # [g m^{-3}], [time, lay, y, x]

    if y_index is None:
        hwp = dy * wc.sum(dim = "y") # [g m^{-2}], [time, lay, x]
    else:
        hwp = dy * wc # [g m^{-2}], [time, lay, x]

    return hwp