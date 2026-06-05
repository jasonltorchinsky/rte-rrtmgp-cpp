import xarray as xr

from calc_wc import calc_wc

def calc_hwp(rad_tran_infile, in_time_index, x_index = slice(0, None), zmax_index = None, detailed_calc = False):
    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    x = rad_tran_inds["x"].isel(x = [0, 1]).values # [m]

    rad_tran_inds.close()

    dx = x[1] - x[0] # [m]

    wc = calc_wc(rad_tran_infile, in_time_index, x_index, zmax_index, detailed_calc) # [g m^{-3}], [time, lay, y, x]

    breakpoint()

    if x_index is slice(0, None):
        hwp = dx * wc.sum(dim = "x") # [g m^{-2}], [time, lay, y]
    else:
        hwp = dx * wc # [g m^{-2}], [time, lay, y]

    return hwp