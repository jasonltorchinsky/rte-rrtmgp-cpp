import xarray as xr

from consts import R_d

def calc_vmr(rad_tran_infile, in_time_index, x_index = slice(0, None), zmax_index = None, detailed_calc = False):
    laymax_index = zmax_index
    
    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    vmr_keys = [key for key in rad_tran_inds.keys() if "vmr" in key]
    vmr = rad_tran_inds[vmr_keys].isel(time = in_time_index, x = x_index, lay = slice(0, laymax_index)) # Multiple fields, [mol mol^{-1}], [time, lay, y, x]

    rad_tran_inds.close()

    return vmr
