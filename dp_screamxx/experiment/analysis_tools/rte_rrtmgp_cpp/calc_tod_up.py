import xarray as xr

def calc_tod_up(rad_tran_outfile, out_time_index):
    rad_tran_outds = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)

    ts_tod_up = rad_tran_outds["sw_flux_up"].isel(lev = -1, time = out_time_index) # [W m^{-2}] [time, y, x]

    rt_tod_up = rad_tran_outds["rt_flux_tod_up"].isel(time = out_time_index) # [W m^{-2}] [time, y, x]

    rad_tran_outds.close()

    return [ts_tod_up, rt_tod_up]