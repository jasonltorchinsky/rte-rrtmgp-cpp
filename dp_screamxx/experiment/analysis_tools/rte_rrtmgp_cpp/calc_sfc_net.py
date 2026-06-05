import xarray as xr

def calc_sfc_net(rad_tran_outfile, out_time_index):
    rad_tran_outds = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)

    ts_sfc_dn = rad_tran_outds["sw_flux_dn"].isel(lev = 0, time = out_time_index) # [W m^{-2}] [time, y, x]
    ts_sfc_up = rad_tran_outds["sw_flux_up"].isel(lev = 0, time = out_time_index) # [W m^{-2}] [time, y, x]

    rt_sfc_dif = rad_tran_outds["rt_flux_sfc_dif"].isel(time = out_time_index) # [W m^{-2}] [time, y, x]
    rt_sfc_dir = rad_tran_outds["rt_flux_sfc_dir"].isel(time = out_time_index) # [W m^{-2}] [time, y, x]
    rt_sfc_up = rad_tran_outds["rt_flux_sfc_up"].isel(time = out_time_index) # [W m^{-2}] [time, y, x]

    rad_tran_outds.close()

    rt_sfc_dn = rt_sfc_dir + rt_sfc_dif # [time, y, x]

    ts_sfc_net = ts_sfc_dn - ts_sfc_up
    rt_sfc_net = rt_sfc_dn - rt_sfc_up

    return [ts_sfc_net, rt_sfc_net]