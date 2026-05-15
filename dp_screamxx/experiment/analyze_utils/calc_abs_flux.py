import xarray as xr

def calc_abs_flux(rad_tran_outfile, out_time_index, 
    x_index = slice(0, None), y_index = slice(0, None), zmax_index = None):
    levmax_index = zmax_index
    laymax_index = zmax_index
    if zmax_index is not None:
        levmax_index += 1

    rad_tran_outds = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)

    ts_flux_dn = rad_tran_outds["sw_flux_dn"].isel(
        time = out_time_index, lev = slice(0, levmax_index), 
        y = y_index, x = x_index) # [W m^{-2}] [time, lev, y, x]
    ts_flux_up = rad_tran_outds["sw_flux_up"].isel(
        time = out_time_index, lev = slice(0, levmax_index), 
        y = y_index, x = x_index) # [W m^{-2}] [time, lev, yx]
    lev = rad_tran_outds["lev"].isel(lev = [0, 1]).values # [m]

    rt_flux_abs_dif = rad_tran_outds["rt_flux_abs_dif"].isel(
        time = out_time_index, z = slice(0, laymax_index), 
        y = y_index, x = x_index) # [W m^{-3}] [time, lay, y, x]
    rt_flux_abs_dir = rad_tran_outds["rt_flux_abs_dir"].isel(
        time = out_time_index, z = slice(0, laymax_index), 
        y = y_index, x = x_index) # [W m^{-3}] [time, lay, y, x]

    rad_tran_outds.close()

    dz = lev[1] - lev[0] # [m]
    ts_flux_dn_diff = ts_flux_dn.diff("lev").rename({"lev": "z"}).rename("ts_flux_diff").assign_coords(z = rt_flux_abs_dif["z"].values) # dn[i+1] - dn[i]; [W m^{-2}] [time lay, x]
    ts_flux_up_diff = -ts_flux_up.diff("lev").rename({"lev": "z"}).rename("ts_flux_diff").assign_coords(z = rt_flux_abs_dif["z"].values) # up[i] - up[i+1]; [W m^{-2}] [time lay, x]
    ts_flux_abs = (ts_flux_dn_diff + ts_flux_up_diff) / dz # [W m^{-3}] [time, lay, x]

    rt_flux_abs = rt_flux_abs_dir + rt_flux_abs_dif # [W m^{-3}] [time, lay, x]

    return [ts_flux_abs, rt_flux_abs]