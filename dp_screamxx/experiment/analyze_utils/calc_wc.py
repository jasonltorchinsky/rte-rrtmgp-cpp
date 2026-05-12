import xarray as xr

from consts import g, R_d

def calc_wc(rad_tran_infile, in_time_index, y_index = slice(0, None), zmax_index = None, detailed_calc = False):
    levmax_index = zmax_index
    laymax_index = zmax_index
    if zmax_index is not None:
        levmax_index += 1

    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    lwp = rad_tran_inds["lwp"].isel(time = in_time_index, y = y_index, lay = slice(0, laymax_index)) # [kg m^{-2}], [time, lay, x]
    iwp = rad_tran_inds["iwp"].isel(time = in_time_index, y = y_index, lay = slice(0, laymax_index)) # [kg m^{-2}], [time, lay, x]

    p_lev = rad_tran_inds["p_lev"].isel(time = in_time_index, y = y_index, lev = slice(0, levmax_index)) # [Pa], [time, lev, x]
    p_lay = rad_tran_inds["p_lay"].isel(time = in_time_index, y = y_index, lay = slice(0, laymax_index)) # [Pa], [time, lay, x]
    t_lay = rad_tran_inds["t_lay"].isel(time = in_time_index, y = y_index, lay = slice(0, laymax_index)) # [K], [time, lay, x]

    x = rad_tran_inds["x"].isel(x = [0, 1]).values # [m]
    lev = rad_tran_inds["lev"].isel(lev = [0, 1]).values # [m]

    rad_tran_inds.close()

    dp = -p_lev.diff("lev").rename({"lev": "lay"}).rename("dp").assign_coords(lay = lwp["lay"].values) # Pressure thickness # [time, lay, y, x]
    qc = g * lwp / dp # Cloud Liquid-Water Mass-Mixing Ratio [kg kg^{-1}], [time, lay, y, x]
    qi = g * iwp / dp # Cloud Ice Water Mass-Mixing Ratio [kg kg^{-1}], [time, lay, y, x]

    # ASSUME: Volume of each element is constant, dx = dy
    dx = x[1] - x[0]
    dz = lev[1] - lev[0] # [m]
    vol = dx**2 * dz # [m^{3}]

    # TO-DO: Get specific gas constant.
    R = R_d
    mass_air = (p_lay * vol) / (R * t_lay) # [kg], [time, lay, y, x]

    lwc = qc * mass_air / vol # [kg m^{-3}], [time, lay, y, x]
    iwc = qi * mass_air / vol # [kg m^{-3}], [time, lay, y, x]
    wc = (lwc + iwc) * 1.e3 # [g m^{-3}], [time, lay, y, x]

    return wc
