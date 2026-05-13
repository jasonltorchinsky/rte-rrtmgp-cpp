import xarray as xr

from calc_abs_flux import calc_abs_flux
from consts import g, R_d, cp_d, cp_iw, cp_lw, sec_per_day

def calc_atm_heating(rad_tran_infile, rad_tran_outfile, in_time_index, out_time_index, 
    y_index = slice(0, None), zmax_index = None, detailed_calc = False):
    [ts_abs_flux, rt_abs_flux] = calc_abs_flux(rad_tran_outfile, out_time_index,
        y_index, zmax_index)

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

    if detailed_calc:
        print("WARNING: Using total cell mass to calculate heating rate.")
        mass_lw = qc * mass_air # [kg], [time, lay, y, x]
        mass_iw = qi * mass_air # [kg], [time, lay, y, x]

        mass_cell = mass_air + mass_lw + mass_iw
        cp_cell = ((cp_d * mass_air) + (cp_lw * mass_lw) + (cp_iw * mass_iw)) / mass_cell
        
        cp_cell = cp_cell.rename({"lay": "z"}).assign_coords(time = ts_abs_flux["time"].values)
    else:
        mass_cell = mass_air
        cp_cell = cp_d

    mass_cell = mass_cell.rename({"lay": "z"}).assign_coords(time = ts_abs_flux["time"].values)
    density_cell = mass_cell / vol # [kg m^{-3}],  [time, lay, y, x]

    # ASSUME: --detailed-calc = False
    ts_atm_heating = (ts_abs_flux / (density_cell * cp_cell)) * sec_per_day # [K d^{-1}]
    rt_atm_heating = (rt_abs_flux / (density_cell * cp_cell)) * sec_per_day # [K d^{-1}]

    return [ts_atm_heating, rt_atm_heating]
