import xarray as xr

from consts import g, R_d
from calc_mass_air import calc_mass_air

def calc_wc(rad_tran_infile, in_time_index, x_index = slice(0, None), zmax_index = None, detailed_calc = False):
    levmax_index = zmax_index
    laymax_index = zmax_index
    if zmax_index is not None:
        levmax_index += 1

    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    lwp = rad_tran_inds["lwp"].isel(time = in_time_index, x = x_index, lay = slice(0, laymax_index)) # [kg m^{-2}], [time, lay, y]
    iwp = rad_tran_inds["iwp"].isel(time = in_time_index, x = x_index, lay = slice(0, laymax_index)) # [kg m^{-2}], [time, lay, y]

    p_lev = rad_tran_inds["p_lev"].isel(time = in_time_index, x = x_index, lev = slice(0, levmax_index)) # [Pa], [time, lev, y]

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

    mass_air = calc_mass_air(rad_tran_infile, in_time_index, 
        x_index = x_index, zmax_index = zmax_index, detailed_calc = detailed_calc) # [kg], [time, lay, y, x]

    lwc = qc * mass_air / vol # [kg m^{-3}], [time, lay, y, x]
    iwc = qi * mass_air / vol # [kg m^{-3}], [time, lay, y, x]
    wc = (lwc + iwc) * 1.e3 # [g m^{-3}], [time, lay, y, x]

    return wc
