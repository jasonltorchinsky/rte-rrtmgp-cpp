import xarray as xr

from consts import R_d

def calc_mass_air(rad_tran_infile, in_time_index, x_index = slice(0, None), zmax_index = None, detailed_calc = False):
    laymax_index = zmax_index

    rad_tran_inds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)

    p_lay = rad_tran_inds["p_lay"].isel(time = in_time_index, x = x_index, lay = slice(0, laymax_index)) # [Pa], [time, lay, y]
    t_lay = rad_tran_inds["t_lay"].isel(time = in_time_index, x = x_index, lay = slice(0, laymax_index)) # [K], [time, lay, y]

    x = rad_tran_inds["x"].isel(x = [0, 1]).values # [m]
    lev = rad_tran_inds["lev"].isel(lev = [0, 1]).values # [m]

    rad_tran_inds.close()

    # ASSUME: Volume of each element is constant, dx = dy
    dx = x[1] - x[0]
    dz = lev[1] - lev[0] # [m]
    vol = dx**2 * dz # [m^{3}]

    # TO-DO: Get specific gas constant.
    R = R_d
    mass_air = (p_lay * vol) / (R * t_lay) # [kg], [time, lay, y, x]

    return mass_air
