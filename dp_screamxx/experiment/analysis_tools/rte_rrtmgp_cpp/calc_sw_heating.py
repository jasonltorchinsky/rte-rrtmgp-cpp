# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g, R_d, R_v, cp_d, cp_v, cp_lw, cp_iw, sec_per_day
from .calc_mass_moist_air import calc_mass_moist_air

"""
Calculate shortwave heating rates.
"""
def calc_sw_heating(rad_tran_infile: str, rad_tran_outfile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, solver: str = "rt") -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    assert(solver in ["rt", "ts"])

    #---------------------------------------------------------------------------
    # Calculate absorbed shorwave flux
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    if solver == "rt":
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_abs_dir: XR_DATARRAY = xr_rad_tran["rt_flux_abs_dir"] # Absorbed direct shortwave flux; [time, z, y, x]; [W m^{-3}]
            sw_flux_abs_dif: XR_DATARRAY = xr_rad_tran["rt_flux_abs_dif"] # Absorbed diffuse shortwave flux; [time, z, y, x]; [W m^{-3}]

        #-----------------------------------------------------------------------
        # Select relevant times for fields from RTE-RRTMGP-CPP file
        #-----------------------------------------------------------------------
        sw_flux_abs_dir = sw_flux_abs_dir.isel(time = time_indices) # [time, lay, y, x]
        sw_flux_abs_dif = sw_flux_abs_dif.isel(time = time_indices) # [time, lay, y, x]

        #-----------------------------------------------------------------------
        # Get absorbed shortwave flux
        #-----------------------------------------------------------------------
        sw_flux_abs: XR_DATARRAY = sw_flux_abs_dir + sw_flux_abs_dif

    else: # solver == "ts"
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_up: XR_DATARRAY = xr_rad_tran["sw_flux_up"] # Upwelling shortwave flux; [time, lev, y, x]; [W m^{-2}]
            sw_flux_dn: XR_DATARRAY = xr_rad_tran["sw_flux_dn"] # Downwelling shortwave flux; [time, lev, y, x]; [W m^{-2}]
            z: XR_DATARRAY = xr_rad_tran["z"] # Geometric height at layer midpoints; [z]; [m]

        #-----------------------------------------------------------------------
        # Select relevant times for fields from RTE-RRTMGP-CPP file
        #-----------------------------------------------------------------------
        sw_flux_up = sw_flux_up.isel(time = time_indices) # [time, lev, y, x]
        sw_flux_dn = sw_flux_dn.isel(time = time_indices) # [time, lev, y, x]

        #-----------------------------------------------------------------------
        # Get absorbed shortwave flux
        #-----------------------------------------------------------------------
        dz: NP_REAL = NP_REAL((z[1] - z[0]).to_numpy()) # [m]
        sw_flux_dn_diff: XR_DATARRAY = sw_flux_dn.diff("lev").rename({"lev": "z"}).assign_coords(z = z) # dn[i+1] - dn[i]; [time, lay, y, x]; [W m^{-2}] 
        sw_flux_up_diff: XR_DATARRAY = -sw_flux_up.diff("lev").rename({"lev": "z"}).assign_coords(z = z) # up[i] - up[i+1]; [time, lay, y, x]; [W m^{-2}]
        sw_flux_abs: XR_DATARRAY = (sw_flux_dn_diff + sw_flux_up_diff) / dz # [time, z, x]; [W m^{-3}]

    #---------------------------------------------------------------------------
    # Calculate relevant mixing ratios - qv, qc, qi
    #---------------------------------------------------------------------------
    # ASSUME - Hydrostatic pressure.
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        p_lev: XR_DATAARRAY = xr_rad_tran["p_lev"] # Hydrostatic pressure at levels; [nt, lev, y, x]; [Pa]
        p_lay: XR_DATAARRAY = xr_rad_tran["p_lay"] # Hydrostatic pressure at layers; [nt, lay, y, x]; [Pa]
        t_lay: XR_DATAARRAY = xr_rad_tran["t_lay"] # Temperature at layers; [nt, lay, y, x]; [K]
        lwp: XR_DATAARRAY = xr_rad_tran["lwp"] # Cloud liquid water path at layers; [nt, lay, y, x]; [kg m^{-2}]
        iwp: XR_DATAARRAY = xr_rad_tran["iwp"] # Cloud ice water path at layers; [nt, lay, y, x]; [kg m^{-2}]

    #---------------------------------------------------------------------------
    # Select relevant times for fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    p_lev = p_lev.isel(time = time_indices) # [time, lev, y, x]
    p_lay = p_lay.isel(time = time_indices) # [time, lay, y, x]
    t_lay = t_lay.isel(time = time_indices) # [time, lay, y, x]
    lwp = lwp.isel(time = time_indices) # [time, lay, y, x]
    iwp = iwp.isel(time = time_indices) # [time, lay, y, x]

    #---------------------------------------------------------------------------
    # Calculate water vapor moist mixing ratio
    #---------------------------------------------------------------------------
    dz: NP_REAL = NP_REAL((p_lev["lev"][1] - p_lev["lev"][0]).to_numpy())
    pdel: XR_DATAARRAY = (p_lev.diff("lev")).rename({"lev" : "lay"}).assign_coords({"lay" : p_lay["lay"]}) # Pressure-thickness; [time, lay, y]; [Pa]
    rho: XR_DATARRAY = pdel / (g * dz) # Moist air density; [time, lay, y, x]; [kg m^{-3}]
    qv: XR_DATARRAY = (1. / (R_v - R_d)) * ((p_lay / (rho * t_lay)) - R_d) # Water vapor moist mixing ratio; ; [time, lay, y, x]; [kg kg^{-1}]

    #---------------------------------------------------------------------------
    # Calculate cloud ice water and liquid water moist mixing ratios
    #---------------------------------------------------------------------------
    # Assume uniform horizontal grid spacing in x- and y-.
    dx: NP_REAL = NP_REAL((p_lev["x"][1] - p_lev["x"][0]).to_numpy())
    dy: NP_REAL = dx

    mass_moist_air: XR_DATARRAY = calc_mass_moist_air(rad_tran_infile, time_indices) # [time, lay, y, x]; [kg]
    qc: XR_DATAARRAY = (lwp * dx * dy) / mass_moist_air
    qi: XR_DATAARRAY = (iwp * dx * dy) / mass_moist_air

    #---------------------------------------------------------------------------
    # Calculate shortwave heating rate
    #---------------------------------------------------------------------------
    if x_indices is None:
        # Mass, specific heat, and volume calculations
        mass_cell: XR_DATARRAY = (1. + qc + qi) * mass_moist_air # Mass of moist air and cloud water in cell; [time, lay, y, x]; [kg]
        cp_cell: XR_DATARRAY = (cp_d * (1. - qv) + cp_v * qv + cp_lw * qc + cp_iw * qi) / (1. + qc + qi) # Specific heat at constant pressure of cell; [time, lay, y, x]; [J K^{-1} kg^{-1}]

        vol_cell: NP_REAL = dx * dy * dz # Volume of cell; [time, lev, y, x]; [kg]

        sw_heating: XR_DATAARRAY = ((sw_flux_abs * vol_cell) / (cp_cell * mass_cell)) * sec_per_day # [time, lay, y, x]; [K d^{-1}]
        sw_heating = (sw_heating
            .assign_attrs({"units" : "K d^{-1}", 
                            "long_name" : "shortwave heating rate",
                            "standard_name" : "shortwave_heating_rate"})
            .rename("shortwave_heating_rate"))
    else:
        sw_heating: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            sw_flux_abs_x: NP_ARRAY[NP_REAL] = (sw_flux_abs.isel(time = ii, x = x_indices[ii])).to_numpy().astype(NP_REAL) # Absorbed shortwave fluxes; [lev, y]; [W m^{-3}]

            # Mass, specific heat, and volume calculations
            qv_x: NP_ARRAY[NP_REAL] = qv.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [lay, ny]
            qc_x: NP_ARRAY[NP_REAL] = qc.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [lay, ny]
            qi_x: NP_ARRAY[NP_REAL] = qi.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [lay, ny]
            mass_moist_air_x: NP_ARRAY[NP_REAL] = mass_moist_air.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # Moist air mass; 3 * [lay, y]; [kg]
            mass_cell_x: NP_ARRAY[NP_REAL] = (1. + qc_x + qi_x) * mass_moist_air_x # Mass of moist air and cloud water in cell; [lay, y]; [kg]
            cp_cell_x: NP_ARRAY[NP_REAL] = (cp_d * (1. - qv_x) + cp_v * qv_x + cp_lw * qc_x + cp_iw * qi_x) / (1. + qc_x + qi_x) # Specific heat at constant pressure of cell; [lay, y]; [J K^{-1} kg^{-1}]

            vol_cell_x: NP_REAL = dx * dy * dz # Volume of cell; [lay, y]; [m^{3}]

            sw_heating[ii] = ((sw_flux_abs_x * vol_cell_x) / (cp_cell_x * mass_cell_x)) * sec_per_day # [lay, y]; [K d^{-1}]

    return sw_heating