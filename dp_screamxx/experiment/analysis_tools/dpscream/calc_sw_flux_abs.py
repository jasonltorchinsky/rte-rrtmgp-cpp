# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import cp_d, cp_v, cp_lw, cp_iw, sec_per_day
from .calc_mass_moist_air import calc_mass_moist_air

"""
Calculate shortwave absorbed flux
"""
def calc_sw_flux_abs(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, method: str = "pdel") -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    assert(method in ["pdel", "flux"])

    xr_dp_scream: XR_DATASET
    if method == "pdel":
        with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
            sw_heating_pdel: XR_DATAARRAY = xr_dp_scream["SW_heating_pdel"] # Shortwave heating times pressure-thickness; [time, ncol, lev]; [Pa K s^{-1}]
            p_int: XR_DATAARRAY = xr_dp_scream["p_int"] # Pressure at level interfaces; [time, ncol, ilev]; [Pa]
            qv: XR_DATAARRAY = xr_dp_scream["qv"] # Water vapor moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
            qc: XR_DATAARRAY = xr_dp_scream["qc"] # Cloud liquid water moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
            qi: XR_DATAARRAY = xr_dp_scream["qi"] # Cloud ice water moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
            z_int: XR_DATAARRAY = xr_dp_scream["z_int"] # Geometrix height at level interfaces; [nt, ncol, ilev]; [m]

        #-----------------------------------------------------------------------
        # Sort and reshape fields from DP-SCREAM file
        #-----------------------------------------------------------------------
        sw_heating_pdel: XR_DATAARRAY = (sw_heating_pdel
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, lev, y, x]
        p_int: XR_DATAARRAY = (p_int
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        qv: XR_DATAARRAY = (qv
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        qc: XR_DATAARRAY = (qc
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        qi: XR_DATAARRAY = (qi
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        z_int: XR_DATAARRAY = (z_int
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]

        #-----------------------------------------------------------------------
        # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
        #-----------------------------------------------------------------------
        dx: NP_REAL = NP_REAL((z_int["x"][1] - z_int["x"][0]).to_numpy()) # [m]
        dy: NP_REAL = dx # [m]
        dz: XR_DATAARRAY = (-z_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : qc["lev"]}) # Vertical grid spacing; [m]; [time, lev, y, x]
        
        #-----------------------------------------------------------------------
        # Calculate shortwave absorbed flux
        #-----------------------------------------------------------------------
        if x_indices is None:
            # Shortwave heating calculation
            pdel: XR_DATAARRAY = (p_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : sw_heating_pdel["lev"]}) # Pressure-thickness; [time, lev, y]; [Pa]
            sw_heating: XR_DATAARRAY = (sw_heating_pdel / pdel) * sec_per_day # [time, lev, y, x]; [K d^{-1}]

            # Mass, specific heat, and volume calculations
            mass_moist_air: XR_DATAARRAY = calc_mass_moist_air(dp_scream_file, 
                sort_mask, time_indices, x_indices) # Moist air mass; [time, lev, y, x]; [kg]
            mass_cell: XR_DATARRAY = (1. + qc + qi) * mass_moist_air # Mass of moist air and cloud water in cell; [time, lev, y, x]; [kg]
            cp_cell: XR_DATARRAY = (cp_d * (1. - qv) + cp_v * qv + cp_lw * qc + cp_iw * qi) / (1. + qc + qi) # Specific heat at constant pressure of cell; [time, lev, y, x]; [J K^{-1} kg^{-1}]

            vol_cell: XR_DATARRAY = dx * dy * dz # Volume of cell; [time, lev, y, x]; [kg]

            # Absorbed flux calculation
            sw_flux_abs = (sw_heating * cp_cell * mass_cell) / (vol_cell * sec_per_day) # [time, lev, y, x]; [W m^{-3}]
            sw_flux_abs = (sw_flux_abs
                .assign_attrs({"units" : "W m^{-3}", 
                               "long_name" : "shortwave absorbed flux",
                               "standard_name" : "shortwave_flux_absorbed"})
                .rename("shortwave_flux_absorbed"))
        else:
            mass_moist_air_x: list[NP_ARRAY[NP_REAL]] = calc_mass_moist_air(dp_scream_file, 
                sort_mask, time_indices, x_indices) # Moist air mass; 3 * [lev, y]; [kg]
            sw_flux_abs: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
            for ii in range(0, 3): # Assume Morning-Noon-Night indices
                sw_heating_pdel_x: XR_DATAARRAY = sw_heating_pdel.isel(time = ii, x = x_indices[ii]) # [lev, ny]
                p_int_x: XR_DATAARRAY = p_int.isel(time = ii, x = x_indices[ii]) # [lev, ny]
                pdel_x: XR_DATAARRAY = (p_int_x.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : sw_heating_pdel["lev"]}) # Pressure-thickness; [lev, ny]; [Pa]

                sw_heating_x = ((sw_heating_pdel_x / pdel_x) * sec_per_day).to_numpy().astype(NP_REAL) # [lev, y]; [K d^{-1}]

                # Mass, specific heat, and volume calculations
                dz_x: NP_ARRAY[NP_REAL] = dz.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # Level geometric thickness; [lev, y]; [m]
                qv_x: NP_ARRAY[NP_REAL] = qv.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [ilev, ny]
                qc_x: NP_ARRAY[NP_REAL] = qc.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [ilev, ny]
                qi_x: NP_ARRAY[NP_REAL] = qi.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [ilev, ny]
                mass_cell_x: NP_ARRAY[NP_REAL] = (1. + qc_x + qi_x) * mass_moist_air_x[ii] # Mass of moist air and cloud water in cell; [lev, y]; [kg]
                cp_cell_x: NP_ARRAY[NP_REAL] = (cp_d * (1. - qv_x) + cp_v * qv_x + cp_lw * qc_x + cp_iw * qi_x) / (1. + qc_x + qi_x) # Specific heat at constant pressure of cell; [lev, y]; [J K^{-1} kg^{-1}]

                vol_cell_x: NP_ARRAY[NP_REAL] = dx * dy * dz_x # Volume of cell; [lev, y]; [m^{3}]

                sw_flux_abs[ii] = (sw_heating_x * cp_cell_x * mass_cell_x) / (vol_cell_x * sec_per_day) # [lev, y]; [W m^{-3}]

    else: # method == "flux"
        with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
            sw_flux_up: XR_DATAARRAY = xr_dp_scream["SW_flux_up"] # Upwelling shortwave flux; [time, ncol, ilev]; [W m^{-2}]
            sw_flux_dn: XR_DATAARRAY = xr_dp_scream["SW_flux_dn"] # Upwelling shortwave flux; [time, ncol, ilev]; [W m^{-2}]
            z_int: XR_DATAARRAY = xr_dp_scream["z_int"] # Geometric height at level interfaces; [nt, ncol, ilev]; [m]
            z_mid: XR_DATAARRAY = xr_dp_scream["z_mid"] # Geometric height at level interfaces; [nt, ncol, lev]; [m]

        #-----------------------------------------------------------------------
        # Sort and reshape fields from DP-SCREAM file
        #-----------------------------------------------------------------------
        sw_flux_up: XR_DATAARRAY = (sw_flux_up
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        sw_flux_dn: XR_DATAARRAY = (sw_flux_dn
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        z_int: XR_DATAARRAY = (z_int
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, ilev, y, x]
        z_mid: XR_DATAARRAY = (z_mid
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [time, lev, y, x]

        #-----------------------------------------------------------------------
        # Calculate vertical grid spacing
        #-----------------------------------------------------------------------
        dz: XR_DATAARRAY = (-z_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : z_mid["lev"]}) # Vertical grid spacing; [m]; [time, lev, y, x]

        #-----------------------------------------------------------------------
        # Calculate shortwave heating rate
        #-----------------------------------------------------------------------
        if x_indices is None:
            # Absorbed shortwave flux calculation
            sw_flux_up_abs: XR_DATAARRAY = (sw_flux_up.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : z_mid["lev"]}) # Absorbed upwelling shortwave fluxes; [time, lev, y, x]; [W m^{-2}]
            sw_flux_dn_abs: XR_DATAARRAY = (-sw_flux_dn.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : z_mid["lev"]}) # Absorbed downwelling shortwave fluxes; [time, lev, y, x]; [W m^{-2}]
            sw_flux_abs: XR_DATAARRAY = (sw_flux_up_abs + sw_flux_dn_abs) / dz # Absorbed shortwave fluxes; [time, lev, y, x]; [W m^{-3}]

            sw_flux_abs = (sw_flux_abs
                .assign_attrs({"units" : "W m^{-3}", 
                               "long_name" : "shortwave absorbed flux",
                               "standard_name" : "shortwave_absorbed_flux"})
                .rename("shortwave_absorbed_flux"))
        else:
            sw_flux_abs: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
            for ii in range(0, 3): # Assume Morning-Noon-Night indices
                # Absorbed shortwave flux calculation
                sw_flux_up_x: XR_DATAARRAY = sw_flux_up.isel(time = ii, x = x_indices[ii]) # [ilev, ny]
                sw_flux_dn_x: XR_DATAARRAY = sw_flux_dn.isel(time = ii, x = x_indices[ii]) # [ilev, ny]
                sw_flux_up_abs_x: XR_DATAARRAY = (sw_flux_up_x.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : z_mid["lev"]}) # Absorbed upwelling shortwave fluxes; [lev, y]; [W m^{-2}]
                sw_flux_dn_abs_x: XR_DATAARRAY = (-sw_flux_dn_x.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : z_mid["lev"]}) # Absorbed downwelling shortwave fluxes; [lev, y]; [W m^{-2}]

                dz_x: NP_ARRAY[NP_REAL] = dz.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # Level geometric thickness; [lev, y]; [m]
                sw_flux_abs[ii] = (sw_flux_up_abs_x + sw_flux_dn_abs_x).to_numpy().astype(NP_REAL) / dz_x # Absorbed shortwave fluxes; [lev, y]; [W m^{-3}]

    return sw_flux_abs