# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g, R_d, R_v, cp_d, cp_v, cp_lw, cp_iw, sec_per_day
from .calc_mass_moist_air import calc_mass_moist_air

"""
Calculate shortwave absorbed flux
"""
def calc_sw_flux_abs(rad_tran_infile: str, rad_tran_outfile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, solver: str = "rt", zmax: Optional[NP_REAL] = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
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
        sw_flux_abs_dir = sw_flux_abs_dir.isel(time = time_indices) # [time, z, y, x]
        sw_flux_abs_dif = sw_flux_abs_dif.isel(time = time_indices) # [time, z, y, x]

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
        sw_flux_dn_diff: XR_DATARRAY = sw_flux_dn.diff("lev").rename({"lev": "z"}).assign_coords(z = z) # dn[i+1] - dn[i]; [time, z, y, x]; [W m^{-2}] 
        sw_flux_up_diff: XR_DATARRAY = -sw_flux_up.diff("lev").rename({"lev": "z"}).assign_coords(z = z) # up[i] - up[i+1]; [time, z, y, x]; [W m^{-2}]
        sw_flux_abs: XR_DATARRAY = (sw_flux_dn_diff + sw_flux_up_diff) / dz # [time, z, y, x]; [W m^{-3}]

    sw_flux_abs = (sw_flux_abs
        .rename({"z" : "lay"})
    ) # [time, lay, y, x]; [W m^{-3}]

    #---------------------------------------------------------------------------
    # Calculate absorbed shortwave flux
    #---------------------------------------------------------------------------
    if x_indices is None:
        if zmax is not None:
            sw_flux_abs = sw_flux_abs.sel(lay = slice(0, zmax * 1.e3)) # zmax is in km
        return sw_flux_abs
    else:
        sw_flux_abs_list: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            sw_flux_abs_x: XR_DATAARRAY = sw_flux_abs.isel(time = ii, x = x_indices[ii])
            if zmax is not None:
                sw_flux_abs_x = sw_flux_abs_x.sel(lay = slice(0, zmax * 1.e3))
            sw_flux_abs_list[ii] = (sw_flux_abs_x).to_numpy().astype(NP_REAL) # Absorbed shortwave fluxes; [lev, y]; [W m^{-3}]

        return sw_flux_abs_list