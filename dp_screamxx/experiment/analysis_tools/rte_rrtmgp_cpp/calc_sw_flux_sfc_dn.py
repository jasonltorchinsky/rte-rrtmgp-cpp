# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Calculate downwelling surface flux
"""
def calc_sw_flux_sfc_dn(rad_tran_infile: str, rad_tran_outfile: str, time_indices: NP_ARRAY[NP_INT],
    solver: str = "rt") -> XR_DATAARRAY:
    assert(solver in ["rt", "ts"])

    #---------------------------------------------------------------------------
    # Calculate downwelling surface shortwave flux
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    if solver == "rt":
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_sfc_dir: XR_DATARRAY = xr_rad_tran["rt_flux_sfc_dir"] # Downwelling surface direct shortwave flux; [time, y, x]; [W m^{-2}]
            sw_flux_sfc_dif: XR_DATARRAY = xr_rad_tran["rt_flux_sfc_dif"] # Downwelling surface diffuse shortwave flux; [time, y, x]; [W m^{-2}]

        #-----------------------------------------------------------------------
        # Select relevant times for fields from RTE-RRTMGP-CPP file
        #-----------------------------------------------------------------------
        sw_flux_sfc_dir = sw_flux_sfc_dir.isel(time = time_indices) # [time, y, x]
        sw_flux_sfc_dif = sw_flux_sfc_dif.isel(time = time_indices) # [time, y, x]

        #-----------------------------------------------------------------------
        # Get downwelling surface shortwave flux
        #-----------------------------------------------------------------------
        sw_flux_sfc_dn: XR_DATARRAY = sw_flux_sfc_dir + sw_flux_sfc_dif

    else: # solver == "ts"
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_dn: XR_DATARRAY = xr_rad_tran["sw_flux_dn"] # Downwelling shortwave flux; [time, lev, y, x]; [W m^{-2}]

        #-----------------------------------------------------------------------
        # Select relevant times for fields from RTE-RRTMGP-CPP file
        #-----------------------------------------------------------------------
        sw_flux_dn = sw_flux_dn.isel(time = time_indices) # [time, lev, y, x]

        #-----------------------------------------------------------------------
        # Get downwelling surface shortwave flux
        #-----------------------------------------------------------------------
        sw_flux_sfc_dn: XR_DATARRAY = sw_flux_dn.isel(lev = 0) # [time, y, x]; [W m^{-2}]

    return sw_flux_sfc_dn