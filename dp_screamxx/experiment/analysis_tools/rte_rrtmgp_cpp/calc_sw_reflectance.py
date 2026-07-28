# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Calculate shortwave reflectance
"""
def calc_sw_reflectance(rad_tran_infile: str,
    rad_tran_outfile: str, 
    time_indices: Optional[NP_ARRAY[NP_INT]] = None,
    solver: str = "rt") -> XR_DATAARRAY:
    assert(solver in ["rt", "ts"])

    #---------------------------------------------------------------------------
    # Get indexers for xarray data arrays
    #---------------------------------------------------------------------------
    isel_indexers: dict = {}
    if (time_indices is not None):
        isel_indexers["time"] = XR_DATAARRAY(time_indices, dims = "time")

    #---------------------------------------------------------------------------
    # Calculate upwelling top-of-domain shortwave flux
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    if solver == "rt":
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_tod_up: XR_DATAARRAY = (xr_rad_tran["rt_flux_tod_up"]
                .isel(indexers = isel_indexers)
                .load()) # Downwelling surface direct shortwave flux; [time, y, x]; [W m^{-2}]

    else: # solver == "ts"
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_tod_up: XR_DATAARRAY = (xr_rad_tran["sw_flux_up"]
                .isel(indexers = isel_indexers)
                .isel(lev = -1)
                .load()) # Upwelling shortwave flux; [time, y, x]; [W m^{-2}]

    #---------------------------------------------------------------------------
    # Calculate reflectance
    #---------------------------------------------------------------------------
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        tsi: XR_DATAARRAY = (xr_rad_tran["tsi"]
            .isel(indexers = isel_indexers)
            .load()) # Total solar irradiance; [time, y, x]; [W m^{-2}]
        mu0: XR_DATAARRAY = (xr_rad_tran["mu0"]
            .isel(indexers = isel_indexers)
            .load()) # Cosine solar zenith angle; [time, y, x]; [N/A]

    sw_reflectance: XR_DATAARRAY = (sw_flux_tod_up) / (tsi * mu0)
    sw_reflectance = (sw_reflectance
        .assign_attrs({"units" : "N/A",
                       "long_name" : "atmospheric shortwave reflectance",
                       "standard_name" : "sw_reflectance"})
        .rename("sw_reflectance"))

    return sw_reflectance