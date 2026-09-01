# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

"""
Calculate downwelling surface flux
"""
def calc_sw_flux_sfc_dn(ml3drt_outfile: str, 
    time_indices: Optional[NP_ARRAY[NP_INT]] = None) -> XR_DATAARRAY:

    #---------------------------------------------------------------------------
    # Get indexers for xarray data arrays
    #---------------------------------------------------------------------------
    isel_indexers: dict = {}
    if (time_indices is not None):
        isel_indexers["time"] = XR_DATAARRAY(time_indices, dims = "time")

    #---------------------------------------------------------------------------
    # Calculate downwelling surface shortwave flux
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(ml3drt_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        sw_flux_sfc_dn: XR_DATARRAY = (xr_rad_tran["rt_flux_sfc_dn_total_pred"]
            .isel(indexers = isel_indexers)
            .load()) # Downwelling surface  shortwave flux; [time, y, x]; [W m^{-2}]


    return sw_flux_sfc_dn