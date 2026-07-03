# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_rel(rad_tran_infile: str, 
    time_indices: Optional[NP_ARRAY[NP_INT]] = None,
    x_indices: Optional[NP_ARRAY[NP_INT]] = None, 
    z_max: Optional[NP_REAL] = None) -> XR_DATAARRAY:
    # z_max is in [km], convert to [m] locally
    #---------------------------------------------------------------------------
    # Get indexers for xarray data arrays
    #---------------------------------------------------------------------------
    isel_indexers: dict = {}
    if ((time_indices is not None) and (x_indices is not None)):
        isel_indexers["time"] = XR_DATAARRAY(time_indices, dims = "slice")
        isel_indexers["x"] = XR_DATAARRAY(x_indices, dims = "slice")
    elif ((time_indices is not None) and (x_indices is None)):
        isel_indexers["time"] = XR_DATAARRAY(time_indices, dims = "time")
    elif ((time_indices is None) and (x_indices is not None)):
        isel_indexers["x"] = XR_DATAARRAY(x_indices, dims = "x")

    sel_indexers: dict = {}
    if z_max is not None:
        sel_indexers["lay"] = slice(0, z_max * 1.e3) # [km] => [m]
    else:
        sel_indexers["lay"] = slice(0, None)

    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        rel: XR_DATAARRAY = (xr_rad_tran["rel"]
            .isel(indexers = isel_indexers)
            .sel(indexers = sel_indexers)
            .load()) # Cloud liquid water effective radius at layers; [nt, lay, y, x]; [μm]

    return rel
