# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def get_sort_mask(dp_scream_file: str) -> NP_ARRAY[NP_INT]:
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        ## Construct a sorting mask for reordering "ncol" into x- and y-columns
        lon: XR_DATAARRAY = xr_dp_scream["lon"] # Column-center - x-dimension [m]; (ncol)
        lat: XR_DATAARRAY = xr_dp_scream["lat"] # Column center - y-dimension [m]; (ncol)

        sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

        return sort_mask