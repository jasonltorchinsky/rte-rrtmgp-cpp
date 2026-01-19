# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET

def get_sort_mask(xr_dpscream: XR_DATASET) -> NP_ARRAY[NP_INT]:
    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)).astype(NP_INT) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    return sort_mask