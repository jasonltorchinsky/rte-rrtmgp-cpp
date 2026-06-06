# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_REAL, XR_DATASET, XR_DATAARRAY

"""
Get vertical grid at specific time
"""
def get_z(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, levels: str = "mid") -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    assert(levels in ["mid", "int"])

    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        z: XR_DATARRAY = xr_dp_scream["z_" + levels] # Vertcial grid; [time, ncol, lev/ilev]

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    z: XR_DATARRAY = (z
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Trim to specific x_indices if necessary
    #---------------------------------------------------------------------------
    if x_indices is None:
        z_out: XR_DATAARRAY = z
    else:
        z_out: list[NP_ARRAY[NP_REAL]] = \
            [(z.isel(time = ii, x = x_indices[ii])).to_numpy().astype(NP_REAL) for ii in range(0, 3)]

    return z_out