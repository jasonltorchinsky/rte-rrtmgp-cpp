# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_rel(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:

    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        rel: XR_DATAARRAY = xr_dp_scream["eff_radius_qc"] # Cloud liquid water effective radius at level midpoints; [nt, ncol, ilev]; [μm]

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    rel: XR_DATARRAY = (rel
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Extract cloud liquid water effective radius
    #---------------------------------------------------------------------------
    if x_indices is None:
        return rel
    else:
        rel_list: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            rel_list[ii] = rel.isel(time = ii, x = x_indices[ii])

        return rel_list
