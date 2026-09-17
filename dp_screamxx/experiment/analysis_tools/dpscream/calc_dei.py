# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_dei(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:

    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        rei: XR_DATAARRAY = xr_dp_scream["eff_radius_qi"] # Cloud ice water effective radius at level midpoints; [nt, ncol, ilev]; [μm]

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    rei: XR_DATARRAY = (rei
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Extract cloud liquid water effective radius
    #---------------------------------------------------------------------------
    if x_indices is None:
        return 2. * rei
    else:
        dei_list: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            dei_list[ii] = 2. * rei.isel(time = ii, x = x_indices[ii])

        return dei_list
