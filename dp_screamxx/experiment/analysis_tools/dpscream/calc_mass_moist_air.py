# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g

def calc_mass_moist_air(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None, detailed_calc: bool = False) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        p_int: XR_DATAARRAY = xr_dp_scream["p_int"] # Hydrostatic pressure at level interfaces; [nt, ncol, ilev]; [Pa]
        lev: XR_DATAARRAY = xr_dp_scream["lev"] # Hybrid level at level midpoints; [nt, ncol, lev]; [m]

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    p_int: XR_DATARRAY = (p_int
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
    #---------------------------------------------------------------------------
    dx: NP_REAL = NP_REAL((p_int["x"][1] - p_int["x"][0]).to_numpy()) # [m]
    dy: NP_REAL = dx # [m]

    #---------------------------------------------------------------------------
    # Calculate moist air mass
    #---------------------------------------------------------------------------
    if x_indices is None:
        pdel: XR_DATAARRAY = (p_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : lev}) # Pressure-thickness; [time, lay, y]; [Pa]
        mass_moist_air: XR_DATARRAY = (pdel * dx * dy) / g # From hydrostatic pressure definition; [kg]
        mass_moist_air = (mass_moist_air
            .assign_attrs({"units" : "kg", 
                           "long_name" : "midpoint moist air mass",
                           "standard_name" : "moist_air_mass"})
            .rename("mass_moist_air"))
    else:
        mass_moist_air: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            p_int_x: XR_DATAARRAY = p_int.isel(time = ii, x = x_indices[ii])
            pdel_x: XR_DATAARRAY = (p_int_x.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : lev}) # Pressure-thickness; [time, lay, y]; [Pa]
            
            mass_moist_air[ii] = (pdel_x * dx * dy) / g # From hydrostatic pressure definition; [kg]

    return mass_moist_air
