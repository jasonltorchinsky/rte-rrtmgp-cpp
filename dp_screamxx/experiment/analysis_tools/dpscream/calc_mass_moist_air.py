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
        lev: XR_DATAARRAY = xr_dp_scream["lev"].load() # Hybrid level at level midpoints

        if ("y" in xr_dp_scream) and ("x" in xr_dp_scream):
            y_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["y"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
            x_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["x"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
        elif ("lat" in xr_dp_scream) and ("lon" in xr_dp_scream):
            y_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["lat"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
            x_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["lon"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
        else:
            raise ValueError("DP-SCREAM file must contain either x/y or lon/lat fields on ncol.")

        p_int: XR_DATAARRAY = (xr_dp_scream["p_int"] # Hydrostatic pressure at level interfaces; [time, ncol, ilev]; [Pa]
            .isel(time = time_indices, ncol = sort_mask)
            .assign_coords({
                "y" : ("ncol", y_coord),
                "x" : ("ncol", x_coord)
            })
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")
            .transpose("time", "ilev", "y", "x")
            .load()) # [time, ilev, y, x]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
    #---------------------------------------------------------------------------
    if p_int.sizes["x"] > 1:
        dx: NP_REAL = NP_REAL(abs((p_int["x"][1] - p_int["x"][0]).to_numpy())) # [m]
    else:
        dx: NP_REAL = NP_REAL(1.)

    if p_int.sizes["y"] > 1:
        dy: NP_REAL = NP_REAL(abs((p_int["y"][1] - p_int["y"][0]).to_numpy())) # [m]
    else:
        dy: NP_REAL = dx

    #---------------------------------------------------------------------------
    # Calculate moist air mass
    #---------------------------------------------------------------------------
    if x_indices is None:
        pdel: XR_DATAARRAY = (abs(p_int.diff("ilev"))
            .rename({"ilev" : "lev"})
            .assign_coords({"lev" : lev})) # Pressure-thickness; [time, lev, y, x]; [Pa]

        mass_moist_air: XR_DATAARRAY = (pdel * dx * dy) / g # From hydrostatic pressure definition; [kg]
        mass_moist_air = (mass_moist_air
            .assign_attrs({"units" : "kg", 
                           "long_name" : "midpoint moist air mass",
                           "standard_name" : "moist_air_mass"})
            .rename("mass_moist_air"))
    else:
        mass_moist_air: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            p_int_x: XR_DATAARRAY = p_int.isel(time = ii, x = x_indices[ii])
            pdel_x: XR_DATAARRAY = (abs(p_int_x.diff("ilev"))
                .rename({"ilev" : "lev"})
                .assign_coords({"lev" : lev})) # Pressure-thickness; [lev, y]; [Pa]
            
            mass_moist_air[ii] = ((pdel_x * dx * dy) / g).to_numpy().astype(NP_REAL) # [kg]; [lev, y]

    return mass_moist_air