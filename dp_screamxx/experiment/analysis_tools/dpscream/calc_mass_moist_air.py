# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import R_d, R_v

def calc_mass_moist_air(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None, detailed_calc: bool = False) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        p_mid: XR_DATAARRAY = xr_dp_scream["p_mid"] # Pressure at level midpoints; [nt, ncol, lev]; [Pa]
        qv: XR_DATAARRAY = xr_dp_scream["qv"] # Water vapor moist mixing ratio at midpoints; [nt, ncol, lev]; [kg kg^{-1}]
        T_mid: XR_DATAARRAY = xr_dp_scream["T_mid"] # Temperature at level midpoints; [nt, ncol, lev]; [K]
        z_int: XR_DATAARRAY = xr_dp_scream["z_int"] # Geometric height at level interfaces; [nt, ncol, ilev]; [m]

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    p_mid: XR_DATARRAY = (p_mid
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]
    T_mid: XR_DATARRAY = (T_mid
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]
    qv: XR_DATARRAY = (qv
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]
    z_int: XR_DATARRAY = (z_int
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, ilev, y, x]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
    #---------------------------------------------------------------------------
    dx: NP_REAL = NP_REAL((z_int["x"][1] - z_int["x"][0]).to_numpy()) # [m]
    dy: NP_REAL = dx # [m]
    dz: XR_DATAARRAY = (-z_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : p_mid["lev"]}) # Vertical grid spacing; [m]; [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Calculate moist air mass
    #---------------------------------------------------------------------------
    if x_indices is None:
        mass_moist_air: XR_DATARRAY = (p_mid * dx * dy * dz) / ((R_d + (R_v - R_d) * qv) * T_mid) # [kg]
        mass_moist_air = (mass_moist_air
            .assign_attrs({"units" : "kg", 
                           "long_name" : "midpoint moist air mass",
                           "standard_name" : "moist_air_mass"})
            .rename("mass_moist_air"))
    else:
        mass_moist_air: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            p_mid_x: XR_DATAARRAY = p_mid.isel(time = ii, x = x_indices[ii])
            T_mid_x: XR_DATAARRAY = T_mid.isel(time = ii, x = x_indices[ii])
            qv_x: XR_DATAARRAY = qv.isel(time = ii, x = x_indices[ii])
            dz_x: XR_DATARRAY = dz.isel(time = ii, x = x_indices[ii])

            mass_moist_air[ii] = ((p_mid_x * dx * dy * dz_x) / ((R_d + (R_v - R_d) * qv_x) * T_mid_x)).to_numpy().astype(NP_REAL) # [kg]

    return mass_moist_air
