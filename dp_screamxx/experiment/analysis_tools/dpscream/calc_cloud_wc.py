# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g, R_d
from .calc_mass_moist_air import calc_mass_moist_air

def calc_cloud_wc(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None, detailed_calc: bool = False) -> list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        qc: XR_DATAARRAY = xr_dp_scream["qc"] # Cloud liquid water moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
        qi: XR_DATAARRAY = xr_dp_scream["qi"] # Cloud ice water moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
        z_int: XR_DATAARRAY = xr_dp_scream["z_int"] # Geometrix height at level interfaces; [nt, ncol, ilev]; [m]

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    qc: XR_DATARRAY = (qc
        .isel(time = time_indices, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")) # [time, lev, y, x]
    qi: XR_DATARRAY = (qi
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
    dz: XR_DATAARRAY = (-z_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : qc["lev"]}) # Vertical grid spacing; [m]; [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Calculate cloud water content
    #---------------------------------------------------------------------------
    mass_moist_air: XR_DATARRAY | list[NP_ARRAY[NP_REAL]] = calc_mass_moist_air(dp_scream_file,
        sort_mask, time_indices, x_indices, zmax_index, detailed_calc) # [kg]
    
    if x_indices is None:
        cloud_wc: XR_DATARRAY = (((qc + qi) * mass_moist_air) / (dx * dy * dz)) * 1.e-3 # [g m^{-3}]; [time, lev, y, x]
        cloud_wc = (cloud_wc
            .assign_attrs({"units" : "g m^{-3}", 
                           "long_name" : "midpoint cloud water content",
                           "standard_name" : "cloud_water_content"})
            .rename("cloud_water_content"))
    else:
        cloud_wc: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            qc_x: XR_DATAARRAY = qc.isel(time = ii, x = x_indices[ii])
            qi_x: XR_DATAARRAY = qi.isel(time = ii, x = x_indices[ii])
            dz_x: XR_DATAARRAY = dz.isel(time = ii, x = x_indices[ii])
            mass_moist_air_x: NP_ARRAY[NP_REAL] = mass_moist_air[ii]

            cloud_wc[ii] = (((qc_x + qi_x) / (dx * dy * dz_x)) * 1.e-3).to_numpy().astype(NP_REAL) * mass_moist_air_x # [g m^{-3}]; [lev, y]

    return cloud_wc
