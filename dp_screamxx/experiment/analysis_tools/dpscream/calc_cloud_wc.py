# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g, R_d
from .calc_mass_moist_air import calc_mass_moist_air

def calc_cloud_wc(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None, detailed_calc: bool = False) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        qc: XR_DATAARRAY = xr_dp_scream["qc"] # Cloud liquid water moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
        qi: XR_DATAARRAY = xr_dp_scream["qi"] # Cloud ice water moist mixing ratio; [nt, ncol, lev]; [kg kg^{-1}]
        z_int: XR_DATAARRAY = xr_dp_scream["z_int"] # Geometric height at level interfaces; [nt, ncol, ilev]; [m]

        if ("y" in xr_dp_scream) and ("x" in xr_dp_scream):
            y_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["y"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
            x_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["x"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
        elif ("lat" in xr_dp_scream) and ("lon" in xr_dp_scream):
            y_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["lat"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
            x_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["lon"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
        else:
            raise ValueError("DP-SCREAM file must contain either x/y or lon/lat fields on ncol.")

    #---------------------------------------------------------------------------
    # Sort and reshape fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    qc = (qc
        .isel(time = time_indices, ncol = sort_mask)
        .assign_coords({
            "y" : ("ncol", y_coord),
            "x" : ("ncol", x_coord)
        })
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")
        .transpose("time", "lev", "y", "x")) # [time, lev, y, x]

    qi = (qi
        .isel(time = time_indices, ncol = sort_mask)
        .assign_coords({
            "y" : ("ncol", y_coord),
            "x" : ("ncol", x_coord)
        })
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")
        .transpose("time", "lev", "y", "x")) # [time, lev, y, x]

    z_int = (z_int
        .isel(time = time_indices, ncol = sort_mask)
        .assign_coords({
            "y" : ("ncol", y_coord),
            "x" : ("ncol", x_coord)
        })
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")
        .transpose("time", "ilev", "y", "x")) # [time, ilev, y, x]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
    #---------------------------------------------------------------------------
    if z_int.sizes["x"] > 1:
        dx: NP_REAL = NP_REAL(np.abs((z_int["x"][1] - z_int["x"][0]).to_numpy())) # [m]
    else:
        dx: NP_REAL = NP_REAL(1.)

    if z_int.sizes["y"] > 1:
        dy: NP_REAL = NP_REAL(np.abs((z_int["y"][1] - z_int["y"][0]).to_numpy())) # [m]
    else:
        dy: NP_REAL = dx

    dz: XR_DATAARRAY = (np.abs(z_int.diff("ilev"))
        .rename({"ilev" : "lev"})
        .assign_coords({"lev" : qc["lev"]})) # Vertical grid spacing; [m]; [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Calculate cloud water content
    #---------------------------------------------------------------------------
    mass_moist_air: XR_DATAARRAY | list[NP_ARRAY[NP_REAL]] = calc_mass_moist_air(dp_scream_file,
        sort_mask, time_indices, x_indices, zmax_index, detailed_calc) # [kg]

    if x_indices is None:
        cloud_wc: XR_DATAARRAY = (((qc + qi) * mass_moist_air) / (dx * dy * dz)) * NP_REAL(1.e3) # [g m^{-3}]; [time, lev, y, x]
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

            cloud_wc[ii] = ((((qc_x + qi_x) * mass_moist_air_x) / (dx * dy * dz_x)) * NP_REAL(1.e3)).to_numpy().astype(NP_REAL) # [g m^{-3}]; [lev, y]

    return cloud_wc