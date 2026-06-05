# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import sec_per_day

"""
Calculate shortwave heating rates.
"""
def calc_sw_heating(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT], time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, method: str = "pdel") -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    assert(method in ["pdel", "flux"])

    xr_dp_scream: XR_DATASET
    if method == "pdel":
        with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
            sw_heating_pdel: XR_DATAARRAY = xr_dp_scream["SW_heating_pdel"] # Shortwave heating times pressure-thickness; [time, ncol, lev]; [Pa K s^{-1}]
            p_int: XR_DATAARRAY = xr_dp_scream["p_int"] # Pressure at level interfaces; [time, ncol, ilev]; [Pa]

        # Sort, reshape, and resize matrices
        sw_heating_pdel: XR_DATAARRAY = (sw_heating_pdel
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [3, lev, y, x]
        p_int: XR_DATAARRAY = (p_int
            .isel(time = time_indices, ncol = sort_mask)
            .rename({"lat": "y", "lon": "x"})
            .set_index(ncol = ["y", "x"])
            .unstack("ncol")) # [3, ilev, y, x]
        
        #-----------------------------------------------------------------------
        # Calculate shortwave heating rate
        #-----------------------------------------------------------------------
        if x_indices is None:
            breakpoint()
            # CHECK SIGN OF pdel
            pdel: XR_DATAARRAY = (p_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : sw_heating_pdel["lev"]}) # Pressure-thickness; [time, lev, y]; [Pa]
            sw_heating: XR_DATAARRAY = (sw_heating_pdel / pdel) * sec_per_day # [time, lev, y, x]; [K d^{-1}]
            sw_heating = (sw_heating
                .assign_attrs({"units" : "K d^{-1}", 
                               "long_name" : "shortwave heating rate",
                               "standard_name" : "shortwave_heating_rate"})
                .rename("shortwave_heating_rate"))
        else:
            sw_heating: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
            for ii in range(0, 3): # Assume Morning-Noon-Night indices
                sw_heating_pdel_x: XR_DATAARRAY = sw_heating_pdel.isel(time = ii, x = x_indices[ii]) # [lev, ny]
                p_int_x: XR_DATAARRAY = p_int.isel(time = ii, x = x_indices[ii]) # [lev, ny]
                pdel_x: XR_DATAARRAY = (p_int_x.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : sw_heating_pdel["lev"]}) # Pressure-thickness; [lev, ny]; [Pa]

                sw_heating[ii] = ((sw_heating_pdel_x / pdel_x) * sec_per_day).to_numpy().astype(NP_REAL) # [lev, y]; [K d^{-1}]

    return sw_heating