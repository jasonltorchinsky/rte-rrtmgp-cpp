# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL, NP_INF

def calc_cloud_wc(rad_tran_infile: str, 
    time_indices: Optional[NP_ARRAY[NP_INT]] = None,
    x_indices: Optional[NP_ARRAY[NP_INT]] = None, 
    z_max: Optional[NP_REAL] = None) -> XR_DATAARRAY:
    # z_max is in [km], convert to [m] locally
    #---------------------------------------------------------------------------
    # z_max corresponds to layers, find the z_max that corresponds to levels
    #---------------------------------------------------------------------------
    if z_max is not None:
        xr_rad_tran: XR_DATASET
        with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            lay: XR_DATAARRAY = (xr_rad_tran["lay"]
                .load()) # Layer midpoints; [lay]; [m]

        diff_arr: NP_ARRAY[NP_REAL] = NP_REAL(lay.to_numpy()) - z_max * 1.e3
        neg_diff: NP_ARRAY[NP_REAL] = np.where(diff_arr <= NP_SMALL, diff_arr, -NP_INF)

        z_max_index: NP_INT = NP_INT(np.argmax(neg_diff))
        z_max: NP_REAL = NP_REAL(lay[z_max_index] + NP_SMALL) * 1.e-3 # [m] => [km]

    #---------------------------------------------------------------------------
    # Get indexers for xarray data arrays
    #---------------------------------------------------------------------------
    isel_indexers: dict = {}
    if ((time_indices is not None) and (x_indices is not None)):
        isel_indexers["time"] = XR_DATAARRAY(time_indices, dims = "slice")
        isel_indexers["x"] = XR_DATAARRAY(x_indices, dims = "slice")
    elif ((time_indices is not None) and (x_indices is None)):
        isel_indexers["time"] = XR_DATAARRAY(time_indices, dims = "time")
    elif ((time_indices is None) and (x_indices is not None)):
        isel_indexers["x"] = XR_DATAARRAY(x_indices, dims = "x")

    sel_indexers: dict = {}
    if z_max is not None:
        sel_indexers["lay"] = slice(0, z_max * 1.e3) # [km] => [m]
    else:
        sel_indexers["lay"] = slice(0, None)

    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        lwp: XR_DATAARRAY = (xr_rad_tran["lwp"]
            .isel(indexers = isel_indexers)
            .sel(indexers = sel_indexers)
            .load()) # Cloud liquid water path; [nt, lay, y, x]; [g m^{2}]
        iwp: XR_DATAARRAY = (xr_rad_tran["iwp"]
            .isel(indexers = isel_indexers)
            .sel(indexers = sel_indexers)
            .load()) # Cloud ice water path; [nt, lay, y, x]; [g m^{2}]
        lay: XR_DATAARRAY = (xr_rad_tran["lay"]
            .sel(indexers = sel_indexers)
            .load()) # Layer midpoints; [lay]; [m]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Uniform vertical grid in time and space
    #---------------------------------------------------------------------------
    dz: NP_REAL = NP_REAL((lay[1] - lay[0]).to_numpy())

    #---------------------------------------------------------------------------
    # Calculate cloud water content
    #---------------------------------------------------------------------------
    cloud_wc: XR_DATARRAY = ((lwp + iwp) / dz) # [g m^{-3}]; [time, lay, y, x] / [lay, slices]
    cloud_wc = (cloud_wc
        .assign_attrs({"units" : "g m^{-3}", 
                        "long_name" : "midpoint cloud water content",
                        "standard_name" : "cloud_water_content"})
        .rename("cloud_water_content"))

    return cloud_wc