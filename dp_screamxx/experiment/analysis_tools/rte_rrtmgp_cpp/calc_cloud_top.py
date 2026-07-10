# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL

def calc_cloud_top(rad_tran_infile: str, 
    time_indices: Optional[NP_ARRAY[NP_INT]] = None,
    x_indices: Optional[NP_ARRAY[NP_INT]] = None) -> NP_REAL:

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

    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        lwp: XR_DATAARRAY = (xr_rad_tran["lwp"]
            .isel(indexers = isel_indexers)
            .load()) # Cloud liquid water path; [time, lay, y, x]; [g m^{2}]
        iwp: XR_DATAARRAY = (xr_rad_tran["iwp"]
            .isel(indexers = isel_indexers)
            .load()) # Cloud ice water path; [time, lay, y, x]; [g m^{2}]
        lay: XR_DATAARRAY = (xr_rad_tran["lay"]
            .load()) # Layer midpoints; [lay]; [m]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Uniform vertical grid in time and space
    #---------------------------------------------------------------------------
    dz: NP_REAL = NP_REAL((lay[1] - lay[0]).to_numpy()) # [m]

    #---------------------------------------------------------------------------
    # Calculate cloud water content
    #---------------------------------------------------------------------------
    cloud_wc: XR_DATARRAY = ((lwp + iwp) / dz) # [g m^{-3}]; [time, lay, y, x] / [lay, slices]
    cloud_wc = (cloud_wc
        .assign_attrs({"units" : "g m^{-3}", 
                        "long_name" : "midpoint cloud water content",
                        "standard_name" : "cloud_water_content"})
        .rename("cloud_water_content"))

    cloud_wc = cloud_wc.mean(dim = ["time", "y", "x"]) # [g m^{-3}]; [lay]

    #---------------------------------------------------------------------------
    # Cloud top is calculated similarly to tropopause height.
    # The cloud top is defined as the lowest level at which the cloud water
    # content decreases to cloud_wc_tol g m^{-3} or less, provided that the cloud water 
    # content between that level and all other higher levels exceed cloud_wc_tol g m^{-3}.
    #---------------------------------------------------------------------------
    nlay: NP_INT = NP_INT(lay.size)
    cloud_wc_tol: NP_REAL = NP_REAL(1.e-6)
    below_thresh: NP_ARRAY[NP_INT] = NP_INT(np.where(cloud_wc <= cloud_wc_tol)[0]) # All lapse rates at or below 2 K km^{-1}

    z_cloud_top: Optional[NP_REAL] = None
    ii: int
    for ii in range(0, below_thresh.size):
        lay_index: NP_INT = below_thresh[ii]
        lay_above: NP_ARRAY[NP_INT] = np.arange(lay_index, nlay, dtype = NP_INT) # Indexes of all layers above current layer
        lay_above_indices: NP_ARRAY[NP_INT] = lay_above[~np.isin(lay_above, below_thresh)] # Index of higher layers with lapse rate below 2 K km^{-1}

        if lay_above_indices.size == 0:
            z_cloud_top = NP_REAL(lay[lay_index]) * 1.e-3 # [m] => [km]
            break

    return z_cloud_top