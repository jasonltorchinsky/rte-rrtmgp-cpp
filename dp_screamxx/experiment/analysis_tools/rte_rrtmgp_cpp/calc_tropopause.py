# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_tropopause(rad_tran_infile: str, 
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
        t_lev: XR_DATAARRAY = (xr_rad_tran["t_lev"]
            .isel(indexers = isel_indexers)
            .load()) # Temperature at levels; [time, lev, y, x]; [K]
        lay: XR_DATAARRAY = (xr_rad_tran["lay"]
            .load()) # Layer midpoints; [lay]; [m]
        lev: XR_DATAARRAY = (xr_rad_tran["lev"]
            .load()) # Layer interfaces; [lev]; [m]

    #---------------------------------------------------------------------------
    # Calculate lapse rate at layers
    #---------------------------------------------------------------------------
    dz: NP_REAL = NP_REAL(lev[1] - lev[0]) * 1.e-3 # Layer thickness; [m] => [km]
    lapse_rate: XR_DATAARRAY = (-t_lev.diff("lev")
        .rename({"lev" : "lay"})
        .assign_coords({"lay" : NP_REAL(lay.to_numpy())})) / (2. * dz) # Lapse rate at layer midpoints; [K km^{-1}]; [time, lay, y, x]

    lapse_rate: XR_DATAARRAY = lapse_rate.mean(dim = ["time", "y", "x"])

    #---------------------------------------------------------------------------
    # Calculate tropopause height.
    # "The tropopause is defined as the lowest level at which the lapse rate 
    # decreases to 2 K km^{-1} or less, provided that the average lapse-rate 
    # between that level and all other higher levels within 2.0 km does not 
    # exceed 2 K km^{-1}." ISBN 978-92-63-02182-3
    #---------------------------------------------------------------------------
    nlay: NP_INT = NP_INT(lay.size)
    below_thresh: NP_ARRAY[NP_INT] = NP_INT(np.where(lapse_rate <= 2.0)[0]) # All lapse rates at or below 2 K km^{-1}

    z_tropopause: Optional[NP_REAL] = None
    ii: int
    for ii in range(0, below_thresh.size):
        lay_index: NP_INT = below_thresh[ii]
        lay_above: NP_ARRAY[NP_INT] = np.arange(lay_index, nlay, dtype = NP_INT) # Indexes of all layers above current layer
        lay_above_indices: NP_ARRAY[NP_INT] = lay_above[~np.isin(lay_above, below_thresh)] # Index of higher layers with lapse rate below 2 K km^{-1}

        if lay_above_indices.size > 0:
            lay_above_index: NP_INT = lay_above_indices[0]
        
            dist_btw_layers: NP_REAL = NP_REAL(lay[lay_above_index] - lay[lay_index]) * 1.e-3 # Distance between current layer and next highest layer with lapse rate less than 2 K km^{-1}

            if dist_btw_layers >= 2.0:
                z_tropopause = NP_REAL(lay[lay_index]) * 1.e-3 # [m] => [km]
                break
        else:
            z_tropopause = NP_REAL(lay[lay_index]) * 1.e-3 # [m] => [km]
            break

    return z_tropopause
