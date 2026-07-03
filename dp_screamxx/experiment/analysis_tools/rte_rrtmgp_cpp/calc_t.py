# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_t(rad_tran_infile: str, 
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
            lev: XR_DATAARRAY = (xr_rad_tran["lev"]
                .load()) # Layer interfaces; [lev]; [m]
            z_max_index: NP_INT = NP_INT(lay
                    .indexes["lay"]
                    .searchsorted(z_max * 1.e3, side = "right"))
            zh_max_index: NP_INT = z_max_index + 1
            zh_max: NP_REAL = (NP_REAL(lev[zh_max_index - 1]) + NP_SMALL) * 1.e-3 # [m] => [km]
            # Honestly, I don't know what's going on with the indexing here. This seems to work though

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
        sel_indexers["zh"] = slice(0, zh_max * 1.e3) # [km] => [m]
        sel_indexers["lev"] = slice(0, zh_max * 1.e3) # [km] => [m]
    else:
        sel_indexers["zh"] = slice(0, None)
        sel_indexers["lev"] = slice(0, None)

    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        t_lev: XR_DATAARRAY = (xr_rad_tran["t_lev"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lev" : sel_indexers["lev"]})
            .load()) # Temperature at levels; [nt, lev, y, x]; [K]

    return t_lev
