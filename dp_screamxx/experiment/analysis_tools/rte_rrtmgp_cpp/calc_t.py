# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_INF

def calc_t(rad_tran_infile: str, 
    time_indices: Optional[NP_ARRAY[NP_INT]] = None,
    x_indices: Optional[NP_ARRAY[NP_INT]] = None, 
    z_max_info: Optional[dict] = None) -> XR_DATAARRAY:
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
    if z_max_info is not None:
        sel_indexers["zh"] = z_max_info["sel_indexers"]["zh"]
        sel_indexers["lev"] = z_max_info["sel_indexers"]["lev"]
    else:
        sel_indexers["zh"] = slice(-NP_INF, None)
        sel_indexers["lev"] = slice(-NP_INF, None)

    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        t_lev: XR_DATAARRAY = (xr_rad_tran["t_lev"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lev" : sel_indexers["lev"]})
            .load()) # Temperature at levels; [time, lev, y, x]; [K]

    return t_lev
