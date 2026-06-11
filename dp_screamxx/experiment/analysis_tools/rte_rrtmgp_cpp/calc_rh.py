# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_rh(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        rh: XR_DATAARRAY = xr_rad_tran["rh"] # Relative humidity; [nt, lay, y, x]; [Pa Pa^{-1}]

    #---------------------------------------------------------------------------
    # Select relevant times for fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    rh = rh.isel(time = time_indices) # [time, lay, y, x]; [Pa Pa^{-1}]

    #---------------------------------------------------------------------------
    # Calculate cloud water content
    #---------------------------------------------------------------------------
    if x_indices is None:
        return rh
    else:
        rh_list: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            rh_list[ii] = rh.isel(time = ii, x = x_indices[ii]).to_numpy().astype(NP_REAL) # [Pa Pa^{-1}]; [lay, y]

        return rh_list
