# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_dei(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax: Optional[NP_REAL] = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        dei: XR_DATAARRAY = xr_rad_tran["dei"] # Cloud ice water effective diameter at layers; [nt, lay, y, x]; [μm]

    #---------------------------------------------------------------------------
    # Select relevant times for fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    dei = dei.isel(time = time_indices) # [time, lay, y, x]; [μm]

    #---------------------------------------------------------------------------
    # Calculate cloud ice water effective diameter
    #---------------------------------------------------------------------------
    if x_indices is None:
        if zmax is not None:
            dei = dei.sel(lay = slice(0, zmax * 1.e3)) # zmax [km] => [m]
        return dei
    else:
        dei_list: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            dei_x: XR_DATAARRAY = dei.isel(time = ii, x = x_indices[ii])
            if zmax is not None:
                dei_x = dei_x.sel(lay = slice(0, zmax * 1.e3)) # zmax [km] => [m]
            dei_list[ii] = dei_x.to_numpy().astype(NP_REAL) # [μm]; [lay, y]

        return dei_list
