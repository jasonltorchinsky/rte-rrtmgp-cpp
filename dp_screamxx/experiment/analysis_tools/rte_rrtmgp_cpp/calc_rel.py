# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_rel(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax: Optional[NP_REAL] = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        rel: XR_DATAARRAY = xr_rad_tran["rel"] # Cloud liquid water effective radius at layers; [nt, lay, y, x]; [μm]

    #---------------------------------------------------------------------------
    # Select relevant times for fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    rel = rel.isel(time = time_indices) # [time, lay, y, x]; [μm]

    #---------------------------------------------------------------------------
    # Calculate cloud liquid water effective radius
    #---------------------------------------------------------------------------
    if x_indices is None:
        if zmax is not None:
            rel = rel.sel(lay = slice(0, zmax * 1.e3)) # zmax [km] => [m]
        return rel
    else:
        rel_list: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            rel_x: XR_DATAARRAY = rel.isel(time = ii, x = x_indices[ii])
            if zmax is not None:
                rel_x = rel_x.sel(lay = slice(0, zmax * 1.e3)) # zmax [km] => [m]
            rel_list[ii] = rel_x.to_numpy().astype(NP_REAL) # [μm]; [lay, y]

        return rel_list
