# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL, NP_INF

from .calc_cloud_top import calc_cloud_top
from .calc_tropopause import calc_tropopause

"""
Calculate appropriate z_max information
"""
def calc_z_max_info(rad_tran_infile: str,
    z_max: Optional[NP_REAL] = None,
    method: Optional[str] = "default") -> dict:
    # z_max is in [km], convert to [m] locally

    if z_max is None:
        assert(method in ["default", "tropopause", "cloud_top"])

        z_max: NP_REAL
        if method == "default": # Maximum of tropopause height and cloud top height; [km]
            z_max = max(calc_tropopause(rad_tran_infile), calc_cloud_top(rad_tran_infile))
        elif method == "tropopause":
            z_max = calc_tropopause(rad_tran_infile) # Tropopause height; [km]
        else: # method == "cloud_top"
            z_max = calc_cloud_top(rad_tran_infile) # Cloud top height; [km]

    #---------------------------------------------------------------------------
    # Fit z_max to where it actually is in the input file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        lay: XR_DATAARRAY = (xr_rad_tran["lay"]
            .load()) # Layer midpoints; [lay]; [m]
        lev: XR_DATAARRAY = (xr_rad_tran["lev"]
            .load()) # Layer interfaces; [lev]; [m]
        
    diff_arr: NP_ARRAY[NP_REAL] = NP_REAL(lay.to_numpy()) - z_max * 1.e3
    neg_diff: NP_ARRAY[NP_REAL] = np.where(diff_arr <= NP_SMALL, diff_arr, -NP_INF)

    z_max_index: NP_INT = NP_INT(np.argmax(neg_diff))
    z_max: NP_REAL = NP_REAL(lay[z_max_index] + NP_SMALL) * 1.e-3 # [m] => [km]

    zh_max_index: NP_INT = z_max_index + 1
    zh_max: NP_REAL = (NP_REAL(lev[zh_max_index]) + NP_SMALL) * 1.e-3 # [m] => [km]

    #---------------------------------------------------------------------------
    # Store in a dict
    #---------------------------------------------------------------------------
    z_max_info: dict = {
        "z_max" : z_max,
        "z_max_index" : z_max_index,
        "zh_max" : zh_max,
        "zh_max_index" : zh_max_index,
        "sel_indexers" : {
            "z" : slice(-NP_INF, z_max * 1.e3), # [km] => [m]
            "zh" : slice(-NP_INF, zh_max * 1.e3), # [km] => [m]
            "lay" : slice(-NP_INF, z_max * 1.e3), # [km] => [m]
            "lev" : slice(-NP_INF, zh_max * 1.e3), # [km] => [m]
        },
        "isel_indexers" : {
            "z" : slice(0, z_max_index + 1),
            "zh" : slice(0, zh_max_index + 1),
            "lay" : slice(0, z_max_index + 1),
            "lev" : slice(0, zh_max_index + 1),
        }
    }

    return z_max_info