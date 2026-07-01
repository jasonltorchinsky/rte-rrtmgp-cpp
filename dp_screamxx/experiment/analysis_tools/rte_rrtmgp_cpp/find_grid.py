# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import XR_DATASET, XR_DATAARRAY

"""
Find grids used for RTE-RRTMGP-CPP.
"""
def find_grid(rad_tran_infile: str) -> dict:
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        xh: XR_DATAARRAY = xr_rad_tran["xh"].load() # x-position of column interfaces; [m]; [n_xh]
        x: XR_DATAARRAY = xr_rad_tran["x"].load() # x-position of column midpoints; [m]; [n_x]
        yh: XR_DATAARRAY = xr_rad_tran["yh"].load() # y-position of column interfaces; [m]; [n_yh]
        y: XR_DATAARRAY = xr_rad_tran["y"].load() # y-position of column midpoints; [m]; [n_y]
        zh: XR_DATAARRAY = xr_rad_tran["zh"].load() # Geometric height of layer interfaces; [m]; [n_zh]
        z: XR_DATAARRAY = xr_rad_tran["z"].load() # Geometric height of layer midpoints; [m]; [n_z]

    grid: dict = {
        "xh" : xh,
        "x" : x,
        "yh" : yh,
        "y" : y,
        "zh" : zh,
        "z" : z,
    }

    return grid