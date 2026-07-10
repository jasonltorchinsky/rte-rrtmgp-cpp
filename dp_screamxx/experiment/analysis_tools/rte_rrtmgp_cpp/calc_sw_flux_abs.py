# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL, NP_INF

"""
Calculate shortwave absorbed flux
"""
def calc_sw_flux_abs(rad_tran_infile: str,
    rad_tran_outfile: str,
    time_indices: Optional[NP_ARRAY[NP_INT]] = None,
    x_indices: Optional[NP_ARRAY[NP_INT]] = None,
    z_max_info: Optional[dict] = None,
    solver: str = "rt") -> XR_DATAARRAY:
    assert(solver in ["rt", "ts"])
    
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
    out_isel_indexers: dict = {}
    if z_max_info is not None:
        sel_indexers["z"] = z_max_info["sel_indexers"]["z"]
        sel_indexers["zh"] = z_max_info["sel_indexers"]["zh"]
        sel_indexers["lay"] = z_max_info["sel_indexers"]["lay"]
        sel_indexers["lev"] = z_max_info["sel_indexers"]["lev"]

        out_isel_indexers["lay"] = z_max_info["isel_indexers"]["lay"]
        out_isel_indexers["lev"] = z_max_info["isel_indexers"]["lev"]
    else:
        sel_indexers["z"] = slice(-NP_INF, None)
        sel_indexers["zh"] = slice(-NP_INF, None)
        sel_indexers["lay"] = slice(-NP_INF, None)
        sel_indexers["lev"] = slice(-NP_INF, None)

        out_isel_indexers["lay"] = slice(0, None)
        out_isel_indexers["lev"] = slice(0, None)

    #---------------------------------------------------------------------------
    # Calculate absorbed shorwave flux
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    if solver == "rt":
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_abs_dir: XR_DATARRAY = (xr_rad_tran["rt_flux_abs_dir"]
                .isel(indexers = isel_indexers)
                .sel(indexers = {"z" : sel_indexers["z"]})
                .load()) # Absorbed direct shortwave flux; [time, z, y, x]; [W m^{-3}]
            sw_flux_abs_dif: XR_DATARRAY = (xr_rad_tran["rt_flux_abs_dif"]
                .isel(indexers = isel_indexers)
                .sel(indexers = {"z" : sel_indexers["z"]})
                .load()) # Absorbed diffuse shortwave flux; [time, z, y, x]; [W m^{-3}]

        #-----------------------------------------------------------------------
        # Get absorbed shortwave flux
        #-----------------------------------------------------------------------
        sw_flux_abs: XR_DATARRAY = sw_flux_abs_dir + sw_flux_abs_dif # [time, z, y, x]; [W m^{-3}]

    else: # solver == "ts"
        with xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
            sw_flux_up: XR_DATARRAY = (xr_rad_tran["sw_flux_up"]
                .isel(indexers = isel_indexers)
                .isel(indexers = {"lev" : out_isel_indexers["lev"]})
                .load()) # Upwelling shortwave flux; [time, lev, y, x]; [W m^{-2}]
            sw_flux_dn: XR_DATARRAY = (xr_rad_tran["sw_flux_dn"]
                .isel(indexers = isel_indexers)
                .isel(indexers = {"lev" : out_isel_indexers["lev"]})
                .load()) # Downwelling shortwave flux; [time, lev, y, x]; [W m^{-2}]
            z: XR_DATARRAY = (xr_rad_tran["z"]
                .sel(indexers = {"z" : sel_indexers["z"]})
                .load()) # Geometric height at layer midpoints; [z]; [m]

        #-----------------------------------------------------------------------
        # Get absorbed shortwave flux
        #-----------------------------------------------------------------------
        dz: NP_REAL = NP_REAL((z[1] - z[0]).to_numpy()) # [m]
        sw_flux_dn_diff: XR_DATARRAY = (sw_flux_dn.diff("lev")
            .rename({"lev" : "z"})
            .assign_coords(z = z)) # dn[i+1] - dn[i]; [time, z, y, x]; [W m^{-2}] 
        sw_flux_up_diff: XR_DATARRAY = (-sw_flux_up.diff("lev")
            .rename({"lev" : "z"})
            .assign_coords(z = z)) # up[i] - up[i+1]; [time, z, y, x]; [W m^{-2}]
        sw_flux_abs: XR_DATARRAY = (sw_flux_dn_diff + sw_flux_up_diff) / dz # [time, z, y, x]; [W m^{-3}]

    sw_flux_abs = (sw_flux_abs
        .rename({"z" : "lay"})
    ) # [time, lay, y, x]; [W m^{-3}]

    return sw_flux_abs