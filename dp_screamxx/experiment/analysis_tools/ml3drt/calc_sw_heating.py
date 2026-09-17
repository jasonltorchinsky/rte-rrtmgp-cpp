# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g, R_d, R_v, cp_d, cp_v, cp_lw, cp_iw, sec_per_day
from consts.numeric import NP_SMALL, NP_INF

"""
Calculate shortwave heating rates.
"""
def calc_sw_heating(rad_tran_infile: str,
    ml3drt_outfile: str,
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
    with xr.open_dataset(ml3drt_outfile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        sw_flux_abs: XR_DATARRAY = (xr_rad_tran["rt_flux_abs_total_pred"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"z" : sel_indexers["z"]})
            .load()) # Absorbed direct shortwave flux; [time, z, y, x]; [W m^{-3}]

    sw_flux_abs = (sw_flux_abs
        .rename({"z" : "lay"})
    ) # [time, lay, y, x]; [W m^{-3}]

    #---------------------------------------------------------------------------
    # Calculate relevant mixing ratios - qv, qc, qi
    #---------------------------------------------------------------------------
    # ASSUME - Hydrostatic pressure.
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        p_lev: XR_DATAARRAY = (xr_rad_tran["p_lev"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lev" : sel_indexers["lev"]})
            .load()) # Hydrostatic pressure at levels; [nt, lev, y, x]; [Pa]
        p_lay: XR_DATAARRAY = (xr_rad_tran["p_lay"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lay" : sel_indexers["lay"]})
            .load()) # Hydrostatic pressure at layers; [nt, lay, y, x]; [Pa]
        t_lay: XR_DATAARRAY = (xr_rad_tran["t_lay"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lay" : sel_indexers["lay"]})
            .load()) # Temperature at layers; [nt, lay, y, x]; [K]
        lwp: XR_DATAARRAY = (xr_rad_tran["lwp"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lay" : sel_indexers["lay"]})
            .load()) # Cloud liquid water path at layers; [nt, lay, y, x]; [g m^{-2}]
        iwp: XR_DATAARRAY = (xr_rad_tran["iwp"]
            .isel(indexers = isel_indexers)
            .sel(indexers = {"lay" : sel_indexers["lay"]})
            .load()) # Cloud ice water path at layers; [nt, lay, y, x]; [g m^{-2}]
        x: XR_DATAARRAY = (xr_rad_tran["x"]
            .load()) # Column x-midpoints; [x]; [m]
        y: XR_DATAARRAY = (xr_rad_tran["y"]
            .load()) # Column y-midpoints; [y]; [m]
        lay: XR_DATAARRAY = (xr_rad_tran["lay"]
            .sel(indexers = {"lay" : sel_indexers["lay"]})
            .load()) # Layer midpoints; [lay]; [m]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Uniform vertical grid in time and space
    #---------------------------------------------------------------------------
    dx: NP_REAL = NP_REAL((x[1] - x[0]).to_numpy())
    dy: NP_REAL = NP_REAL((y[1] - y[0]).to_numpy())
    dz: NP_REAL = NP_REAL((lay[1] - lay[0]).to_numpy())

    #---------------------------------------------------------------------------
    # Calculate water vapor moist mixing ratio
    #---------------------------------------------------------------------------
    pdel: XR_DATAARRAY = (p_lev.diff("lev")
        .rename({"lev" : "lay"})
        .assign_coords({"lay" : NP_REAL(lay.to_numpy())})) # Pressure-thickness; [time, lay, y, x]; [Pa]
    rho: XR_DATARRAY = pdel / (g * dz) # Moist air density; [time, lay, y, x]; [kg m^{-3}]
    qv: XR_DATARRAY = (1. / (R_v - R_d)) * ((p_lay / (rho * t_lay)) - R_d) # Water vapor moist mixing ratio; ; [time, lay, y, x]; [kg kg^{-1}]

    mass_moist_air: XR_DATARRAY = rho * (dx * dy * dz) # Moist air mass; [time, lay, y, x]; [kg]
    qc: XR_DATAARRAY = ((lwp * 1.e-3) * dx * dy) / mass_moist_air # [kg kg^{-1}]
    qi: XR_DATAARRAY = ((iwp * 1.e-3) * dx * dy) / mass_moist_air # [kg kg^{-1}]

    #---------------------------------------------------------------------------
    # Calculate shortwave heating rate
    #---------------------------------------------------------------------------
    # Mass, specific heat, and volume calculations
    mass_total: XR_DATARRAY = (1. + qc + qi) * mass_moist_air # Mass of moist air and cloud water in cell; [time, lay, y, x]; [kg]
    cp_total: XR_DATARRAY = (cp_d * (1. - qv) + cp_v * qv + cp_lw * qc + cp_iw * qi) / (1. + qc + qi) # Specific heat at constant pressure of cell; [time, lay, y, x]; [J K^{-1} kg^{-1}]

    vol_total: NP_REAL = dx * dy * dz # Volume of cell; [m^{3}]

    sw_heating: XR_DATAARRAY = ((sw_flux_abs * vol_total) / (cp_total * mass_total)) * sec_per_day # [time, lay, y, x]; [K d^{-1}]
    sw_heating = (sw_heating
        .assign_attrs({"units" : "K d^{-1}", 
                        "long_name" : "shortwave heating rate",
                        "standard_name" : "shortwave_heating_rate"})
        .rename("shortwave_heating_rate"))

    return sw_heating