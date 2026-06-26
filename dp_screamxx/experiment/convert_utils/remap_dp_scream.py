# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATAARRAY
from consts.physical import cp_d, cp_v, cp_lw, cp_iw, mu_d, g, sec_per_hour
from consts.rte_rrtmgp_cpp_fields import rte_rrtmgp_cpp_gas_keys as gas_keys

from .print_msg import print_msg

def remap_dp_scream(dp_scream_file: str, time_idx: NP_INT, rad_tran_src_grid: dict, 
    sort_mask: NP_ARRAY[NP_INT], l_rank: NP_INT) -> dict:
    #---------------------------------------------------------------------------
    # Calculate relevant DP-SCREAM quantities on the DP-SCREAM grid
    #---------------------------------------------------------------------------
    msg: str = "Calculating relevant quantities from DP-SCREAM..."
    print_msg(msg, l_rank)

    z_int_src: XR_DATAARRAY = calc_z_int(dp_scream_file, time_idx) # [ncol, ilev]; [m]
    p_top: NP_REAL = calc_p_top(dp_scream_file) # [Pa]

    # Masses to remap other quantities
    msg: str = "Calculating air, water vapor, and cloud water masses from DP-SCREAM..."
    print_msg(msg, l_rank)
    mass_moist_air_src: XR_DATAARRAY # [ncol, lev]; [kg]
    mass_dry_air_src: XR_DATAARRAY # [ncol, lev]; [kg]
    mass_wv_src: XR_DATAARRAY # [ncol, lev]; [kg]
    mass_lw_src: XR_DATAARRAY # [ncol, lev]; [kg]
    mass_iw_src: XR_DATAARRAY # [ncol, lev]; [kg]
    [mass_moist_air_src, mass_dry_air_src, mass_wv_src, mass_lw_src, mass_iw_src] = \
        calc_masses(dp_scream_file, time_idx)

    # Molar amounts to remap volume mixing ratios
    msg: str = "Calculating number of dry air and trace gas moles from DP-SCREAM..."
    print_msg(msg, l_rank)
    nmole_dry_air_src: XR_DATAARRAY # [ncol, lev]; [mol]
    nmole_gases_src: dict # [ncol, lev]; [mol]
    [nmole_dry_air_src, nmole_gases_src] = \
        calc_nmole_gases(dp_scream_file, time_idx, mass_dry_air_src) 

    # Quantities that need to be specially remapped
    msg: str = "Calculating temperature from DP-SCREAM..."
    print_msg(msg, l_rank)
    T_src: XR_DATAARRAY = calc_T(dp_scream_file, time_idx) # [ncol, lev]; [K]

    msg: str = "Calculating cloud water effective sizes from DP-SCREAM..."
    print_msg(msg, l_rank)
    rel_src: XR_DATAARRAY # [ncol, lev]; [μm]
    dei_src: XR_DATAARRAY # [ncol, lev]; [μm]
    [rel_src, dei_src] = calc_effective_sizes(dp_scream_file, time_idx) 

    # Two-dimensional quantities
    msg: str = "Calculating two-dimensional quantities from DP-SCREAM..."
    print_msg(msg, l_rank)
    mu0: XR_DATAARRAY # [ncol]; [N/A]
    azi: XR_DATAARRAY # [ncol]; [N/A]
    tsi: XR_DATAARRAY # [ncol]; [W m^{-2}]
    sfc_alb_dir: XR_DATAARRAY # [ncol, swband]; [N/A]
    sfc_alb_dif: XR_DATAARRAY # [ncol, swband]; [N/A]
    [mu0, azi, tsi, sfc_alb_dir, sfc_alb_dif] = calc_2d_quantities(dp_scream_file, time_idx)

    #---------------------------------------------------------------------------
    # Vertically remap relevant DP-SCREAM quantities to the RTE-RRTMGP-CPP source grid
    #---------------------------------------------------------------------------
    msg: str = "Vertically remapping relevant quantities to RTE-RRTMGP-CPP source grid..."
    print_msg(msg, l_rank)

    z_mid_tgt: NP_ARRAY[NP_REAL] = rad_tran_src_grid["z"] # [lev]; [m]
    z_int_tgt: NP_ARRAY[NP_REAL] = rad_tran_src_grid["zh"] # [ilev]; [m]
    
    # Masses to remap other quantities
    mass_moist_air_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_moist_air_src) # [ncol, lev]; [kg]
    mass_dry_air_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_dry_air_src) # [ncol, lev]; [kg]
    mass_wv_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_wv_src) # [ncol, lev]; [kg]
    mass_lw_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_lw_src) # [ncol, lev]; [kg]
    mass_iw_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_iw_src) # [ncol, lev]; [kg]
    
    # Molar amounts to remap volume mixing ratios
    nmole_dry_air_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, nmole_dry_air_src) # [ncol, lev]; [kg]
    nmole_gases_tgt: dict = {}
    for vmr_gas_key in nmole_gases_src.keys():
        nmole_gas_tgt: XR_DATARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
            z_mid_tgt, nmole_gases_src[vmr_gas_key]) # [ncol, lev]; [mole]
        nmole_gases_tgt[vmr_gas_key] = sort_dp_scream(nmole_gas_tgt, sort_mask)

    # Quanities that need to be specially remapped
    T_tgt: XR_DATAARRAY = conservative_vertical_remap_T(z_int_src, z_int_tgt,
        z_mid_tgt, T_src, mass_dry_air_src, mass_wv_src, mass_lw_src, mass_iw_src) # [ncol, lev], [ncol, ilev]; [K]

    rel_tgt: XR_DATAARRAY = conservative_vertical_remap_effective_size(z_int_src, z_int_tgt,
        z_mid_tgt, rel_src, mass_lw_src) # [ncol, lev]; [μm]
    dei_tgt: XR_DATAARRAY = conservative_vertical_remap_effective_size(z_int_src, z_int_tgt,
        z_mid_tgt, dei_src, mass_iw_src) # [ncol, lev]; [μm]

    dp_scream_remap: dict = {
        "p_top" : p_top,
        "mass_moist_air" : sort_dp_scream(mass_moist_air_tgt, sort_mask), # [n_z, n_y, n_x]; [kg]
        "mass_dry_air" : sort_dp_scream(mass_dry_air_tgt, sort_mask), # [n_z, n_y, n_x]; [kg]
        "mass_wv" : sort_dp_scream(mass_wv_tgt, sort_mask), # [n_z, n_y, n_x]; [kg]
        "mass_lw" : sort_dp_scream(mass_lw_tgt, sort_mask), # [n_z, n_y, n_x]; [kg]
        "mass_iw" : sort_dp_scream(mass_iw_tgt, sort_mask), # [n_z, n_y, n_x]; [kg]
        "nmole_dry_air" : sort_dp_scream(nmole_dry_air_tgt, sort_mask), # [n_z, n_y, n_x]; [mol]
        **nmole_gases_tgt, # [n_z, n_y, n_x]; [mol]
        "T" : sort_dp_scream(T_tgt, sort_mask), # [n_z, n_y, n_x]; [K]
        "rel" : sort_dp_scream(rel_tgt, sort_mask), # [n_z, n_y, n_x]; [μm]
        "dei" : sort_dp_scream(dei_tgt, sort_mask), # [n_z, n_y, n_x]; [μm]
        "mu0" : sort_dp_scream(mu0, sort_mask), # [n_y, n_x]; [N/A]
        "azi" : sort_dp_scream(azi, sort_mask), # [n_y, n_x]; [radians]
        "tsi" : sort_dp_scream(tsi, sort_mask), # [n_y, n_x]; [W m^{-2}]
        "sfc_alb_dir" : sort_dp_scream(sfc_alb_dir, sort_mask), # [n_y, n_x]; [N/A]
        "sfc_alb_dif" : sort_dp_scream(sfc_alb_dif, sort_mask), # [n_y, n_x]; [N/A]
        }

    return dp_scream_remap

def calc_z_int(dp_scream_file: str, time_idx: NP_INT) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        z_int: XR_DATAARRAY = xr_dp_scream["z_int"].isel(time = time_idx).load() # Geometric height at level interfaces; [nt, ncol, ilev]; [μm]

    return z_int

def calc_p_top(dp_scream_file: str) -> NP_REAL:
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        p_int: XR_DATAARRAY = xr_dp_scream["p_int"].isel(time = 0, ncol = 0, ilev = 0).load() # Hydrostatic pressure at level interfaces [top -> bot]; [nt, ncol, ilev]; [Pa]
    p_top: NP_REAL = NP_REAL(p_int) # ASSUME - Constant in space and time

    return p_top

def calc_masses(dp_scream_file: str, time_idx: NP_INT) -> list[XR_DATAARRAY]:
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        p_int: XR_DATAARRAY = xr_dp_scream["p_int"].isel(time = time_idx).load() # Hydrostatic pressure at level interfaces; [nt, ncol, ilev]; [Pa]
        qv: XR_DATAARRAY = xr_dp_scream["qv"].isel(time = time_idx).load() # Water vapor moist mixing ratio at level midpoints; [ncol, lev]; [kg kg^{-1}]
        qc: XR_DATAARRAY = xr_dp_scream["qc"].isel(time = time_idx).load() # Cloud liquid water moist mixing ratio at level midpoints; [ncol, lev]; [kg kg^{-1}]
        qi: XR_DATAARRAY = xr_dp_scream["qi"].isel(time = time_idx).load() # Cloud ice water moist mixing ratio at level midpoints; [ncol, lev]; [kg kg^{-1}]
        lev: XR_DATAARRAY = xr_dp_scream["lev"].load() # Hybrid level at level midpoints; [lev]; [N/A]
        ncol: XR_DATAARRAY = xr_dp_scream["ncol"].load() # Horizontal grid; [ncol]
        lon: XR_DATAARRAY = xr_dp_scream["lon"].load() # x-position at column midpoints; [ncol]; [m]
        lat: XR_DATAARRAY = xr_dp_scream["lat"].load() # y-position at column midpoints; [ncol]; [m]
        time: XR_DATAARRAY = xr_dp_scream["time"].load() # Time since simulation start; [ns]
    pdel: XR_DATAARRAY = (p_int.diff("ilev")).rename({"ilev" : "lev"}).assign_coords({"lev" : lev}) # Hydrostatic pressure-thickness; [time, lev, ncol]; [Pa]
    time_data: NP_REAL = NP_REAL(time[time_idx] - time[0]) / (sec_per_hour * 1.0e9) # Time since simulation start; [ns] => [h]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
    #---------------------------------------------------------------------------
    x: NP_ARRAY[NP_REAL] = NP_REAL(np.unique(lon)) # [n_x]; [m]
    dx: NP_REAL = x[1] - x[0] # [m]
    dy: NP_REAL = dx # [m]

    #---------------------------------------------------------------------------
    # Set up common elements across as xarray data arrays
    #---------------------------------------------------------------------------
    dims: tuple[str] = ("ncol", "lev")
    coords: dict = {
        "time" : time_data,
        "lat" : lat.copy(deep = True),
        "lon" : lon.copy(deep = True),
        "ncol" : NP_REAL(ncol.to_numpy()),
        "lev" : lev.copy(deep = True),
    }

    #---------------------------------------------------------------------------
    # Calculate mass moist air
    #---------------------------------------------------------------------------
    mass_moist_air: XR_DATARRAY = (pdel * dx * dy) / g # From hydrostatic pressure definition; [kg]
    mass_moist_air = XR_DATAARRAY(data = NP_REAL(mass_moist_air.to_numpy()),
        dims = dims,
        coords = coords,
        name = "mass_moist_air",
        attrs = {
            "units" : "kg", 
            "long_name" : "midpoint moist air mass",
            "standard_name" : "moist_air_mass",
        })

    #---------------------------------------------------------------------------
    # Calculate mass dry air
    #---------------------------------------------------------------------------
    mass_dry_air: XR_DATARRAY = (1. - qv) * mass_moist_air # Water vapor mass; [ncol, lev]; [kg]
    mass_dry_air = XR_DATAARRAY(data = NP_REAL(mass_dry_air.to_numpy()),
        dims = dims,
        coords = coords,
        name = "mass_dry_air",
        attrs = {
            "units" : "kg", 
            "long_name" : "midpoint dry air mass",
            "standard_name" : "dry_air_mass"
        })

    #---------------------------------------------------------------------------
    # Calculate mass water vapor
    #---------------------------------------------------------------------------
    mass_wv: XR_DATARRAY = qv * mass_moist_air # Water vapor mass; [ncol, lev]; [kg]
    mass_wv = XR_DATAARRAY(data = NP_REAL(mass_wv.to_numpy()),
        dims = dims,
        coords = coords,
        name = "mass_wv",
        attrs = {
            "units" : "kg", 
            "long_name" : "midpoint water vapor mass",
            "standard_name" : "water_vapor_mass"
        })

    #---------------------------------------------------------------------------
    # Calculate mass cloud liquid water
    #---------------------------------------------------------------------------
    mass_lw: XR_DATARRAY = qc * mass_moist_air # Cloud liquid water mass; [ncol, lev]; [kg]
    mass_lw = XR_DATAARRAY(data = NP_REAL(mass_lw.to_numpy()),
        dims = dims,
        coords = coords,
        name = "mass_lw",
        attrs = {
            "units" : "kg", 
            "long_name" : "midpoint cloud liquid water mass",
            "standard_name" : "cloud_liquid_water_mass"
        })

    #---------------------------------------------------------------------------
    # Calculate mass cloud ice water
    #---------------------------------------------------------------------------
    mass_iw: XR_DATARRAY = qi * mass_moist_air # Cloud ice water mass; [ncol, lev]; [kg]
    mass_iw = XR_DATAARRAY(data = NP_REAL(mass_iw.to_numpy()),
        dims = dims,
        coords = coords,
        name = "mass_iw",
        attrs = {
            "units" : "kg", 
            "long_name" : "midpoint cloud ice water mass",
            "standard_name" : "cloud_ice_water_mass"
        })

    return [mass_moist_air, mass_dry_air, mass_wv, mass_lw, mass_iw]

def calc_nmole_gases(dp_scream_file: str, time_idx: NP_INT, 
    mass_dry_air: XR_DATAARRAY) -> list[XR_DATAARRAY, dict]:
    #---------------------------------------------------------------------------
    # Calculate number of dry air moles
    #---------------------------------------------------------------------------
    nmole_dry_air: XR_DATARRAY = mass_dry_air / mu_d # Number of dry air moles; [ncol, lev]; [mol]
    nmole_dry_air = (nmole_dry_air
        .assign_attrs({"units" : "mol", 
                       "long_name" : "midpoint dry air number of moles",
                       "standard_name" : "dry_air_moles"})
        .rename("nmole_dry_air"))
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file common across all trace gases
    #---------------------------------------------------------------------------
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        lev: XR_DATAARRAY = xr_dp_scream["lev"].load() # Hybrid level at level midpoints; [lev]; [N/A]
        ncol: XR_DATAARRAY = xr_dp_scream["ncol"].load() # Horizontal grid; [ncol]
        lon: XR_DATAARRAY = xr_dp_scream["lon"].load() # x-position at column midpoints; [ncol]; [m]
        lat: XR_DATAARRAY = xr_dp_scream["lat"].load() # y-position at column midpoints; [ncol]; [m]
        time: XR_DATAARRAY = xr_dp_scream["time"].load() # Time since simulation start; [ns]
    time_data: NP_REAL = NP_REAL(time[time_idx] - time[0]) / (sec_per_hour * 1.0e9) # Time since simulation start; [ns] => [h]

    #---------------------------------------------------------------------------
    # Set up common elements across as xarray data arrays
    #---------------------------------------------------------------------------
    dims: tuple[str] = ("ncol", "lev")
    coords: dict = {
        "time" : time_data,
        "lat" : lat.copy(deep = True),
        "lon" : lon.copy(deep = True),
        "ncol" : NP_REAL(ncol.to_numpy()),
        "lev" : lev.copy(deep = True),
    }

    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    nmole_gases: dict = {}
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        for gas_key in gas_keys:
            dp_scream_key: str = gas_key + "_volume_mix_ratio"
            if dp_scream_key in xr_dp_scream.keys():
                vmr_gas: XR_DATAARRAY = xr_dp_scream[dp_scream_key].isel(time = time_idx).load() # Gas dry volume mixing ratio at level midpoints; [nt, ncol, lev]; [mol mol^{-1}]
                nmole_gas: XR_DATAARRAY = vmr_gas * nmole_dry_air # Moles of gas at midpoints; [ncol, lev]; [mol]
                nmole_gas = XR_DATAARRAY(data = NP_REAL(nmole_gas.to_numpy()),
                    dims = dims,
                    coords = coords,
                    name = "nmole_" + gas_key,
                    attrs = {
                        "units" : "mole", 
                        "long_name" : "midpoint " + gas_key + " number of moles",
                        "standard_name" : gas_key + "_moles"
                    })
                
                nmole_gases["nmole_" + gas_key] = nmole_gas

    return [nmole_dry_air, nmole_gases]

def calc_effective_sizes(dp_scream_file: str, time_idx: NP_INT) -> list[XR_DATAARRAY]:
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        rel: XR_DATAARRAY = xr_dp_scream["eff_radius_qc"].isel(time = time_idx).load() # Cloud liquid water effective radius at level midpoints; [nt, ncol, lev]; [μm]
        rei: XR_DATAARRAY = xr_dp_scream["eff_radius_qi"].isel(time = time_idx).load() # Cloud ice water effective radius at level midpoints; [nt, ncol, lev]; [μm]
        lon: XR_DATAARRAY = xr_dp_scream["lon"].load() # x-position at column midpoints; [ncol]; [m]
        lat: XR_DATAARRAY = xr_dp_scream["lat"].load() # y-position at column midpoints; [ncol]; [m]
        time: XR_DATAARRAY = xr_dp_scream["time"].load() # Time since simulation start; [ns]
    time_data: NP_REAL = NP_REAL(time[time_idx] - time[0]) / (sec_per_hour * 1.0e9) # Time since simulation start; [ns] => [h]
    
    #---------------------------------------------------------------------------
    # Set up common elements across xarray data arrays
    #---------------------------------------------------------------------------
    coords: dict = {
        "time" : time_data,
        "lat" : lat.copy(deep = True),
        "lon" : lon.copy(deep = True),
    }

    #---------------------------------------------------------------------------
    # Set up xarray data arrays for cloud liquid water effective radius and
    # cloud ice water effective diameter
    #---------------------------------------------------------------------------
    rel = rel.assign_coords(coords)

    dei: XR_DATAARRAY = 2. * rei
    dei = dei.assign_coords(coords)

    return [rel, dei]

def calc_T(dp_scream_file: str, time_idx: NP_INT) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        T: XR_DATAARRAY = xr_dp_scream["T_mid"].isel(time = time_idx).load() # Temperature at level midpoints; [nt, ncol, lev]; [K]
        lon: XR_DATAARRAY = xr_dp_scream["lon"].load() # x-position at column midpoints; [ncol]; [m]
        lat: XR_DATAARRAY = xr_dp_scream["lat"].load() # y-position at column midpoints; [ncol]; [m]
        time: XR_DATAARRAY = xr_dp_scream["time"].load() # Time since simulation start; [ns]
    time_data: NP_REAL = NP_REAL(time[time_idx] - time[0]) / (sec_per_hour * 1.0e9) # Time since simulation start; [ns] => [h]

    #---------------------------------------------------------------------------
    # Set up common elements across xarray data arrays
    #---------------------------------------------------------------------------
    coords: dict = {
        "time" : time_data,
        "lat" : lat.copy(deep = True),
        "lon" : lon.copy(deep = True),
    }

    #---------------------------------------------------------------------------
    # Set up xarray data array for temperature
    #---------------------------------------------------------------------------
    T = T.assign_coords(coords)

    return T

def calc_2d_quantities(dp_scream_file: str, time_idx: NP_INT) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract relevant fields from DP-SCREAM file
    #---------------------------------------------------------------------------
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        mu0: XR_DATAARRAY = xr_dp_scream["cosine_solar_zenith_angle"].isel(time = time_idx).load() # Cosine solar zenith angle; [ncol]; [N/A]
        swband: XR_DATAARRAY = xr_dp_scream["swband"].load() # Shortwave bands; [swband]
        ncol: XR_DATAARRAY = xr_dp_scream["ncol"].load() # Horizontal grid; [ncol]
        lon: XR_DATAARRAY = xr_dp_scream["lon"].load() # x-position at column midpoints; [ncol]; [m]
        lat: XR_DATAARRAY = xr_dp_scream["lat"].load() # y-position at column midpoints; [ncol]; [m]
        time: XR_DATAARRAY = xr_dp_scream["time"].load() # Time since simulation start; [ns]
    time_data: NP_REAL = NP_REAL(time[time_idx] - time[0]) / (sec_per_hour * 1.0e9) # Time since simulation start; [ns] => [h]
    
    #---------------------------------------------------------------------------
    # Set up common elements across xarray data arrays
    #---------------------------------------------------------------------------
    coords: dict = {
        "time" : time_data,
        "lat" : lat.copy(deep = True),
        "lon" : lon.copy(deep = True),
        "ncol" : NP_REAL(ncol.to_numpy()),
    }

    albedo_coords: dict = {
        "time" : time_data,
        "lat" : lat.copy(deep = True),
        "lon" : lon.copy(deep = True),
        "ncol" : NP_REAL(ncol.to_numpy()),
        "band_sw" : NP_REAL(swband.to_numpy()),
    }

    #---------------------------------------------------------------------------
    # Set up xarray data arrays to output cosine solar zenith angle, solar
    # azimuthal angle, total solar irradiance, and direct and diffuse surface
    # albedo
    #---------------------------------------------------------------------------
    mu0 = mu0.assign_coords(coords)

    standard_azi: NP_REAL = NP_REAL(0.)
    azi_data: NP_ARRAY[NP_REAL] = standard_azi + np.zeros_like(ncol, dtype = NP_REAL) # [ncol]; [radians]
    azi: XR_DATAARRAY = XR_DATAARRAY(data = azi_data,
        dims = ("ncol"),
        coords = coords,
        name = "azi",
        attrs = {
            "units": "radians",
            "long_name": "solar_azimuthal_angle",
            "standard_name": "solar_azimuthal_angle",
        })

    standard_tsi: NP_REAL = NP_REAL(1361.) # Globally- and annually-averaged
    tsi_data: NP_ARRAY[NP_REAL] = standard_tsi + np.zeros_like(ncol, dtype = NP_REAL) # [ncol]; [W m^{-2}]
    tsi: XR_DATAARRAY = XR_DATAARRAY(data = tsi_data,
        dims = ("ncol"),
        coords = coords,
        name = "tsi",
        attrs = {
            "units": "W m^{-2}",
            "long_name": "total_solar_irradiance",
            "standard_name": "total_solar_irradiance",
        })

    nswband: NP_INT = NP_INT(swband.size)
    nncol: NP_INT = NP_INT(ncol.size)

    standard_sfc_alb_dir: NP_REAL = NP_REAL(0.07) # Approximately close to some standard
    standard_sfc_alb_dif: NP_REAL = NP_REAL(0.07) # Approximately close to some standard

    sfc_alb_dir_data: NP_ARRAY[NP_REAL] = standard_sfc_alb_dir + np.zeros((nncol, nswband), dtype = NP_REAL) # [ncol, swband]; [N/A]
    sfc_alb_dif_data: NP_ARRAY[NP_REAL] = standard_sfc_alb_dif + np.zeros((nncol, nswband), dtype = NP_REAL) # [ncol, swband]; [N/A]

    sfc_alb_dir: XR_DATAARRAY = XR_DATAARRAY(data = sfc_alb_dir_data,
        dims = ("ncol", "band_sw"),
        coords = albedo_coords,
        name = "sfc_alb_dir",
        attrs = {
            "units": "N/A",
            "long_name": "surface albedo - direct",
            "standard_name": "surface albedo - direct",
        })

    sfc_alb_dif: XR_DATAARRAY = XR_DATAARRAY(data = sfc_alb_dif_data,
        dims = ("ncol", "band_sw"),
        coords = albedo_coords,
        name = "sfc_alb_dif",
        attrs = {
            "units": "N/A",
            "long_name": "surface albedo - diffuse",
            "standard_name": "surface albedo - diffuse",
        })

    return [mu0, azi, tsi, sfc_alb_dir, sfc_alb_dif]

def conservative_vertical_remap(z_int_src: XR_DATAARRAY, z_int_tgt: NP_ARRAY[NP_REAL],
    z_mid_tgt: NP_ARRAY[NP_REAL], field_src: XR_DATAARRAY) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    ncol: NP_INT = NP_INT(field_src["ncol"].size)
    nlev: NP_INT = NP_INT(field_src["lev"].size)
    nilev: NP_INT = NP_INT(z_int_src["ilev"].size)

    dz_tgt_signed: NP_ARRAY[NP_REAL] = np.diff(z_int_tgt)

    #---------------------------------------------------------------------------
    # Precompute target layer bounds
    #---------------------------------------------------------------------------
    z_tgt_lo: NP_ARRAY[NP_REAL] = np.minimum(z_int_tgt[:-1], z_int_tgt[1:])
    z_tgt_hi: NP_ARRAY[NP_REAL] = np.maximum(z_int_tgt[:-1], z_int_tgt[1:])

    #---------------------------------------------------------------------------
    # Allocate output
    #---------------------------------------------------------------------------
    field_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((ncol, nlev), dtype = NP_REAL)

    z_src_data: NP_ARRAY[NP_REAL] = z_int_src.to_numpy().astype(NP_REAL)
    field_src_data: NP_ARRAY[NP_REAL] = field_src.to_numpy().astype(NP_REAL)

    #---------------------------------------------------------------------------
    # Conservative remap by geometric overlap
    #---------------------------------------------------------------------------
    for icol in range(0, ncol):
        z_src_col: NP_ARRAY[NP_REAL] = z_src_data[icol, :]
        field_src_col: NP_ARRAY[NP_REAL] = field_src_data[icol, :]

        dz_src_signed: NP_ARRAY[NP_REAL] = np.diff(z_src_col)

        dz_src: NP_ARRAY[NP_REAL] = np.abs(dz_src_signed)

        z_src_lo: NP_ARRAY[NP_REAL] = np.minimum(z_src_col[:-1], z_src_col[1:])
        z_src_hi: NP_ARRAY[NP_REAL] = np.maximum(z_src_col[:-1], z_src_col[1:])

        for ilev_tgt in range(0, nlev):
            overlap: NP_ARRAY[NP_REAL] = (np.minimum(z_tgt_hi[ilev_tgt], z_src_hi)
                - np.maximum(z_tgt_lo[ilev_tgt], z_src_lo))

            overlap: NP_REAL = np.maximum(overlap, 0.)

            has_overlap: NP_ARRAY[NP_BOOL] = overlap > 0.

            if np.any(has_overlap):
                field_tgt_data[icol, ilev_tgt] = np.sum(
                    field_src_col[has_overlap] * overlap[has_overlap] / dz_src[has_overlap]
                )

    #---------------------------------------------------------------------------
    # Construct output xarray DataArray
    #---------------------------------------------------------------------------
    field_tgt: XR_DATAARRAY = field_src.copy()
    field_tgt.data = field_tgt_data
    field_tgt = field_tgt.rename({"lev" : "z"}).assign_coords({"z" : z_mid_tgt})

    return field_tgt

def conservative_vertical_remap_effective_size(z_int_src: XR_DATAARRAY, z_int_tgt: NP_ARRAY[NP_REAL],
    z_mid_tgt: NP_ARRAY[NP_REAL], effective_size_src: XR_DATAARRAY, mass_src: XR_DATAARRAY) -> XR_DATAARRAY:
    # For use with rel and dei
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    ncol: NP_INT = NP_INT(effective_size_src["ncol"].size)
    nlev: NP_INT = NP_INT(effective_size_src["lev"].size)

    #---------------------------------------------------------------------------
    # Remap mass-weighted effective size and mass
    #---------------------------------------------------------------------------
    effective_size_mass_src: XR_DATARRAY = effective_size_src * mass_src

    effective_size_mass_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, effective_size_mass_src)
    mass_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_src)

    #---------------------------------------------------------------------------
    # Unweight by mass and construct proper xarray data array
    #---------------------------------------------------------------------------
    nonzero_mask: NP_ARRAY[NP_BOOL] = mass_tgt > 0.
    effective_size_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((ncol, nlev), dtype = NP_REAL)
    effective_size_tgt_data[nonzero_mask] = \
        effective_size_mass_tgt.to_numpy().astype(NP_REAL)[nonzero_mask] \
        / mass_tgt.to_numpy().astype(NP_REAL)[nonzero_mask]
    
    effective_size_tgt: XR_DATAARRAY = effective_size_src.copy()
    effective_size_tgt.data = effective_size_tgt_data
    effective_size_tgt = effective_size_tgt.rename({"lev" : "z"}).assign_coords({"z" : z_mid_tgt})

    return effective_size_tgt

def conservative_vertical_remap_T(z_int_src: XR_DATAARRAY, z_int_tgt: NP_ARRAY[NP_REAL],
    z_mid_tgt: NP_ARRAY[NP_REAL], T_src: XR_DATAARRAY, mass_dry_air_src: XR_DATAARRAY,
    mass_wv_src: XR_DATAARRAY, mass_lw_src: XR_DATAARRAY, mass_iw_src: XR_DATAARRAY) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    ncol: NP_INT = NP_INT(T_src["ncol"].size)
    nlev: NP_INT = NP_INT(T_src["lev"].size)

    #---------------------------------------------------------------------------
    # Remap mass-specific-heat-weighted temperature and mass-specific-heat
    #---------------------------------------------------------------------------
    mass_specific_heat_src: XR_DATARRAY = (cp_d * mass_dry_air_src
        + cp_v * mass_wv_src + cp_lw * mass_lw_src + cp_iw * mass_iw_src)
    mass_specific_heat_T_src: XR_DATARRAY = mass_specific_heat_src * T_src

    mass_specific_heat_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_specific_heat_src)
    mass_specific_heat_T_tgt: XR_DATAARRAY = conservative_vertical_remap(z_int_src, z_int_tgt,
        z_mid_tgt, mass_specific_heat_T_src)

    #---------------------------------------------------------------------------
    # Unweight by mass and construct proper xarray data array
    #---------------------------------------------------------------------------
    nonzero_mask: NP_ARRAY[NP_BOOL] = mass_specific_heat_tgt > 0.
    T_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((ncol, nlev), dtype = NP_REAL)
    T_tgt_data[nonzero_mask] = \
        mass_specific_heat_T_tgt.to_numpy().astype(NP_REAL)[nonzero_mask] \
        / mass_specific_heat_tgt.to_numpy().astype(NP_REAL)[nonzero_mask]
    
    T_tgt: XR_DATAARRAY = T_src.copy()
    T_tgt.data = T_tgt_data
    T_tgt = T_tgt.rename({"lev" : "z"}).assign_coords({"z" : z_mid_tgt})
    
    return T_tgt

def sort_dp_scream(field: XR_DATAARRAY, sort_mask: NP_ARRAY[NP_INT]) -> XR_DATARRAY:
    if "band_sw" in field.dims:
        return (field.isel(ncol = sort_mask)
                .rename({"lon" : "x", "lat" : "y"})
                .set_index(ncol = ["y", "x"])
                .unstack("ncol")
                .transpose(..., "y", "x", "band_sw")
        )
    else:
        return (field.isel(ncol = sort_mask)
                .rename({"lon" : "x", "lat" : "y"})
                .set_index(ncol = ["y", "x"])
                .unstack("ncol")
                .transpose(..., "y", "x")
        )