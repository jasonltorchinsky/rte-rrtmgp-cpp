# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports
import numpy as np
import re
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, MPI_COMM, XR_DATAARRAY
from consts.numeric import MPI_ROOT
from consts.physical import cp_d, cp_v, cp_lw, cp_iw, mu_d, g
from consts.rte_rrtmgp_cpp_fields import rte_rrtmgp_cpp_gas_keys as gas_keys

from .print_msg import print_msg

def coarsen_dp_scream(dp_scream_remap: dict, rad_tran_src_grid: dict, 
    rad_tran_tgt_grids: dict, comm: MPI_COMM) -> dict:
    #---------------------------------------------------------------------------
    # Obtain MPI information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Calculate relevant DP-SCREAM quantities on the DP-SCREAM grid
    #---------------------------------------------------------------------------
    msg: str = "Looping through RTE-RRTMGP-CPP target grids..."
    print_msg(msg, l_rank)

    dp_scream_coarsen: dict = {}

    coarse_factor_str: str
    for coarse_factor_str in rad_tran_tgt_grids:
        msg: str = "Coarsening relevant DP-SCREAM fields to {}...".format(coarse_factor_str)
        print_msg(msg, l_rank)

        rad_tran_tgt_grid: dict = rad_tran_tgt_grids[coarse_factor_str]

        # Masses to remap other quantities
        mass_moist_air_tgt: XR_DATAARRAY = conservative_horizontal_remap(
            rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["mass_moist_air"])
        mass_dry_air_tgt: XR_DATAARRAY = conservative_horizontal_remap(
            rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["mass_dry_air"])
        mass_wv_tgt: XR_DATAARRAY = conservative_horizontal_remap(
            rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["mass_wv"])
        mass_lw_tgt: XR_DATAARRAY = conservative_horizontal_remap(
            rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["mass_lw"])
        mass_iw_tgt: XR_DATAARRAY = conservative_horizontal_remap(
            rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["mass_iw"])

        # Molar amounts to remap volume mixing ratios
        nmole_dry_air_tgt: XR_DATAARRAY = conservative_horizontal_remap(
            rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["nmole_dry_air"])
        nmole_gases_tgt: dict = {}
        for gas_key in gas_keys:
            nmole_gas_key: str = "nmole_" + gas_key
            if nmole_gas_key in dp_scream_remap.keys():
                nmole_gases_tgt[nmole_gas_key] = conservative_horizontal_remap(
                    rad_tran_src_grid, rad_tran_tgt_grid,
                    dp_scream_remap[nmole_gas_key])

        # Quanities that need to be specially remapped
        T_lay_tgt: XR_DATAARRAY
        T_lev_tgt: XR_DATAARRAY 
        [T_lay_tgt, T_lev_tgt] = conservative_horizontal_remap_T(rad_tran_src_grid, rad_tran_tgt_grid,
            dp_scream_remap["T"], dp_scream_remap["mass_dry_air"], dp_scream_remap["mass_wv"],
            dp_scream_remap["mass_lw"], dp_scream_remap["mass_iw"])
        
        rel_tgt: XR_DATAARRAY = conservative_horizontal_remap_effective_size(
            rad_tran_src_grid, rad_tran_tgt_grid, dp_scream_remap["rel"],
            dp_scream_remap["mass_lw"])
        dei_tgt: XR_DATAARRAY = conservative_horizontal_remap_effective_size(
            rad_tran_src_grid, rad_tran_tgt_grid, dp_scream_remap["dei"],
            dp_scream_remap["mass_iw"])

        # Quantities need to be derived
        p_lay_tgt: XR_DATAARRAY 
        p_lev_tgt: XR_DATAARRAY
        [p_lay_tgt, p_lev_tgt] = calc_p_tgt(rad_tran_tgt_grid, mass_moist_air_tgt,
            dp_scream_remap["p_top"])

        vmr_gases_tgt: dict = {}
        for gas_key in gas_keys:
            vmr_gas_key: str = "vmr_" + gas_key
            nmole_gas_key: str = "nmole_" + gas_key
            if nmole_gas_key in nmole_gases_tgt.keys():
                vmr_gases_tgt[vmr_gas_key] = calc_vmr_gas_tgt(nmole_gases_tgt[nmole_gas_key], nmole_dry_air_tgt)

        lwp_tgt: XR_DATAARRAY = calc_wp(mass_lw_tgt)
        iwp_tgt: XR_DATAARRAY = calc_wp(mass_iw_tgt)

        # Two-dimensional quantities
        mu0_tgt: XR_DATAARRAY = horizontal_interpolate(rad_tran_tgt_grid, dp_scream_remap["mu0"])
        azi_tgt: XR_DATAARRAY = horizontal_interpolate(rad_tran_tgt_grid, dp_scream_remap["azi"])
        tsi_tgt: XR_DATAARRAY = horizontal_interpolate(rad_tran_tgt_grid, dp_scream_remap["tsi"])
        sfc_alb_dir_tgt: XR_DATAARRAY = horizontal_interpolate(rad_tran_tgt_grid, dp_scream_remap["sfc_alb_dir"])
        sfc_alb_dif_tgt: XR_DATAARRAY = horizontal_interpolate(rad_tran_tgt_grid, dp_scream_remap["sfc_alb_dif"])

        # Null grid dimensions
        n_x_tgt: NP_INT = NP_INT(rad_tran_tgt_grid["x"].size)
        n_y_tgt: NP_INT = NP_INT(rad_tran_tgt_grid["y"].size)
        n_z_tgt: NP_INT = NP_INT(rad_tran_tgt_grid["z"].size)

        ngrid_x: XR_DATAARRAY = XR_DATAARRAY(data = max(NP_INT(1), n_x_tgt // 10),
            name = "ngrid_x",
            attrs = {
                "long_name" : "number_acceleration_grid_points_x",
                "standard_name" : "null_points_x",
            })
        ngrid_y: XR_DATAARRAY = XR_DATAARRAY(data = max(NP_INT(1), n_y_tgt // 10),
            name = "ngrid_y",
            attrs = {
                "long_name" : "number_acceleration_grid_points_y",
                "standard_name" : "null_points_y",
            })
        ngrid_z: XR_DATAARRAY = XR_DATAARRAY(data = max(NP_INT(1), n_z_tgt // 10),
            name = "ngrid_z",
            attrs = {
                "long_name" : "number_acceleration_grid_points_z",
                "standard_name" : "null_points_z",
            })

        #-----------------------------------------------------------------------
        # Store in a dict
        #-----------------------------------------------------------------------
        dp_scream_coarsen[coarse_factor_str] = {
            "ngrid_x" : ngrid_x,
            "ngrid_y" : ngrid_y,
            "ngrid_z" : ngrid_z,
            "p_lay" : p_lay_tgt,
            "p_lev" : p_lev_tgt,
            "t_lay" : T_lay_tgt,
            "t_lev" : T_lev_tgt,
            "lwp" : lwp_tgt,
            "rel" : rel_tgt,
            "iwp" : iwp_tgt,
            "dei" : dei_tgt,
            **vmr_gases_tgt,
            "mu0" : mu0_tgt, # [n_y, n_x]; [N/A]
            "azi" : azi_tgt, # [n_y, n_x]; [radians]
            "tsi" : tsi_tgt, # [n_y, n_x]; [W m^{-2}]
            "sfc_alb_dir" : sfc_alb_dir_tgt, # [n_y, n_x]; [N/A]
            "sfc_alb_dif" : sfc_alb_dif_tgt, # [n_y, n_x]; [N/A]
            }

    return dp_scream_coarsen

def _calc_overlap_fraction_from_interfaces(coord_int_src: NP_ARRAY[NP_REAL],
    coord_int_tgt: NP_ARRAY[NP_REAL]) -> NP_ARRAY[NP_REAL]:
    d_src: NP_ARRAY[NP_REAL] = np.diff(coord_int_src)
    d_tgt: NP_ARRAY[NP_REAL] = np.diff(coord_int_tgt)

    src_lo: NP_ARRAY[NP_REAL] = np.minimum(coord_int_src[:-1], coord_int_src[1:])
    src_hi: NP_ARRAY[NP_REAL] = np.maximum(coord_int_src[:-1], coord_int_src[1:])
    tgt_lo: NP_ARRAY[NP_REAL] = np.minimum(coord_int_tgt[:-1], coord_int_tgt[1:])
    tgt_hi: NP_ARRAY[NP_REAL] = np.maximum(coord_int_tgt[:-1], coord_int_tgt[1:])

    src_width: NP_ARRAY[NP_REAL] = src_hi - src_lo


    overlap: NP_ARRAY[NP_REAL] = np.maximum(
        0., np.minimum(tgt_hi[:, np.newaxis], src_hi[np.newaxis, :]) - np.maximum(tgt_lo[:, np.newaxis], src_lo[np.newaxis, :]),
    )

    weights: NP_ARRAY[NP_REAL] = overlap / src_width[np.newaxis, :]

    return weights

def conservative_horizontal_remap(grid_src: dict, grid_tgt: dict, field_src: XR_DATAARRAY) -> xr.DataArray:
    #---------------------------------------------------------------------------
    # Get grid information
    #---------------------------------------------------------------------------
    n_x_src: NP_INT = NP_INT(grid_src["x"].size)
    n_y_src: NP_INT = NP_INT(grid_src["y"].size)
    xh_src: NP_ARRAY[NP_REAL] = grid_src["xh"]
    yh_src: NP_ARRAY[NP_REAL] = grid_src["yh"]

    x_tgt: NP_ARRAY[NP_REAL] = grid_tgt["x"]
    xh_tgt: NP_ARRAY[NP_REAL] = grid_tgt["xh"]
    y_tgt: NP_ARRAY[NP_REAL] = grid_tgt["y"]
    yh_tgt: NP_ARRAY[NP_REAL] = grid_tgt["yh"]


    w_x: NP_ARRAY[NP_REAL] = _calc_overlap_fraction_from_interfaces(xh_src, xh_tgt) # [n_x_tgt, n_x_src]
    w_y: NP_ARRAY[NP_REAL] = _calc_overlap_fraction_from_interfaces(yh_src, yh_tgt) # [n_y_tgt, n_y_src]

    field_src_data: NP_ARRAY[NP_REAL] = field_src.to_numpy().astype(NP_REAL)

    # First remap y, then x:
    tmp: NP_ARRAY[NP_REAL] = np.einsum("...yx,jy->...jx", field_src_data, w_y)

    field_tgt_data: NP_ARRAY[NP_REAL] = np.einsum("...jx,ix->...ji", tmp, w_x)

    #---------------------------------------------------------------------------
    # Construct target xarray data array
    #---------------------------------------------------------------------------
    field_tgt: XR_DATAARRAY = XR_DATAARRAY(data = field_tgt_data,
        dims = field_src.dims,
        coords = {
            "time" : field_src.coords["time"].copy(deep = True),
            field_src.dims[0] : grid_tgt[field_src.dims[0]],
            "y" : grid_tgt["y"],
            "x" : grid_tgt["x"],
        },
        name = field_src.name,
        attrs = field_src.attrs)

    return field_tgt

def horizontal_interpolate(grid_tgt: dict, field_src: XR_DATAARRAY,
    method: str = "linear") -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Get grid information
    #---------------------------------------------------------------------------
    x_tgt: NP_ARRAY[NP_REAL] = grid_tgt["x"] # [n_x_tgt]; [m]
    y_tgt: NP_ARRAY[NP_REAL] = grid_tgt["y"] # [n_y_tgt]; [m]

    #---------------------------------------------------------------------------
    # Interpolate field horizontally
    #---------------------------------------------------------------------------
    field_tgt: XR_DATAARRAY = field_src.interp(
        {
            "x": x_tgt,
            "y": y_tgt,
        },
        method = method,
        assume_sorted = False,
    )

    field_tgt = field_tgt.assign_attrs(field_src.attrs.copy())
    field_tgt = field_tgt.rename(field_src.name)

    field_tgt["x"].attrs["units"] = "m"
    field_tgt["y"].attrs["units"] = "m"

    return field_tgt

def conservative_horizontal_remap_effective_size(grid_src: dict, grid_tgt: dict, 
    effective_size_src: XR_DATAARRAY, mass_src: XR_DATAARRAY) -> xr.DataArray:
    # For use with rel and dei
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    n_z_tgt: NP_INT = NP_INT(grid_tgt["z"].size)
    n_y_tgt: NP_INT = NP_INT(grid_tgt["y"].size)
    n_x_tgt: NP_INT = NP_INT(grid_tgt["x"].size)

    #---------------------------------------------------------------------------
    # Remap mass-weighted effective size and mass
    #---------------------------------------------------------------------------
    effective_size_mass_src: XR_DATARRAY = effective_size_src * mass_src

    effective_size_mass_tgt: XR_DATAARRAY = conservative_horizontal_remap(grid_src, grid_tgt,
        effective_size_mass_src)
    mass_tgt: XR_DATAARRAY = conservative_horizontal_remap(grid_src, grid_tgt,
        mass_src)

    #---------------------------------------------------------------------------
    # Unweight by mass and construct proper xarray data array
    #---------------------------------------------------------------------------
    nonzero_mask: NP_ARRAY[NP_BOOL] = mass_tgt > 0.
    effective_size_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((n_z_tgt, n_y_tgt, n_x_tgt), dtype = NP_REAL)
    effective_size_tgt_data[nonzero_mask] = (
        effective_size_mass_tgt.to_numpy().astype(NP_REAL)[nonzero_mask]
        / mass_tgt.to_numpy().astype(NP_REAL)[nonzero_mask])

    #---------------------------------------------------------------------------
    # Limit to acceptable values
    #---------------------------------------------------------------------------
    if effective_size_src.name == "eff_radius_qc":
        rel_min: NP_REAL = NP_REAL(2.5) # Minimum valid cloud liquid water effective radius; [μm]
        rel_max: NP_REAL = NP_REAL(21.5) # Maximum valid cloud liquid water effective radius; [μm]
        
        min_mask: NP_ARRAY[NP_BOOL] = np.logical_and((effective_size_tgt_data > 0), (effective_size_tgt_data < rel_min))
        max_mask: NP_ARRAY[NP_BOOL] = (effective_size_tgt_data > rel_max)
        
        effective_size_tgt_data[min_mask] = rel_min
        effective_size_tgt_data[max_mask] = rel_max
    elif effective_size_src.name == "eff_radius_qi":
        dei_min: NP_REAL = NP_REAL(10.0) # Minimum valid cloud ice water effective diameter; [μm]
        dei_max: NP_REAL = NP_REAL(180.0) # Maximum valid cloud ice water effective diameter; [μm]
        
        min_mask: NP_ARRAY[NP_BOOL] = np.logical_and((effective_size_tgt_data > 0), (effective_size_tgt_data < dei_min))
        max_mask: NP_ARRAY[NP_BOOL] = (effective_size_tgt_data > dei_max)
        
        effective_size_tgt_data[min_mask] = dei_min
        effective_size_tgt_data[max_mask] = dei_max
    
    #---------------------------------------------------------------------------
    # Construct target xarray data array at RTE-RRTMGP-CPP layers
    #---------------------------------------------------------------------------
    effective_size_tgt: XR_DATAARRAY = XR_DATAARRAY(data = effective_size_tgt_data,
        dims = effective_size_src.dims,
        coords = {
            "time" : effective_size_src.coords["time"].copy(deep = True),
            "z" : grid_tgt["z"],
            "y" : grid_tgt["y"],
            "x" : grid_tgt["x"],
        },
        name = effective_size_src.name,
        attrs = effective_size_src.attrs)

    return effective_size_tgt

def conservative_horizontal_remap_T(grid_src: dict, grid_tgt: dict,
    T_src: XR_DATAARRAY, mass_dry_air_src: XR_DATAARRAY, mass_wv_src: XR_DATAARRAY,
    mass_lw_src: XR_DATAARRAY, mass_iw_src: XR_DATAARRAY) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    n_z_tgt: NP_INT = NP_INT(grid_tgt["z"].size)
    n_y_tgt: NP_INT = NP_INT(grid_tgt["y"].size)
    n_x_tgt: NP_INT = NP_INT(grid_tgt["x"].size)

    #---------------------------------------------------------------------------
    # Remap mass-specific-heat-weighted temperature and mass-specific-heat
    #---------------------------------------------------------------------------
    mass_specific_heat_src: XR_DATARRAY = (cp_d * mass_dry_air_src
        + cp_v * mass_wv_src + cp_lw * mass_lw_src + cp_iw * mass_iw_src)
    mass_specific_heat_T_src: XR_DATARRAY = mass_specific_heat_src * T_src

    mass_specific_heat_tgt: XR_DATAARRAY = conservative_horizontal_remap(grid_src, grid_tgt,
        mass_specific_heat_src)
    mass_specific_heat_T_tgt: XR_DATAARRAY = conservative_horizontal_remap(grid_src, grid_tgt,
        mass_specific_heat_T_src)

    #---------------------------------------------------------------------------
    # Unweight by mass and construct proper xarray data array
    #---------------------------------------------------------------------------
    nonzero_mask: NP_ARRAY[NP_BOOL] = mass_specific_heat_tgt > 0.
    T_lay_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((n_z_tgt, n_y_tgt, n_x_tgt), dtype = NP_REAL)
    T_lay_tgt_data[nonzero_mask] = (
        mass_specific_heat_T_tgt.to_numpy().astype(NP_REAL)[nonzero_mask]
        / mass_specific_heat_tgt.to_numpy().astype(NP_REAL)[nonzero_mask])
    
    #---------------------------------------------------------------------------
    # Construct target xarray data array at RTE-RRTMGP-CPP layers
    #---------------------------------------------------------------------------
    T_lay_tgt: XR_DATAARRAY = XR_DATAARRAY(data = T_lay_tgt_data,
        dims = ("z", "y", "x"),
        coords = {
            "time" : T_src.coords["time"].copy(deep = True),
            "z" : grid_tgt["z"],
            "y" : grid_tgt["y"],
            "x" : grid_tgt["x"],
        },
        name = "T_lay",
        attrs = {
            "units": "K",
            "long_name": "temperature_at_layers",
            "standard_name": "temperature_at_layers",
            "cell_methods": T_src.attrs.get("cell_methods", "time: point"),
        })

    #---------------------------------------------------------------------------
    # Interpolate RTE-RRTMGP-CPP layer values to RTE-RRTMGP-CPP levels
    #---------------------------------------------------------------------------
    xr_z_int_tgt: XR_DATARRAY = XR_DATAARRAY(grid_tgt["zh"], dims = ("zh"), name = "zh")
    T_lev_tgt_data: NP_ARRAY[NP_REAL] = T_lay_tgt.interp(z = xr_z_int_tgt, kwargs = {"fill_value" : "extrapolate"}).to_numpy().astype(NP_REAL)
    
    T_lev_tgt: XR_DATAARRAY = XR_DATAARRAY(data = T_lev_tgt_data,
        dims = ("zh", "y", "x"),
        coords = {
            "time" : T_src.coords["time"].copy(deep = True),
            "zh" : grid_tgt["zh"],
            "y" : grid_tgt["y"],
            "x" : grid_tgt["x"],
        },
        name = "T_lev",
        attrs = {
            "units": "K",
            "long_name": "temperature_at_levels",
            "standard_name": "temperature_at_levels",
            "cell_methods": T_src.attrs.get("cell_methods", "time: point"),
        })

    return [T_lay_tgt, T_lev_tgt] # Refers to RTE-RRTMGP-CPP layers and levels

def calc_p_tgt(grid_tgt: dict, mass_moist_air_tgt: XR_DATAARRAY, p_top: NP_REAL) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    n_z: NP_INT = NP_INT(grid_tgt["z"].size)
    n_zh: NP_INT = NP_INT(grid_tgt["zh"].size)
    n_y: NP_INT = NP_INT(grid_tgt["y"].size)
    n_x: NP_INT = NP_INT(grid_tgt["x"].size)

    xh: NP_ARRAY[NP_REAL] = grid_tgt["xh"] # [n_xh]; [m]
    dx: NP_REAL = xh[1] - xh[0] # [m]

    yh: NP_ARRAY[NP_REAL] = grid_tgt["yh"] # [n_yh]; [m]
    dy: NP_REAL = yh[1] - yh[0] # [m]

    #---------------------------------------------------------------------------
    # Calculate hydrostatic pressure at RTE-RRTMGP-CPP levels, save to xarray data array
    #---------------------------------------------------------------------------
    p_lev_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((n_zh, n_y, n_x), dtype = NP_REAL)
    p_lev_tgt_data[-1,...] = p_top
    p_lev_tgt_data[:-1,...] = (
        (mass_moist_air_tgt
            .isel(z = slice(None, None, -1))
            .cumsum(dim = "z")
            .isel(z = slice(None, None, -1)))
        * (g / (dx * dy)) 
        + p_top).to_numpy().astype(NP_REAL)

    p_lev_tgt: XR_DATAARRAY = XR_DATAARRAY(data = p_lev_tgt_data,
        dims = ("zh", "y", "x"),
        coords = {
            "time" : mass_moist_air_tgt.coords["time"].copy(deep = True),
            "zh" : grid_tgt["zh"],
            "y" : grid_tgt["y"],
            "x" : grid_tgt["x"],
        },
        name = "p_lev",
        attrs = {
            "units": "Pa",
            "long_name": "hydrostatic_pressure_at_levels",
            "standard_name": "hydrostatic_pressure_at_levels",
            "cell_methods": mass_moist_air_tgt.attrs.get("cell_methods", "time: point"),
        })

    #---------------------------------------------------------------------------
    # Calculate hydrostatic pressure at RTE-RRTMGP-CPP layers
    #---------------------------------------------------------------------------
    p_lay_tgt_data: NP_ARRAY[NP_REAL] = np.zeros((n_z, n_y, n_x), dtype = NP_REAL)
    p_lay_tgt_data = (
        (mass_moist_air_tgt
            .isel(z = slice(None, None, -1))
            .cumsum(dim = "z")
            .isel(z = slice(None, None, -1))
        - mass_moist_air_tgt / 2.)
        * (g / (dx * dy)) 
        + p_top).to_numpy().astype(NP_REAL)

    p_lay_tgt: XR_DATAARRAY = XR_DATAARRAY(data = p_lay_tgt_data,
        dims = ("z", "y", "x"),
        coords = {
            "time" : mass_moist_air_tgt.coords["time"].copy(deep = True),
            "z" : grid_tgt["z"],
            "y" : grid_tgt["y"],
            "x" : grid_tgt["x"],
        },
        name = "p_lay",
        attrs={
            "units": "Pa",
            "long_name": "hydrostatic_pressure_at_layers",
            "standard_name": "hydrostatic_pressure_at_layers",
            "cell_methods": mass_moist_air_tgt.attrs.get("cell_methods", "time: point"),
        })

    return [p_lay_tgt, p_lev_tgt] # Refers to RTE-RRTMGP-CPP layers and levels

def calc_vmr_gas_tgt(nmole_gas_tgt: XR_DATAARRAY, nmole_dry_air_tgt: XR_DATAARRAY) -> XR_DATAARRAY:
    #---------------------------------------------------------------------------
    # Extract dimensions
    #---------------------------------------------------------------------------
    n_z: NP_INT = NP_INT(nmole_gas_tgt["z"].size)
    n_y: NP_INT = NP_INT(nmole_gas_tgt["y"].size)
    n_x: NP_INT = NP_INT(nmole_gas_tgt["x"].size)

    #---------------------------------------------------------------------------
    # Get VMR values, avoid zeros
    #---------------------------------------------------------------------------
    vmr_gas_data: NP_ARRAY[NP_REAL] = np.zeros((n_z, n_y, n_x), dtype = NP_REAL)
    nonzero_mask: NP_ARRAY[NP_BOOL] = nmole_dry_air_tgt > 0.
    vmr_gas_data[nonzero_mask] = (
        nmole_gas_tgt.to_numpy().astype(NP_REAL)[nonzero_mask] 
        / nmole_dry_air_tgt.to_numpy().astype(NP_REAL)[nonzero_mask])

    #---------------------------------------------------------------------------
    # Construct target xarray data array
    #---------------------------------------------------------------------------
    gas_key: str = re.sub("nmole_", "", nmole_gas_tgt.name)
    vmr_gas_tgt: XR_DATAARRAY = XR_DATAARRAY(data = vmr_gas_data,
        dims = ("z", "y", "x"),
        coords = nmole_gas_tgt.coords,
        name = "vmr_" + gas_key,
        attrs = {
            "units": "mol mol^{-1}",
            "long_name": gas_key + "_dry_volume_mixing_ratio_at_layers",
            "standard_name": gas_key + "_vmr_at_layers",
            "cell_methods": nmole_gas_tgt.attrs.get("cell_methods", "time: point"),
        })

    return vmr_gas_tgt

def calc_wp(mass_water: XR_DATAARRAY) -> XR_DATAARRAY:
    # Calculates a water path from a water mass
    #---------------------------------------------------------------------------
    # Extract grid spacing
    #---------------------------------------------------------------------------
    x: NP_ARRAY[NP_REAL] = mass_water["x"].to_numpy().astype(NP_REAL) # [n_x]; [m]
    dx: NP_REAL = x[1] - x[0] # [m]

    y: NP_ARRAY[NP_REAL] = mass_water["y"].to_numpy().astype(NP_REAL) # [n_y]; [m]
    dy: NP_REAL = y[1] - y[0] # [m]

    #---------------------------------------------------------------------------
    # Calculate water path and save it to xarray data array
    #---------------------------------------------------------------------------
    wp: XR_DATARRAY = (mass_water / (dx * dy)) * 1.e3 # [kg m^{-2}] => [g m^{-2}]
    wp_name: str = re.sub("mass_", "", mass_water.name) + "p"
    wp_long_name: str = re.sub("mass", "vertical path", mass_water.long_name)
    wp_standard_name: str = re.sub("mass", "vertical_path", mass_water.standard_name)
    wp = (wp
        .rename(wp_name)
        .assign_attrs({"units" : "g m^{-2}",
            "long_name" : wp_long_name,
            "standard_name" : wp_standard_name}))

    return wp