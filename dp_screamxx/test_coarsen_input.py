"""
Horizontally coarsens output to RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import ast
import os

from typing import Optional

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc
from scipy.interpolate import RBFInterpolator, griddata
import xarray as xr

# Local Library Imports
from consts import NP_INT, NP_REAL, NP_ARRAY, NC_VARIABLE, \
    XR_DATASET, g
from rte_rrtmgp_cpp_fields import grid_dimensions, grid_descriptions, \
    grid_units, fields_dimensions, fields_descriptions, fields_units

# Type aliases

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "coarsen_rte_rrtmgp_cpp_input",
        description = "Horizontally coarsens input to RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input_root",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["rte_rrtmgp_input"],
        help = "Path to RTE-RRTMGP-CPP input file, with desired base name of the file.")

    parser.add_argument("--method",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["nearest"],
        help = "Interpolation method for vertical regridding [nearest, rbf].")

    parser.add_argument("--output_root",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["rte_rrtmgp_input"],
        help = "Path to RTE-RRTMGP-CPP input file, with desired base name of the file.")
    
    args: argparse.Namespace = parser.parse_args()
    
    input_file_root_path: str = os.path.normpath(args.input_root[0])
    method: str = args.method[0]
    assert(method in ["nearest", "rbf"])
    if args.szas[0] is None:
        szas: Optional[NP_ARRAY[NP_REAL]] = None
    else:
        szas: Optional[NP_ARRAY[NP_REAL]] = \
            np.array(ast.literal_eval(args.szas[0]), dtype = NP_REAL).flatten()
    output_file_root_path: str = os.path.normpath(args.output_root[0])

    ## Set file-independent parameters
    epsilon: float = 1.0 # Shape parameter in case of RBF interpolation

    ### NetCDF fields for RTE-RRTMGP-CPP output
    fields: dict = {}
    dpscream_3dfield_keys: list = ["p", "T", "RelativeHumidity", "qc", "qi",
        "eff_radius_qc", "eff_radius_qi", "ch4_volume_mix_ratio",
        "co_volume_mix_ratio", "co2_volume_mix_ratio",
        "h2o_volume_mix_ratio", "n2_volume_mix_ratio",
        "n2o_volume_mix_ratio", "o2_volume_mix_ratio",
        "o3_volume_mix_ratio"]
    rte_3dfield_keys: list = ["p", "t", "rh", "lwp", "iwp", "rel", "dei", 
        "vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o", 
        "vmr_o2", "vmr_o3"]
    dpscream_2dfield_keys: list[str] = ["surf_radiative_T",
        "cosine_solar_zenith_angle"]
    rte_2dfield_keys: list[str] = ["t_sfc", "mu0"]

    ## Open the DP-SCREAM output file
    input_file_path: str = input_file_root_path + ".nc"
    xr_input: XR_DATASET = xr.open_dataset(input_file_path, engine = "netcdf4")

    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_input["lon"].values # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_input["lat"].values # Column center - y-dimension [m]; (ncol)

    sort_mask: NP_ARRAY[NP_INT] = np.lexsort((lon, lat)) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    n_col_x: int = np.unique(lon).size # No. columns in x
    n_col_y: int = np.unique(lat).size # No. columns in y
    cols: NP_ARRAY[NP_REAL] = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(n_col_x, n_col_y, 2)

    ## Construct the horizontal grids
    ### NOTE: The names xh, yh seem to refer to the interfaces between columns,
    ### but in the original rcemip experiment, they just tack on an extra value.
    ### They don't seem to be directly used in the code, so we will use them
    ### to be interfaces between columns.
    ### NOTE: Assume that horizontal grids are regularly spaced.
    x: NP_ARRAY[NP_REAL] = (cols[:,:,0])[0,:] # x-midpoints of each column [m]; (n_col_x)
    dx: NP_REAL = x[1] - x[0]
    xh: NP_ARRAY[NP_REAL] = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-interfaces of each column [m]; (n_col_x + 1)

    y: NP_ARRAY[NP_REAL] = (cols[:,:,1])[:,0] # x-midpoints of each column [m]; (n_col_y)
    dy: NP_REAL = y[1] - y[0]
    yh: NP_ARRAY[NP_REAL] = np.append(y - (dy / 2.), x[-1] + (dy / 2.)) # y-interfaces of each column [m]; (n_col_y + 1)

    ## Dimension sizes - DP-SCREAM
    ntime: int = xr_input.sizes["time"] # No. time-steps
    ncol: int = xr_input.sizes["ncol"] # No. columns
    nlev: int = xr_input.sizes["lev"] # No. levels (layers)
    nilev: int = xr_input.sizes["ilev"] # No. level (layer) interfaces

    ## Dimension sizes - RTE-RRTMGP-CPP+RT - Only ones that need to be renamed
    n_lay_z: int = nlev # DP-SCREAM "levels" = RTE-RRTMGP-CPP+RT "layers"
    n_lev_z: int = nilev # DP_SCREAM "level interfaces" = RTE-RRTMGP-CPP+RT "levels"
    n_z: int = n_lay_z + n_lev_z

    ## Store spatial grid for outputting to RTE-RRTMGP-CPP input file
    grid: dict = {}

    ### NOTE: The number of points in the horizontal and vertical acceleration grids "should"
    ### be between 1/10 and 1/20 of n_col_x, n_col_y, n_col_z
    ### NOTE: These are the time-independent quantities
    ngrid_x: int = np.ceil(n_col_x / 10)
    ngrid_y: int = np.ceil(n_col_y / 10)
    ngrid_z: int = np.ceil(n_lay_z / 10)

    grid["x"] = x
    grid["xh"] = xh

    grid["y"] = y
    grid["yh"] = yh

    grid["ngrid_x"] = ngrid_x
    grid["ngrid_y"] = ngrid_y
    grid["ngrid_z"] = ngrid_z

    ## Time-independent quantities
    ## Special fields specified in the DP-SCREAM output
    ## Obtain the number of shortwave and longwave bands.
    swband: NP_ARRAY[NP_REAL] = xr_input["swband"].values # Shortwave bands [cm^(-1)]; (n_bnd_sw)
    lwband: NP_ARRAY[NP_REAL] = xr_input["lwband"].values # Longwave bands [cm^(-1)]; (n_bnd_lw)

    n_bnd_sw: int = swband.size
    n_bnd_lw: int = lwband.size

    ## Fields that are not specified in the DP-SCREAM output
    ## Longwave boundary conditions
    fields["emis_sfc"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x, n_bnd_lw), dtype = NP_REAL) # Surface emissivity [N/A]

    ## Shortwave boundary conditions
    fields["sfc_alb_dir"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x, n_bnd_sw), dtype = NP_REAL) * 0.07 # Surface Albedo - Direct
    fields["sfc_alb_dif"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x, n_bnd_sw), dtype = NP_REAL) * 0.07 # Surface Albedo - Diffuse

    fields["tsi"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x), dtype = NP_REAL) * 551.58 # Total Solar Irradiance [W m^(-2)]

    fields["azi"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x), dtype = NP_REAL) * 0.0 # Azimuthal Angle [Radians]

    ## Set quantities not expected to be set in the DP-SCREAM output
    ### Gas volume mixing ratios
    fields["vmr_ccl4"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Carbon Tetrachloride [kg kg^(-1)]
    fields["vmr_cfc11"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Trichlorofluoromethane (CFC-11) [kg kg^(-1)]
    fields["vmr_cfc12"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Dichlorodifluoromethane (CFC-12) [kg kg^(-1)]
    fields["vmr_cfc22"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Chlorodifluoromethane (HCFC-22) [kg kg^(-1)]
    fields["vmr_hfc143a"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # 1,1,1-Trifluoroethane (HFC-143a) [kg kg^(-1)]
    fields["vmr_hfc125"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Pentafluoroethane (HFC-125) [kg kg^(-1)]
    fields["vmr_hfc32"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Difluoromethane (HFC-32) [kg kg^(-1)]
    fields["vmr_hfc23"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Trifluoromethane (HFC-23) [kg kg^(-1)]
    fields["vmr_hfc134a"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # 1,1,1,2-Tetrafluoroethane (HFC-134a) [kg kg^(-1)]
    fields["vmr_cf4"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Carbon Tetrafluoride (CF₄) [kg kg^(-1)]
    fields["vmr_no2"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Nitrogen Dioxide [kg kg^(-1)]

    ### Aerosol mixing ratios
    fields["aermr01"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.03 - 0.5 µm)
    fields["aermr02"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.5 - 5 µm)
    fields["aermr03"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (5 - 20 µm)
    fields["aermr04"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.03 - 0.55 µm)
    fields["aermr05"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.55 - 0.9 µm)
    fields["aermr06"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.9 - 20 µm)
    fields["aermr07"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophilic Organic Matter Aerosol
    fields["aermr08"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophobic Organic Matter Aerosol
    fields["aermr09"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophilic Black Carbon Aerosol
    fields["aermr10"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophobic Black Carbon Aerosol
    fields["aermr11"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x)) # Sulfate Aerosol

    ## For large file sizes, we must go through time-step by time-step
    for tt in range(0, ntime):
        ## Reconstruct the vertical grids (time-dependent)
        z_mid: NP_ARRAY[NP_REAL] = xr_input["z_mid"].isel(time = tt, ncol = sort_mask).values # Level midpoints [m]; (ncol, n_lay_z)
        z_int: NP_ARRAY[NP_REAL] = xr_input["z_int"].isel(time = tt, ncol = sort_mask).values # Level interfaces [m]; (ncol, n_lev_z)

        z_mid: NP_ARRAY[NP_REAL] = z_mid.reshape(n_col_x, n_col_y, n_lay_z) # Layer midpoints [m]; (n_col_x, n_col_y, n_lay_z)
        z_int: NP_ARRAY[NP_REAL] = z_int.reshape(n_col_x, n_col_y, n_lev_z) # Layer interfaces [m]; (n_col_x, n_col_y, n_lev_z)

        ### Interleave these into a single vertical grid
        z: NP_ARRAY[NP_REAL] = np.empty([n_col_x, n_col_y, n_z], dtype = NP_REAL)
        z[:,:,1::2] = z_mid
        z[:,:,0::2] = z_int

        ### Create a regularly-spaced grid for interpolating variables to.
        z_min: NP_REAL = NP_REAL(0.0)
        z_max: NP_REAL = z.max()

        z_lev: NP_ARRAY[NP_REAL] = np.linspace(z_min, z_max, n_lev_z) # Regularly-spaced layer interfaces [m]; (n_lev_z)
        z_lay: NP_ARRAY[NP_REAL] = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced layer midpoints [m]; (n_lay_z)

        ## Store vertical grid
        grid["z"] = z_lay
        grid["zh"] = z_lev
        grid["z_lay"] = z_lay
        grid["z_lev"] = z_lev

        ## Read the 3D fields from the file, remap them to regular z-levels, and store them
        for ii in range(len(dpscream_3dfield_keys)):
            dpscream_field_key: str = dpscream_3dfield_keys[ii]
            rte_field_key: str = rte_3dfield_keys[ii]
            ## Fields either have layer midpoint and interface values with an 
            ## additional string in their key, or just layer midpoint values that are
            ## just the field key
            if dpscream_field_key in xr_input.keys(): # Only have values at midpoints
                field_mid: NP_ARRAY[NP_REAL] = xr_input[dpscream_field_key].isel(time = tt, ncol = sort_mask).values # Field at layer midpoints; (ncol, n_lay_z)
                field_int: Optional[NP_ARRAY[NP_REAL]] = None # Field at layer interfaces; (time, ncol, n_lay_z)

                z_field: np.ndarray = z_mid
            else: # Should have values and midpoints and interfaces
                dpscream_field_key_mid: str = dpscream_field_key + "_mid"
                ## Exceptions
                if dpscream_field_key in ["T"]:
                    dpscream_field_key_int: str = dpscream_field_key + "_int_rad"
                else:
                    dpscream_field_key_int: str = dpscream_field_key + "_int"

                ## We should always have fields values at layer midpoints
                ## Unless we don't, then this needs to be fixed
                assert(dpscream_field_key_mid in xr_input.keys())
                field_mid: NP_ARRAY[NP_REAL] = xr_input[dpscream_field_key_mid].isel(time = tt, ncol = sort_mask).values # Field at layer midpoints; (ncol, n_lay_z)

                if dpscream_field_key_int in xr_input.keys(): # Actually have values at midpoints and interfaces
                    field_int: Optional[NP_ARRAY[NP_REAL]] = xr_input[dpscream_field_key_int].isel(time = tt, ncol = sort_mask).values # Field at layer interfaces; (time, ncol, n_lay_z)
                    z_field: np.ndarray = z
                else: # Actually only have values at midpoints
                    field_int: Optional[NP_ARRAY[NP_REAL]] = None # Field at layer interfaces; (ncol, n_lay_z)
                    z_field: np.ndarray = z_mid

            ## Reshape into x- and y-columns
            field_mid: NP_ARRAY[NP_REAL] = field_mid.reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)
            if field_int is not None:
                field_int: Optional[NP_ARRAY[NP_REAL]] = field_int.reshape(n_col_x, n_col_y, n_lev_z) # (n_col_x, n_col_y, n_lev_z)

            ## If we have field values at layer midpoints and interfaces, interleave them
            if field_int is not None:
                field: NP_ARRAY[NP_REAL] = np.empty([n_col_x, n_col_y, n_z], dtype = NP_REAL)
                field[:,:,1::2] = field_mid
                field[:,:,0::2] = field_int
            else:
                field: NP_ARRAY[NP_REAL] = field_mid

            ## Exceptions
            if rte_field_key in ["dei"]: # DP-SCREAM has rei, RTE-RRTMGP-CPP has dei
                field: NP_ARRAY[NP_REAL] = 2. * field
            elif rte_field_key in ["lwp", "iwp"]: # Derived from multiple quantities
                p_int: NP_ARRAY[NP_REAL] = xr_input["p_int"].isel(time = tt, ncol = sort_mask).values # Pressure at layer interfaces [Pa]; (ncol, n_lev_z)
                p_int: NP_ARRAY[NP_REAL] = p_int.reshape(n_col_x, n_col_y, n_lev_z) # (n_col_x, n_col_y, n_lev_z)
                dp: NP_ARRAY[NP_REAL] = p_int[:,:,1:] - p_int[:,:,:-1] # Layer pressure thickness [Pa]; (n_col_x, n_col_y, n_lay_z)

                field: NP_ARRAY[NP_REAL] = field * dp / g

            ## Get field min and max
            ## Exceptions
            if rte_field_key in ["rel"]: # Between 2.5 μm and 21.5 μm
                field_min: NP_REAL = NP_REAL(2.5)
                field_max: NP_REAL = NP_REAL(21.5)
            elif rte_field_key in ["dei"]: # Between 10. μm and 180. μm
                field_min: NP_REAL = NP_REAL(10.)
                field_max: NP_REAL = NP_REAL(180.)
            else:
                field_min: NP_REAL = field.min()
                field_max: NP_REAL = field.max()

            ## Interpolate the values to regular vertical layers
            if ((method == "nearest") or (rte_field_key == "vmr_h2o")):
                field_lay: NP_ARRAY[NP_REAL] = np.transpose(remap_z(z_field, field, z_lay), axes = (2, 1, 0)) # Field at regular layer midpoints; (n_lay_z, n_col_y, n_col_x)
                field_lev: NP_ARRAY[NP_REAL] = np.transpose(remap_z(z_field, field, z_lev), axes = (2, 1, 0)) # Field at regular layer interfaces; (n_lev_z, n_col_y, n_col_x)
            elif method == "rbf":
                field_rbfinterpolator_z: NP_ARRAY[NP_REAL] = rbfinterpolator_z(z_field, field, epsilon = epsilon) # Interpolant of field in each column
                field_lay: NP_ARRAY[NP_REAL] = np.transpose(eval_rbfinterpolator_z(field_rbfinterpolator_z, z_lay), axes = (2, 1, 0)) # Field at regular layer midpoints; (n_lay_z, n_col_y, n_col_x)
                field_lev: NP_ARRAY[NP_REAL] = np.transpose(eval_rbfinterpolator_z(field_rbfinterpolator_z, z_lev), axes = (2, 1, 0)) # Field at regular layer interfaces; (n_lev_z, n_col_y, n_col_x)

            ## Limit the interpolated (and extrapolated) field values
            ## Exceptions:
            field_lay[field_lay < field_min] = field_min
            field_lay[field_lay > field_max] = field_max

            field_lev[field_lev < field_min] = field_min
            field_lev[field_lev > field_max] = field_max

            ## Exceptions
            if rte_field_key in ["rh", "q", "lwp", "iwp", "rel", "dei",
                "vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o",
                "vmr_o2", "vmr_o3"]:
                fields[rte_field_key] = field_lay
            else:
                rte_field_key_lay: str = rte_field_key + "_lay"
                rte_field_key_lev: str = rte_field_key + "_lev"

                fields[rte_field_key_lay] = field_lay
                fields[rte_field_key_lev] = field_lev

        ## Extract 2-D fields
        for ii in range(len(dpscream_2dfield_keys)):
            dpscream_field_key: str = dpscream_2dfield_keys[ii]
            rte_field_key: str = rte_2dfield_keys[ii]
            assert(dpscream_field_key in xr_input.keys())

            field: NP_ARRAY[NP_REAL] = xr_input[dpscream_field_key].isel(time = tt, ncol = sort_mask).values # 2-D field; (ncol)

            ## Reshape into x- and y-columns
            field: NP_ARRAY[NP_REAL] = field.reshape(n_col_x, n_col_y) # (n_col_x, n_col_y)
            field: NP_ARRAY[NP_REAL] = np.transpose(field, axes = (1, 0)) # (n_col_y, n_col_x)

            ## Exceptions
            if rte_field_key in ["t_sfc", "mu0"]: # In case fill values are unreasonable
                if rte_field_key in ["t_sfc"]: # Between 0 K and 2300 K (max natural temperature on Earth)
                    field_min: NP_REAL = NP_REAL(0.0)
                    field_max: NP_REAL = NP_REAL(2300.0)
                elif rte_field_key in ["mu0"]: # Between -1.0 and 1.0
                    field_min: NP_REAL = NP_REAL(-1.0)
                    field_max: NP_REAL = NP_REAL(1.0)

                field[field > field_max] = field_max
                field[field < field_min] = field_min

            fields[rte_field_key] = field

        ## Write to RTE-RRTMGP-CPP input file, with the varying given solar zenith angles
        nc_float: str = "f8"

        if szas is not None:
            for sza in szas:
                sza_rad: NP_REAL = np.deg2rad(sza)
                fields["mu0"]: NP_REAL = np.zeros((ntime, n_col_y, n_col_x)) + np.cos(sza_rad) # Cosine of SZA

                time_str: str = "{:03d}".format(tt)
                sza_str: str = "{:03.0f}".format(sza)
                output_file_path: str = output_file_root_path + "." + time_str + "." + sza_str + ".in.nc"

                breakpoint()

                nc_file: NC_DATASET = nc.Dataset(output_file_path, mode = "w",
                    datamodel = "NETCDF4", clobber = True)

                nc_file.createDimension("x", n_col_x)
                nc_file.createDimension("y", n_col_y)
                nc_file.createDimension("lay", n_lay_z)
                nc_file.createDimension("lev", n_lev_z)
                nc_file.createDimension("z", n_lay_z)
                nc_file.createDimension("xh", n_col_x + 1)
                nc_file.createDimension("yh", n_col_y + 1)
                nc_file.createDimension("zh", n_lev_z)
                nc_file.createDimension("band_lw", n_bnd_lw)
                nc_file.createDimension("band_sw", n_bnd_sw)

                ## Current time
                nc_curr_time: NC_VARIABLE = nc_file.createVariable("time", nc_float)
                nc_curr_time.description: str = "Time since simulation start"
                nc_curr_time.units: str = "days"
                nc_curr_time[0] = xr_input["time"].isel(time = tt)

                ## Spatial grid
                for rte_grid_key in grid_dimensions.keys():
                    field: NP_ARRAY[NP_REAL] = grid[rte_grid_key]
                    field_dimensions: str | tuple[Optional[str]] = grid_dimensions[rte_grid_key]
                    field_description: str = grid_descriptions[rte_grid_key]
                    field_units: str = grid_units[rte_grid_key]

                    nc_field: NC_VARIABLE = nc_file.createVariable(rte_grid_key, nc_float, field_dimensions)
                    nc_field.description: str = field_description
                    nc_field.units: str = field_units
                    nc_field[...]: NP_ARRAY[NP_REAL] = field

                ## Fields
                for rte_field_key in fields_dimensions:
                    field: NP_ARRAY[NP_REAL] = fields[rte_field_key][:]

                    field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
                    field_description: str = fields_descriptions[rte_field_key]
                    field_units: str = fields_units[rte_field_key]

                    nc_field: NC_VARIABLE = nc_file.createVariable(rte_field_key, nc_float, field_dimensions)
                    nc_field.description: str = field_description
                    nc_field.units: str = field_units
                    nc_field[...]: NP_ARRAY[NP_REAL] = field

                nc_file.close()
        else:
            time_str: str = "{:03d}".format(tt)
            output_file_path: str = output_file_root_path + "." + time_str + ".in.nc"

            nc_file: NC_DATASET = nc.Dataset(output_file_path, mode = "w",
                datamodel = "NETCDF4", clobber = True)

            nc_file.createDimension("x", n_col_x)
            nc_file.createDimension("y", n_col_y)
            nc_file.createDimension("lay", n_lay_z)
            nc_file.createDimension("lev", n_lev_z)
            nc_file.createDimension("z", n_lay_z)
            nc_file.createDimension("xh", n_col_x + 1)
            nc_file.createDimension("yh", n_col_y + 1)
            nc_file.createDimension("zh", n_lev_z)
            nc_file.createDimension("band_lw", n_bnd_lw)
            nc_file.createDimension("band_sw", n_bnd_sw)

            ## Current time
            nc_curr_time: NC_VARIABLE = nc_file.createVariable("time", nc_float)
            nc_curr_time.description: str = "Time since simulation start"
            nc_curr_time.units: str = "days"
            nc_curr_time[0] = xr_input["time"].isel(time = tt)

            ## Spatial grid
            for rte_grid_key in grid_dimensions.keys():
                field: NP_ARRAY[NP_REAL] = grid[rte_grid_key]
                field_dimensions: str | tuple[Optional[str]] = grid_dimensions[rte_grid_key]
                field_description: str = grid_descriptions[rte_grid_key]
                field_units: str = grid_units[rte_grid_key]

                nc_field: NC_VARIABLE = nc_file.createVariable(rte_grid_key, nc_float, field_dimensions)
                nc_field.description: str = field_description
                nc_field.units: str = field_units
                nc_field[...]: NP_ARRAY[NP_REAL] = field

            ## Fields
            for rte_field_key in fields_dimensions:
                field: NP_ARRAY[NP_REAL] = fields[rte_field_key][:]

                field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
                field_description: str = fields_descriptions[rte_field_key]
                field_units: str = fields_units[rte_field_key]

                nc_field: NC_VARIABLE = nc_file.createVariable(rte_field_key, nc_float, field_dimensions)
                nc_field.description: str = field_description
                nc_field.units: str = field_units
                nc_field[...]: NP_ARRAY[NP_REAL] = field

            nc_file.close()

def remap_z(z_src: tuple, values_src: NP_ARRAY[NP_REAL], z_tgt: tuple, 
    method: str = "nearest", fill_value: NP_REAL | float = np.nan, 
    rescale: bool = False) -> NP_ARRAY[NP_REAL]:
    
    ## z_src is the time-dependent, spatially varying vertical grid of values_src
    nx: int
    ny: int

    nx, ny = values_src.shape[0:2]
    nz: int = z_tgt.shape[0]

    values_tgt: NP_ARRAY[NP_REAL] = np.zeros((nx, ny, nz), dtype = NP_REAL)

    for ii in range(0, nx):
        for jj in range(0, ny):
            values_tgt[ii, jj, :] = \
                griddata(z_src[ii, jj, :], values_src[ii, jj, :], 
                z_tgt, method = method, fill_value = fill_value, 
                rescale = rescale)
            
    return values_tgt

def rbfinterpolator_z(z_src: NP_ARRAY[NP_REAL], values_src: NP_ARRAY[NP_REAL],
    neighbors: Optional[int] = None, smoothing: NP_REAL | float = 0.0,
    kernel: str = "thin_plate_spline", epsilon: Optional[float] = None,
    degree: Optional[int] = None) -> NP_ARRAY[NP_REAL]:
    ## z_src is the time-dependent, spatially-varying vertical grid of values_src
    nx: int
    ny: int

    nx, ny = values_src.shape[0:2]

    interp: NP_ARRAY[RBFInterpolator] = np.empty((nx, ny), dtype = "O")

    for ii in range(0, nx):
        for jj in range(0, ny):
            interp[ii, jj] = \
                RBFInterpolator(np.expand_dims(z_src[ii, jj, :], axis = 1),
                    values_src[ii, jj, :], neighbors = neighbors,
                    smoothing = smoothing, kernel = kernel, epsilon = epsilon,
                    degree = degree)
            
    return interp

def eval_rbfinterpolator_z(interp: NP_ARRAY[NP_REAL],
    z_tgt: NP_ARRAY[NP_REAL]) -> NP_ARRAY[NP_REAL]:
    ## z_src is the time-dependent, spatially-varying vertical grid of values_src
    ## z_tgt is the target vertical grid, is time- and space-independent
    nx: int
    ny: int

    nx, ny = interp.shape[0:2]
    nz: int = z_tgt.shape[0]

    values_tgt: NP_ARRAY[NP_REAL] = np.empty((nx, ny, nz), dtype = NP_REAL)

    for ii in range(0, nx):
        for jj in range(0, ny):
            values_tgt[ii, jj, :] = \
                interp[ii,jj](np.expand_dims(z_tgt, axis = 1))

    return values_tgt

if __name__ == "__main__":
    main()

