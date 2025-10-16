"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

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

# Local Library Imports
from consts import R_d, R_v, g, np_float
from rte_rrtmgp_cpp_fields import grid_dimensions, grid_descriptions, grid_units, \
    fields_dimensions, fields_descriptions, fields_units

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "convert_dpscream_output",
        description = "Creates output from DP-SCREAM to input to RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input_root",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to DP-SCREAM output.")

    parser.add_argument("--method",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["nearest"],
        help = "Interpolation method for vertical regridding [nearest, rbf].")
    
    parser.add_argument("--szas",
        action = "store",
        nargs = 1,
        type = Optional[str],
        required = False,
        default = [None],
        help = "Solar zenith angles to create RTE-RRTMGP-CPP input for.")

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
        szas: Optional[np.ndarray] = None
    else:
        szas: Optional[np.ndarray] = np.array(ast.literal_eval(args.szas[0]), dtype = np_float).flatten()
    output_file_root_path: str = os.path.normpath(args.output_root[0])

    ## Set file-independent parameters
    epsilon: float = 1.0 # Shape parameter in case of RBF interpolation

    ### NetCDF fields for RTE-RRTMGP-CPP output
    fields: dict = {}
    dpscream_field_keys: list = ["p", "T", "RelativeHumidity", "qc", "qi",
                                 "eff_radius_qc", "eff_radius_qi", "ch4_volume_mix_ratio",
                                 "co_volume_mix_ratio", "co2_volume_mix_ratio",
                                 "h2o_volume_mix_ratio", "n2_volume_mix_ratio",
                                 "n2o_volume_mix_ratio", "o2_volume_mix_ratio",
                                 "o3_volume_mix_ratio"]
    rte_field_keys: list = ["p", "t", "rh", "lwp", "iwp", "rel", "dei", "vmr_ch4",
                            "vmr_co", "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o", "vmr_o2",
                            "vmr_o3"]

    ## Open the DP-SCREAM output file
    input_file_path: str = input_file_root_path + ".nc"
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)

    ## Time array
    time: np.ma.MaskedArray = nc_input.variables["time"][:].astype(np_float) # Time since start date [days]; (time)
    ntime: int = time.size

    ## Reconstruct the horizontal grid
    lon: np.ma.MaskedArray = nc_input.variables["lon"][:].astype(np_float) # Column-center - x-dimension [m]; (ncol)
    lat: np.ma.MaskedArray = nc_input.variables["lat"][:].astype(np_float) # Column center - y-dimension [m]; (ncol)

    sort_mask: np.ndarray = np.lexsort((lon, lat)) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    n_col_x: int = np.unique(lon).size # No. columns in x
    n_col_y: int = np.unique(lat).size # No. columns in y
    cols: np.ma.MaskedArray = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(n_col_x, n_col_y, 2)

    ### NOTE: The names xh, yh seem to refer to the interfaces between columns,
    ### but in the original rcemip experiment, they just tack on an extra value.
    ### They don't seem to be directly used in the code, so we will use them
    ### to be interfaces between columns.
    ### NOTE: Assume that horizontal grids are regularly spaced.
    x: np.ma.MaskedArray = (cols[:,:,0])[0,:] # x-midpoints of each column [m]; (n_col_x)
    dx: np_float = x[1] - x[0]
    xh: np.ma.MaskedArray = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-interfaces of each column [m]; (n_col_x + 1)

    y: np.ma.MaskedArray = (cols[:,:,1])[:,0] # x-midpoints of each column [m]; (n_col_y)
    dy: np_float = y[1] - y[0]
    yh: np.ma.MaskedArray = np.append(y - (dy / 2.), x[-1] + (dy / 2.)) # y-interfaces of each column [m]; (n_col_y + 1)

    ## Reconstruct the vertical grids (time-dependent)
    z_mid: np.ma.MaskedArray = nc_input.variables["z_mid"][:].astype(np_float) # Level midpoints [m]; (time, ncol, n_lay_z)
    z_int: np.ma.MaskedArray = nc_input.variables["z_int"][:].astype(np_float) # Level interfaces [m]; (time, ncol, n_lev_z)

    n_lay_z: int = z_mid.shape[2] # Assuming same number of vertical layers at each time
    n_lev_z: int = z_int.shape[2] # Assuming same number of vertical levels at each time
    assert(n_lev_z == n_lay_z + 1)

    z_mid: np.ma.MaskedArray = z_mid[:,sort_mask,:].reshape(ntime, n_col_x, n_col_y, n_lay_z) # Layer midpoints [m]; (time, n_col_x, n_col_y, n_lay_z)
    z_int: np.ma.MaskedArray = z_int[:,sort_mask,:].reshape(ntime, n_col_x, n_col_y, n_lev_z) # Layer interfaces [m]; (time, n_col_x, n_col_y, n_lev_z)
    
    ### Interleave these into a single vertical grid
    n_z: int = n_lay_z + n_lev_z
    z: np.ndarray = np.empty([ntime, n_col_x, n_col_y, n_z], dtype = z_mid.dtype)
    z[:,:,:,1::2] = z_mid.filled()
    z[:,:,:,0::2] = z_int.filled()

    ### Create a regularly-spaced grid for interpolating variables to.
    z_min: np_float = 0.0
    z_max: np_float = z_int.max()

    z_lev: np.ndarray = np.linspace(z_min, z_max, n_lev_z) # Regularly-spaced layer interfaces [m]; (n_lev_z)
    z_lay: np.ndarray = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced layer midpoints [m]; (n_lay_z)

    ## Store spatial grid for outputting to RTE-RRTMGP-CPP input file
    grid: dict = {}

    ### NOTE: The number of points in the horizontal and vertical acceleration grids "should"
    ### be between 1/10 and 1/20 of n_col_x, n_col_y, n_col_z
    ngrid_x: int = np.ceil(n_col_x / 10)
    ngrid_y: int = np.ceil(n_col_y / 10)
    ngrid_z: int = np.ceil(n_lay_z / 10)

    grid["x"] = x
    grid["xh"] = xh

    grid["y"] = y
    grid["yh"] = yh

    grid["z"] = z_lay
    grid["zh"] = z_lev
    grid["z_lay"] = z_lay
    grid["z_lev"] = z_lev

    grid["ngrid_x"] = ngrid_x
    grid["ngrid_y"] = ngrid_y
    grid["ngrid_z"] = ngrid_z

    ## Read the time-dependent fields from the file, remap them to regular z-levels, and store them
    for ii in range(len(dpscream_field_keys)):
        dpscream_field_key: str = dpscream_field_keys[ii]
        rte_field_key: str = rte_field_keys[ii]
        ## Fields either have layer midpoint and interface values with an 
        ## additional string in their key, or just layer midpoint values that are
        ## just the field key
        if dpscream_field_key in nc_input.variables.keys(): # Only have values at midpoints
            field_mid: np.ma.MaskedArray = nc_input.variables[dpscream_field_key][:].astype(np_float) # Field at layer midpoints; (time, ncol, n_lay_z)
            field_int: Optional[np.ma.MaskedArray] = None # Field at layer interfaces; (time, ncol, n_lay_z)

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
            assert(dpscream_field_key_mid in nc_input.variables.keys())
            field_mid: np.ma.MaskedArray = nc_input.variables[dpscream_field_key_mid][:].astype(np_float) # Field at layer midpoints; (time, ncol, n_lay_z)

            if dpscream_field_key_int in nc_input.variables.keys(): # Actually have values at midpoints and interfaces
                field_int: Optional[np.ma.MaskedArray] = nc_input.variables[dpscream_field_key_int][:].astype(np_float) # Field at layer interfaces; (time, ncol, n_lay_z)
                z_field: np.ndarray = z
            else: # Actually only have values at midpoints
                field_int: Optional[np.ma.MaskedArray] = None # Field at layer interfaces; (time, ncol, n_lay_z)
                z_field: np.ndarray = z_mid

        ## Reshape into x- and y-columns
        field_mid: np.ma.MaskedArray = field_mid[:,sort_mask,:].reshape(ntime, n_col_x, n_col_y, n_lay_z) # (time, n_col_x, n_col_y, n_lay_z)
        if field_int is not None:
            field_int: np.ma.MaskedArray = field_int[:,sort_mask,:].reshape(ntime, n_col_x, n_col_y, n_lev_z) # (time, n_col_x, n_col_y, n_lev_z)

        ## Adjust the masked value for volume mixing ratios
        if dpscream_field_key[-16:] == "volume_mix_ratio":
            field_mid.fill_value: float = 0.
            if field_int is not None:
                field_int.fill_value: float = 0.

        ## If we have field values at layer midpoints and interfaces, interleave them
        if field_int is not None:
            field: np.ndarray = np.empty([ntime, n_col_x, n_col_y, n_z], dtype = field_mid.dtype)
            field[:,:,:,1::2] = field_mid.filled()
            field[:,:,:,0::2] = field_int.filled()
        else:
            field: np.ndarray = field_mid.filled()

        ## Exceptions
        if rte_field_key in ["dei"]: # DP-SCREAM has rei, RTE-RRTMGP-CPP has dei
            field: np.ndarray = 2. * field
        elif rte_field_key in ["lwp", "iwp"]: # Derived from multiple quantities
            p_int: np.ma.MaskedArray = nc_input.variables["p_int"][:].astype(np_float) # Pressure at layer interfaces [Pa]; (time, ncol, n_lev_z)
            p_int: np.ma.MaskedArray = p_int[:,sort_mask,:].reshape(ntime, n_col_x, n_col_y, n_lev_z) # (time, n_col_x, n_col_y, n_lev_z)
            dp: np.ma.MaskedArray = p_int[:,:,:,1:] - p_int[:,:,:,:-1] # Layer pressure thickness [Pa]; (time, n_col_x, n_col_y, n_lay_z)

            field: np.ndarray = field * dp / g

        ## Get field min and max
        ## Exceptions
        if rte_field_key in ["rel"]: # Between 2.5 μm and 21.5 μm
            field_min: float = 2.5
            field_max: float = 21.5
        elif rte_field_key in ["dei"]: # Between 10. μm and 180. μm
            field_min: float = 10.
            field_max: float = 180.
        else:
            field_min: float = field.min()
            field_max: float = field.max()

        ## Interpolate the values to regular vertical layers
        ### MUST REWORK THIS TO WORK WITH TIME-DEPENDENT GRIDS
        if ((method == "nearest") or (rte_field_key == "vmr_h2o")):
            field_lay: np.ndarray = np.transpose(remap_z(z_field, field, z_lay), axes = (0, 3, 2, 1)) # Field at regular layer midpoints; (time, n_lay_z, n_col_y, n_col_x)
            field_lev: np.ndarray = np.transpose(remap_z(z_field, field, z_lev), axes = (0, 3, 2, 1)) # Field at regular layer interfaces; (time, n_lev_z, n_col_y, n_col_x)
        elif method == "rbf":
            field_rbfinterpolator_z: np.ndarray = rbfinterpolator_z(z_field, field, epsilon = epsilon) # Interpolant of field in each column
            field_lay: np.ndarray = np.transpose(eval_rbfinterpolator_z(field_rbfinterpolator_z, z_lay), axes = (0, 3, 2, 1)) # Field at regular layer midpoints; (n_lay_z, n_col_y, n_col_x)
            field_lev: np.ndarray = np.transpose(eval_rbfinterpolator_z(field_rbfinterpolator_z, z_lev), axes = (0, 3, 2, 1)) # Field at regular layer interfaces; (n_lev_z, n_col_y, n_col_x)

        ## Limit the interpolated (and extrapolated) field values
        ## Exceptions:
        field_lay[field_lay < field_min] = field_min
        field_lay[field_lay > field_max] = field_max

        field_lev[field_lev < field_min] = field_min
        field_lev[field_lev > field_max] = field_max

        ## Exceptions
        if rte_field_key in ["rh", "q", "lwp", "iwp", "rel", "dei", "vmr_ch4", "vmr_co",
                            "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o", "vmr_o2",
                            "vmr_o3"]:
            fields[rte_field_key] = field_lay
        else:
            rte_field_key_lay: str = rte_field_key + "_lay"
            rte_field_key_lev: str = rte_field_key + "_lev"

            fields[rte_field_key_lay] = field_lay
            fields[rte_field_key_lev] = field_lev

    ## Extract 2-D fields
    dpscream_field_keys: list = ["surf_radiative_T", "cosine_solar_zenith_angle"]
    rte_field_keys: list = ["t_sfc", "mu0"]

    for ii in range(len(dpscream_field_keys)):
        dpscream_field_key: str = dpscream_field_keys[ii]
        rte_field_key: str = rte_field_keys[ii]
        assert(dpscream_field_key in nc_input.variables.keys())

        field: np.ma.MaskedArray = nc_input.variables[dpscream_field_key][:].astype(np_float) # 2-D field; (time, ncol)

        ## Reshape into x- and y-columns
        field: np.ma.MaskedArray = field[:,sort_mask].reshape(ntime, n_col_x, n_col_y) # (time, n_col_x, n_col_y)
        field: np.ndarray = np.transpose(np.ma.getdata(field), axes = (0, 2, 1)) # (time, n_col_y, n_col_x)

        ## Exceptions
        if rte_field_key in ["t_sfc", "mu0"]: # In case fill values are unreasonable
            if rte_field_key in ["t_sfc"]: # Between 0 K and 2300 K (max natural temperature on Earth)
                field_min: float | np_float = 0.0
                field_max: float | np_float = 2300.0
            elif rte_field_key in ["mu0"]: # Between -1.0 and 1.0
                field_min: float | np_float = -1.0
                field_max: float | np_float = 1.0

            field[field > field_max] = field_max
            field[field < field_min] = field_min

        fields[rte_field_key] = field

    ## Special fields specified in the DP-SCREAM output
    ## Obtain the number of shortwave and longwave bands.
    swband: np.ma.MaskedArray = nc_input.variables["swband"][:].astype(np_float) # Shortwave bands [cm^(-1)]; (n_bnd_sw)
    lwband: np.ma.MaskedArray = nc_input.variables["lwband"][:].astype(np_float) # Longwave bands [cm^(-1)]; (n_bnd_lw)

    n_bnd_sw: int = swband.size
    n_bnd_lw: int = lwband.size

    ## Fields that are not specified in the DP-SCREAM output
    ## Longwave boundary conditions
    fields["emis_sfc"]: np.ndarray = np.ones((ntime, n_col_y, n_col_x, n_bnd_lw)) # Surface emissivity [N/A]

    ## Shortwave boundary conditions
    fields["sfc_alb_dir"]: np.ndarray = np.ones((ntime, n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Direct
    fields["sfc_alb_dif"]: np.ndarray = np.ones((ntime, n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Diffuse

    fields["tsi"]: np.ndarray = np.ones((ntime, n_col_y, n_col_x)) * 551.58 # Total Solar Irradiance [W m^(-2)]

    fields["azi"]: np.ndarray = np.ones((ntime, n_col_y, n_col_x)) * 0.0 # Azimuthal Angle [Radians]

    ## Set quantities not expected to be set in the DP-SCREAM output
    ### Gas volume mixing ratios
    fields["vmr_ccl4"]: np.ndarray    = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Carbon Tetrachloride [kg kg^(-1)]
    fields["vmr_cfc11"]: np.ndarray   = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Trichlorofluoromethane (CFC-11) [kg kg^(-1)]
    fields["vmr_cfc12"]: np.ndarray   = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Dichlorodifluoromethane (CFC-12) [kg kg^(-1)]
    fields["vmr_cfc22"]: np.ndarray   = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Chlorodifluoromethane (HCFC-22) [kg kg^(-1)]
    fields["vmr_hfc143a"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # 1,1,1-Trifluoroethane (HFC-143a) [kg kg^(-1)]
    fields["vmr_hfc125"]: np.ndarray  = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Pentafluoroethane (HFC-125) [kg kg^(-1)]
    fields["vmr_hfc32"]: np.ndarray   = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Difluoromethane (HFC-32) [kg kg^(-1)]
    fields["vmr_hfc23"]: np.ndarray   = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Trifluoromethane (HFC-23) [kg kg^(-1)]
    fields["vmr_hfc134a"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # 1,1,1,2-Tetrafluoroethane (HFC-134a) [kg kg^(-1)]
    fields["vmr_cf4"]: np.ndarray     = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Carbon Tetrafluoride (CF₄) [kg kg^(-1)]
    fields["vmr_no2"]: np.ndarray     = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Nitrogen Dioxide [kg kg^(-1)]

    ### Aerosol mixing ratios
    fields["aermr01"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.03 - 0.5 µm)
    fields["aermr02"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.5 - 5 µm)
    fields["aermr03"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (5 - 20 µm)
    fields["aermr04"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.03 - 0.55 µm)
    fields["aermr05"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.55 - 0.9 µm)
    fields["aermr06"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.9 - 20 µm)
    fields["aermr07"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Hydrophilic Organic Matter Aerosol
    fields["aermr08"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Hydrophobic Organic Matter Aerosol
    fields["aermr09"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Hydrophilic Black Carbon Aerosol
    fields["aermr10"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Hydrophobic Black Carbon Aerosol
    fields["aermr11"]: np.ndarray = np.zeros((ntime, n_lay_z, n_col_y, n_col_x)) # Sulfate Aerosol

    for tt in range(0, ntime):
        ## Write to RTE-RRTMGP-CPP input file, with the varying given solar zenith angles
        nc_float: str = "f8"

        if szas is not None:
            for sza in szas:
                sza_rad: np.float64 = np.deg2rad(sza)
                fields["mu0"]: np_float = np.zeros((ntime, n_col_y, n_col_x)) + np.cos(sza_rad) # Cosine of SZA

                time_str: str = "{:03d}".format(tt)
                sza_str: str = "{:03.0f}".format(sza)
                output_file_path: str = output_file_root_path + "." + time_str + "." + sza_str + ".in.nc"

                nc_file: nc.Dataset = nc.Dataset(output_file_path, mode = "w",
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
                nc_curr_time: nc._netCDF4.Variable = nc_file.createVariable("time", nc_float)
                nc_curr_time.description: str = "Time since simulation start"
                nc_curr_time.units: str = "days"
                nc_curr_time[0] = time[tt]

                ## Spatial grid
                for rte_grid_key in grid_dimensions.keys():
                    field: np.ndarray = grid[rte_grid_key]
                    field_dimensions: str | tuple[Optional[str]] = grid_dimensions[rte_grid_key]
                    field_description: str = grid_descriptions[rte_grid_key]
                    field_units: str = grid_units[rte_grid_key]

                    nc_field: nc._netCDF4.Variable = nc_file.createVariable(rte_grid_key, nc_float, field_dimensions)
                    nc_field.description: str = field_description
                    nc_field.units: str = field_units
                    nc_field[...]: np.ndarray = field

                ## Fields
                for rte_field_key in fields_dimensions:
                    field: np.ndarray = fields[rte_field_key][tt,...]

                    field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
                    field_description: str = fields_descriptions[rte_field_key]
                    field_units: str = fields_units[rte_field_key]

                    nc_field: nc._netCDF4.Variable = nc_file.createVariable(rte_field_key, nc_float, field_dimensions)
                    nc_field.description: str = field_description
                    nc_field.units: str = field_units
                    nc_field[...]: np.ndarray = field

                nc_file.close()
        else:
            time_str: str = "{:03d}".format(tt)
            output_file_path: str = output_file_root_path + "." + time_str + ".in.nc"

            nc_file: nc.Dataset = nc.Dataset(output_file_path, mode = "w",
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
            nc_curr_time: nc._netCDF4.Variable = nc_file.createVariable("time", nc_float)
            nc_curr_time.description: str = "Time since simulation start"
            nc_curr_time.units: str = "days"
            nc_curr_time[0] = time[tt]

            ## Spatial grid
            for rte_grid_key in grid_dimensions.keys():
                field: np.ndarray = grid[rte_grid_key]
                field_dimensions: str | tuple[Optional[str]] = grid_dimensions[rte_grid_key]
                field_description: str = grid_descriptions[rte_grid_key]
                field_units: str = grid_units[rte_grid_key]

                nc_field: nc._netCDF4.Variable = nc_file.createVariable(rte_grid_key, nc_float, field_dimensions)
                nc_field.description: str = field_description
                nc_field.units: str = field_units
                nc_field[...]: np.ndarray = field

            ## Fields
            for rte_field_key in fields_dimensions:
                field: np.ndarray = fields[rte_field_key][tt,...]

                field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
                field_description: str = fields_descriptions[rte_field_key]
                field_units: str = fields_units[rte_field_key]

                nc_field: nc._netCDF4.Variable = nc_file.createVariable(rte_field_key, nc_float, field_dimensions)
                nc_field.description: str = field_description
                nc_field.units: str = field_units
                nc_field[...]: np.ndarray = field

            nc_file.close()

def remap_z(z_src: tuple, values_src: np.ndarray, z_tgt: tuple, 
            method: str = "nearest", fill_value: float = np.nan, rescale: bool = False):
    
    ## z_src is the time-dependent, spatially varying vertical grid of values_src
    ntime: int
    nx: int
    ny: int

    ntime, nx, ny = values_src.shape[0:3]
    nz: int = z_tgt.shape[0]

    values_tgt: np.ndarray = np.zeros((ntime, nx, ny, nz), dtype = values_src.dtype)

    for tt in range(0, ntime):
        for ii in range(0, nx):
            for jj in range(0, ny):
                values_tgt[tt, ii, jj, :] = griddata(z_src[tt, ii, jj, :], values_src[tt, ii, jj, :], 
                    z_tgt, method = method, fill_value = fill_value, rescale = rescale)
            
    return values_tgt

def rbfinterpolator_z(z_src: np.ndarray, values_src: np.ndarray, neighbors: Optional[int] = None, 
                      smoothing: float = 0.0, kernel: str = "thin_plate_spline", 
                      epsilon: Optional[float] = None, degree: Optional[int] = None) -> np.ndarray:
    ## z_src is the time-dependent, spatially-varying vertical grid of values_src
    ntime: int
    nx: int
    ny: int

    ntime, nx, ny = values.shape[0:3]
    nz: int = z_tgt.shape[0]

    interp: np.ndarray = np.empty((ntime, nx, ny), dtype = "O")

    for tt in range(0, ntime):
        for ii in range(0, nx):
            for jj in range(0, ny):
                interp[tt, ii, jj] = \
                    RBFInterpolator(np.expand_dims(z_src[tt, ii, jj, :], axis = 1),
                        values_src[tt, ii, jj, :], neighbors = neighbors,
                        smoothing = smoothing, kernel = kernel, epsilon = epsilon, degree = degree)
            
    return interp

def eval_rbfinterpolator_z(z_src: np.ndarray, interp: np.ndarray, z_tgt: np.ndarray) -> np.ndarray:
    ## z_src is the time-dependent, spatially-varying vertical grid of values_src
    ## z_tgt is the target vertical grid, is time- and space-independent
    ntime: int
    nx: int
    ny: int

    ntime, nx, ny = values.shape[0:3]
    nz: int = z_tgt.shape[0]

    values_tgt: np.ndarray = np.empty((ntime, nx, ny, nz), dtype = np_float)

    for tt in range(0, ntime):
        for ii in range(0, nx):
            for jj in range(0, ny):
                values_tgt[tt, ii, jj, :] = interp[tt,ii,jj](np.expand_dims(z_tgt, axis = 1))

    return values_tgt

if __name__ == "__main__":
    main()

