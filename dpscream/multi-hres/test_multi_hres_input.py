"""
Converts output from DP-SCREAM at multiple horizontal resolutions into input for
RTE-RRTMGP-CPP.

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
    
    parser.add_argument("--szas",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
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
    szas: np.ndarray = np.array(ast.literal_eval(args.szas[0]), dtype = np_float).flatten()
    output_file_root_path: str = os.path.normpath(args.output_root[0])

    ## Read the DP-SCREAM output file
    input_file_path: str = input_file_root_path + ".nc"
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)

    ## Select a time-step if there are multiple
    time: np.ma.MaskedArray = nc_input.variables["time"][:].astype(np_float) # Days since given date; (time)
    ntime: int = time.size
    if ntime > 0:
        time_idx: Optional[int] = -1
    else:
        time_idx: Optional[idx] = None

    ## Reconstruct the horizontal grid
    lon: np.ma.MaskedArray = nc_input.variables["lon"][:].astype(np_float) # Column-center - x-dimension [m]; (ncol)
    lat: np.ma.MaskedArray = nc_input.variables["lat"][:].astype(np_float) # Column center - y-dimension [m]; (ncol)

    sort_mask: np.ndarray = np.lexsort((lon, lat)) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    n_col_x: int = np.unique(lon).size # No. columns in x
    n_col_y: int = np.unique(lat).size # No. columns in y
    cols: np.ma.MaskedArray = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(n_col_x, n_col_y, 2)

    ## Store spatial grid for outputting to RTE-RRTMGP-CPP input file
    grid: dict = {}

    ### NOTE: The number of points in the horizontal acceleration grid "should"
    ### be between 1/10 and 1/20 of n_col_x, n_col_y.
    ngrid_x: int = np.ceil(n_col_x / 10)
    ngrid_y: int = np.ceil(n_col_y / 10)

    grid["ngrid_x"] = ngrid_x
    grid["ngrid_y"] = ngrid_y
    
    ### NOTE: The names xh, yh seem to refer to the interfaces between columns,
    ### but in the original rcemip experiment, they just tack on an extra value.
    ### They don't seem to be directly used in the code, so we will use them
    ### to be interfaces between columns.
    ### NOTE: Assume that horizontal grids are regularly spaced.
    x: np.ma.MaskedArray = (cols[:,:,0])[0,:] # x-midpoints of each column [m]; (n_col_x)
    dx: np_float = x[1] - x[0]
    xh: np.ma.MaskedArray = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-interfaces of each column [m]; (n_col_x + 1)
    grid["x"] = x
    grid["xh"] = xh

    y: np.ma.MaskedArray = (cols[:,:,1])[:,0] # x-midpoints of each column [m]; (n_col_y)
    dy: np_float = y[1] - y[0]
    yh: np.ma.MaskedArray = np.append(y - (dy / 2.), x[-1] + (dy / 2.)) # y-interfaces of each column [m]; (n_col_y + 1)
    grid["y"] = y
    grid["yh"] = yh

    ## Reconstruct the veritcal grids
    z_mid: np.ma.MaskedArray = nc_input.variables["z_mid"][:].astype(np_float) # Level midpoints [m]; (time, ncol, n_lay_z)
    z_int: np.ma.MaskedArray = nc_input.variables["z_int"][:].astype(np_float) # Level interfaces [m]; (time, ncol, n_lev_z)

    if time_idx is not None:
        z_mid: np.ma.MaskedArray = z_mid[time_idx,...] # (ncol, n_lay_z)
        z_int: np.ma.MaskedArray = z_int[time_idx,...] # (ncol, n_lev_z)
    else:
        z_mid: np.ma.MaskedArray = np.squeeze(z_mid, axis = 0) # (ncol, n_lay_z)
        z_int: np.ma.MaskedArray = np.squeeze(z_int, axis = 0) # (ncol, n_lev_z)

    n_lay_z: int = z_mid.shape[1]
    n_lev_z: int = z_int.shape[1]
    assert(n_lev_z == n_lay_z + 1)

    z_mid: np.ma.MaskedArray = z_mid[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # Layer midpoints [m]; (n_col_x, n_col_y, n_lay_z)
    z_int: np.ma.MaskedArray = z_int[sort_mask,:].reshape(n_col_x, n_col_y, n_lev_z) # Layer interfaces [m]; (n_col_x, n_col_y, n_lev_z)
    
    ### Interleave these into a single vertical grid
    n_z: int = n_lay_z + n_lev_z
    z: np.ndarray = np.empty([n_col_x, n_col_y, n_z], dtype = z_mid.dtype)
    z[:,:,1::2] = z_mid
    z[:,:,0::2] = z_int

    ### NOTE: The number of points in the vertical acceleration grid "should"
    ### be between 1/10 and 1/20 of n_lay_z.
    ngrid_z: int = np.ceil(n_lay_z / 10)

    grid["ngrid_z"] = ngrid_z

    ### Create the scattered point array for interpolating variables from
    XYZ_mid: np.ma.MaskedArray = np.concatenate((np.tile(np.expand_dims(cols, axis = 2), (1, 1, n_lay_z, 1)), np.expand_dims(z_mid, axis = 3)), axis = 3) # (n_col_x, n_col_y, n_lay_z, 3)
    XYZ_mid: np.ma.MaskedArray = XYZ_mid.reshape(n_col_x * n_col_y * n_lay_z, 3)

    XYZ_int: np.ma.MaskedArray = np.concatenate((np.tile(np.expand_dims(cols, axis = 2), (1, 1, n_lev_z, 1)), np.expand_dims(z_int, axis = 3)), axis = 3) # (n_col_x, n_col_y, n_lev_z, 3)
    XYZ_int: np.ma.MaskedArray = XYZ_int.reshape(n_col_x * n_col_y * n_lev_z, 3)

    ### Create a regularly-spaced grid for interpolating variables to.
    z_min: np_float = 0.0
    z_max: np_float = z_int.max()

    z_lev: np.ndarray = np.linspace(z_min, z_max, n_lev_z) # Regularly-spaced layer interfaces [m]; (n_lev_z)
    z_lay: np.ndarray = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced layer midpoints [m]; (n_lay_z)

    grid["z"] = z_lay
    grid["zh"] = z_lev
    grid["z_lay"] = z_lay
    grid["z_lev"] = z_lev

    ZZ_lev: np.ndarray
    _, _, ZZ_lev = np.meshgrid(x, y, z_lev, indexing = "ij")

    XX_lay: np.nadarray
    YY_lay: np.ndarray
    ZZ_lay: np.ndarray
    XX_lay, YY_lay, ZZ_lay = np.meshgrid(x, y, z_lay, indexing = "ij")

    method: str = "rbf"
    assert(method in ["nearest", "rbf"])
    epsilon: float = 1.0

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

    for ii in range(len(dpscream_field_keys)):
        dpscream_field_key: str = dpscream_field_keys[ii]
        rte_field_key: str = rte_field_keys[ii]
        ## Fields either have layer midpoint and interface values with an 
        ## additional string in their key, or just layer midpoint values that are
        ## just the field key
        if dpscream_field_key in nc_input.variables.keys():
            field_mid: np.ma.MaskedArray = nc_input.variables[dpscream_field_key][:].astype(np_float) # Field at layer midpoints; (time, ncol, n_lay_z)
            field_int: Optional[np.ma.MaskedArray] = None # Field at layer interfaces; (time, ncol, n_lay_z)

            z_field: np.ndarray = z_mid
        else:
            dpscream_field_key_mid: str = dpscream_field_key + "_mid"
            ## Exceptions
            if dpscream_field_key in ["T"]:
                dpscream_field_key_int: str = dpscream_field_key + "_int_rad"
            else:
                dpscream_field_key_int: str = dpscream_field_key + "_int"

            ## We should alywas have fields values at layer midpoints
            ## Unless we don't, then this needs to be fixed
            assert(dpscream_field_key_mid in nc_input.variables.keys())
            field_mid: np.ma.MaskedArray = nc_input.variables[dpscream_field_key_mid][:].astype(np_float) # Field at layer midpoints; (time, ncol, n_lay_z)
            
            if dpscream_field_key_int in nc_input.variables.keys():
                field_int: Optional[np.ma.MaskedArray] = nc_input.variables[dpscream_field_key_int][:].astype(np_float) # Field at layer interfaces; (time, ncol, n_lay_z)
                z_field: np.ndarray = z
            else:
                field_int: Optional[np.ma.MaskedArray] = None # Field at layer interfaces; (time, ncol, n_lay_z)
                z_field: np.ndarray = z_mid

        ## Specify a given time-step, if the data file has multiple
        if time_idx is not None:
            field_mid: np.ma.MaskedArray = field_mid[time_idx,...] # (ncol, n_lay_z)
            if field_int is not None:
                field_int: np.ma.MaskedArray = field_int[time_idx,...] # (ncol, n_lev_z)
        else:
            field_mid: np.ma.MaskedArray = np.squeeze(field_mid, axis = 0) # (ncol, n_lay_z)
            if field_int is not None:
                field_int: np.ma.MaskedArray = np.squeeze(field_int, axis = 0) # (ncol, n_lev_z)

        ## Reshape into x- and y-columns
        field_mid: np.ma.MaskedArray = field_mid[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)
        if field_int is not None:
            field_int: np.ma.MaskedArray = field_int[sort_mask,:].reshape(n_col_x, n_col_y, n_lev_z) # (n_col_x, n_col_y, n_lev_z)

        ## Adjust the masked value for volume mixing ratios
        if dpscream_field_key[-16:] == "volume_mix_ratio":
            field_mid.fill_value: float = 0.
            if field_int is not None:
                field_int.fill_value: float = 0.

        ## If we have field values at layer midpoints and interfaces, interleave them
        if field_int is not None:
            field: np.ndarray = np.empty([n_col_x, n_col_y, n_z], dtype = field_mid.dtype)
            field[:,:,1::2] = field_mid.filled()
            field[:,:,0::2] = field_int.filled()
        else:
            field: np.ndarray = field_mid.filled()

        ## Exceptions
        if rte_field_key in ["dei"]: # DP-SCREAM has rei, RTE-RRTMGP-CPP has dei
            field: np.ndarray = 2. * field
        elif rte_field_key in ["lwp", "iwp"]: # Derived from multiple quantities
            p_int: np.ma.MaskedArray = nc_input.variables["p_int"][:].astype(np_float) # Pressure at layer interfaces [Pa]; (time, ncol, n_lev_z)
            if time_idx is not None:
                p_int: np.ma.MaskedArray = p_int[time_idx,...] # (ncol, n_lev_z)
            else:
                p_int: np.ma.MaskedArray = np.squeeze(p_int, axis = 0) # (ncol, n_lev_z)
            p_int: np.ma.MaskedArray = p_int[sort_mask,:].reshape(n_col_x, n_col_y, n_lev_z) # (n_col_x, n_col_y, n_lev_z)
            dp: np.ma.MaskedArray = p_int[:,:,1:] - p_int[:,:,:-1] # Layer pressure thickness [Pa]; (n_col_x, n_col_y, n_lay_z)

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
        if ((method == "nearest") or (rte_field_key == "vmr_h2o")):
            field_lay: np.ndarray = np.transpose(remap_z(z_field, field, ZZ_lay), axes = (2, 1, 0)) # Field at regular layer midpoints; (n_lay_z, n_col_y, n_col_x)
            field_lev: np.ndarray = np.transpose(remap_z(z_field, field, ZZ_lev), axes = (2, 1, 0)) # Field at regular layer interfaces; (n_lev_z, n_col_y, n_col_x)
        elif method == "rbf":
            field_rbfinterpolator_z: np.ndarray = rbfinterpolator_z(z_field, field, epsilon = epsilon) # Interpolant of field in each column
            field_lay: np.ndarray = np.transpose(eval_rbfinterpolator_z(field_rbfinterpolator_z, ZZ_lay), axes = (2, 1, 0)) # Field at regular layer midpoints; (n_lay_z, n_col_y, n_col_x)
            field_lev: np.ndarray = np.transpose(eval_rbfinterpolator_z(field_rbfinterpolator_z, ZZ_lev), axes = (2, 1, 0)) # Field at regular layer interfaces; (n_lev_z, n_col_y, n_col_x)

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
    dpscream_field_keys: list = ["surf_radiative_T"]
    rte_field_keys: list = ["t_sfc"]

    for ii in range(len(dpscream_field_keys)):
        dpscream_field_key: str = dpscream_field_keys[ii]
        rte_field_key: str = rte_field_keys[ii]
        assert(dpscream_field_key in nc_input.variables.keys())

        field: np.ma.MaskedArray = nc_input.variables[dpscream_field_key][:].astype(np_float) # 2-D field; (time, ncol)

        ## Specify a given time-step, if the data file has multiple
        if time_idx is not None:
            field: np.ma.MaskedArray = field[time_idx,...] # (ncol)
        else:
            field: np.ma.MaskedArray = np.squeeze(field, axis = 0) # (ncol)

        ## Reshape into x- and y-columns
        field: np.ma.MaskedArray = field[sort_mask].reshape(n_col_x, n_col_y) # (n_col_x, n_col_y)
        field: np.ndarray = np.transpose(np.ma.getdata(field), axes = (1, 0))

        fields[rte_field_key] = field

    ## Special fields specified in the DP-SCREAM output
    ## Obtain the number of shortwave and longwave bands.
    swband: np.ma.MaskedArray = nc_input.variables["swband"][:].astype(np_float) # Shortwave bands [cm^(-1)]; (n_bnd_sw)
    lwband: np.ma.MaskedArray = nc_input.variables["lwband"][:].astype(np_float) # Longwave bands [cm^(-1)]; (n_bnd_lw)

    n_bnd_sw: int = swband.size
    n_bnd_lw: int = lwband.size

    ## Fields that are not specified in the DP-SCREAM output
    ## Longwave boundary conditions
    fields["emis_sfc"]: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_lw)) # Surface emissivity [N/A]

    ## Shortwave boundary conditions
    fields["sfc_alb_dir"]: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Direct
    fields["sfc_alb_dif"]: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Diffuse

    fields["tsi"]: float = 551.58 # Total Solar Irradiance [W m^(-2)]

    fields["azi"]: float = 0.0 # Azimuthal Angle [Radians]

    ## Set quantities not expected to be set in the DP-SCREAM output
    ### Gas volume mixing ratios
    fields["vmr_ccl4"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))     # Carbon Tetrachloride [kg kg^(-1)]
    fields["vmr_cfc11"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))    # Trichlorofluoromethane (CFC-11) [kg kg^(-1)]
    fields["vmr_cfc12"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))    # Dichlorodifluoromethane (CFC-12) [kg kg^(-1)]
    fields["vmr_cfc22"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))    # Chlorodifluoromethane (HCFC-22) [kg kg^(-1)]
    fields["vmr_hfc143a"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))  # 1,1,1-Trifluoroethane (HFC-143a) [kg kg^(-1)]
    fields["vmr_hfc125"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))   # Pentafluoroethane (HFC-125) [kg kg^(-1)]
    fields["vmr_hfc32"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))    # Difluoromethane (HFC-32) [kg kg^(-1)]
    fields["vmr_hfc23"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))    # Trifluoromethane (HFC-23) [kg kg^(-1)]
    fields["vmr_hfc134a"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))  # 1,1,1,2-Tetrafluoroethane (HFC-134a) [kg kg^(-1)]
    fields["vmr_cf4"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))      # Carbon Tetrafluoride (CF₄) [kg kg^(-1)]
    fields["vmr_no2"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x))      # Nitrogen Dioxide [kg kg^(-1)]

    ### Aerosol mixing ratios
    fields["aermr01"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.03 - 0.5 µm)
    fields["aermr02"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.5 - 5 µm)
    fields["aermr03"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (5 - 20 µm)
    fields["aermr04"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.03 - 0.55 µm)
    fields["aermr05"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.55 - 0.9 µm)
    fields["aermr06"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.9 - 20 µm)
    fields["aermr07"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophilic Organic Matter Aerosol
    fields["aermr08"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophobic Organic Matter Aerosol
    fields["aermr09"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophilic Black Carbon Aerosol
    fields["aermr10"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophobic Black Carbon Aerosol
    fields["aermr11"]: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sulfate Aerosol

    ## Write to RTE-RRTMGP-CPP input file, with the varying given solar zenith angles
    nc_float: str = "f8"
    for sza in szas:
        sza_rad: np.float64 = np.deg2rad(sza)
        fields["mu0"]: np_float = np.zeros((n_col_y, n_col_x)) + np.cos(sza_rad) # Cosine of SZA

        sza_str: str = "{:04.0f}".format(sza)
        output_file_path: str = output_file_root_path + "." + sza_str + ".in.nc"

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

        for rte_field_key in fields_dimensions:
            field: np.ndarray = fields[rte_field_key]
            field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
            field_description: str = fields_descriptions[rte_field_key]
            field_units: str = fields_units[rte_field_key]

            nc_field: nc._netCDF4.Variable = nc_file.createVariable(rte_field_key, nc_float, field_dimensions)
            nc_field.description: str = field_description
            nc_field.units: str = field_units
            nc_field[...]: np.ndarray = field

        nc_file.close()

def remap_z(ZZ_src: tuple, values_src: np.ndarray, ZZ_tgt: tuple, 
            method: str = "nearest", fill_value: float = np.nan, rescale: bool = False):
    
    ## ZZ_* is the third entry in the meshgrid
    src_shape: tuple = ZZ_src.shape
    tgt_shape: tuple = ZZ_tgt.shape

    assert(src_shape[0:2] == tgt_shape[0:2])

    nx: int = src_shape[0]
    ny: int = src_shape[1]
    ntgt: int = tgt_shape[2]

    values_tgt: np.ndarray = np.zeros((nx, ny, ntgt), dtype = values_src.dtype)

    for ii in range(0, nx):
        for jj in range(0, ny):
            values_tgt[ii, jj, :] = griddata(ZZ_src[ii, jj, :], values_src[ii, jj, :], 
                ZZ_tgt[ii, jj, :], method = method, fill_value = fill_value, rescale = rescale)
            
    return values_tgt

def rbfinterpolator_z(y: np.ndarray, d: np.ndarray, neighbors: Optional[int] = None, 
                      smoothing: float = 0.0, kernel: str = "thin_plate_spline", 
                      epsilon: Optional[float] = None, degree: Optional[int] = None) -> np.ndarray:
    ## y is the third entry in the meshgrid
    y_shape: tuple = y.shape
    nx: int
    ny: int
    nx, ny = y_shape[0:2]

    interp: np.ndarray = np.empty((nx, ny), dtype = "O")

    for ii in range(0, nx):
        for jj in range(0, ny):
            interp[ii, jj] = \
                RBFInterpolator(np.expand_dims(y[ii, jj, :], axis = 1), d[ii, jj, :], 
                neighbors = neighbors, smoothing = smoothing, kernel = kernel,
                epsilon = epsilon, degree = degree)
            
    return interp

def eval_rbfinterpolator_z(interp: np.ndarray, y: np.ndarray) -> np.ndarray:
    ## y is the third entry in the meshgrid
    y_shape: tuple = y.shape
    nx: int
    ny: int
    nz: int
    nx, ny, nz = y_shape

    values_tgt: np.ndarray = np.empty((nx, ny, nz), dtype = np_float)

    for ii in range(0, nx):
        for jj in range(0, ny):
            values_tgt[ii, jj, :] = interp[ii,jj](np.expand_dims(y[ii,jj,:], axis = 1))

    return values_tgt

if __name__ == "__main__":
    main()

