"""
Generates a NetCDF file that converts output from DP-SCREAM into input for 
RTE-RRTMGP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import os

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata

# Local Library Imports
from consts import R_d, R_v, L_v, g, p_0, np_float

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "convert_dp_scream_output",
        description = "Creates output from DP-SCREAM to input to RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Path to DP-SCREAM output.")
    
    parser.add_argument("--output",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = False,
                        default = ["rte_rrtmgp_input.nc"],
                        help = "Path to RTE-RRTMGP-CPP input file.")
    
    args: argparse.Namespace = parser.parse_args()

    input_file_path: str = os.path.normpath(args.input[0])
    output_file_path: str = os.path.normpath(args.output[0])

    ## Read the DP-SCREAM output file
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)

    ## Reconstruct the horizontal grid
    lon: np.ma.MaskedArray = nc_input.variables["lon"][:].astype(np_float) # Column-center - x-dimension [m]; (ncol)
    lat: np.ma.MaskedArray = nc_input.variables["lat"][:].astype(np_float) # Column center - y-dimension [m]; (ncol)

    sort_mask: np.ndarray = np.lexsort((lon, lat)) # Mask that sorts arrays for restructuring into 1-D x- and y-grids

    n_col_x: int = np.unique(lon).size # No. columns in x
    n_col_y: int = np.unique(lat).size # No. columns in y
    cols: np.ma.MaskedArray = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(n_col_x, n_col_y, 2)

    ### NOTE: The number of points in the horizontal acceleration grid "should"
    ### be between 1/10 and 1/20 of n_col_x, n_col_y.
    ngrid_x: int = np.ceil(n_col_x / 10)
    ngrid_y: int = np.ceil(n_col_y / 10)
    
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

    ## Reconstruct the veritcal grids
    z_mid: np.ma.MaskedArray = np.squeeze(nc_input.variables["z_mid"][:].astype(np_float), axis = 0) # Level midpoints [m]; (ncol, n_lay_z)
    z_int: np.ma.MaskedArray = np.squeeze(nc_input.variables["z_int"][:].astype(np_float), axis = 0) # Level interfaces [m]; (ncol, n_lev_z)

    n_lay_z: int = z_mid.shape[1]
    n_lev_z: int = z_int.shape[1]
    assert(n_lev_z == n_lay_z + 1)

    z_mid: np.ma.MaskedArray = z_mid[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # Layer midpoints [m]; (n_col_x, n_col_y, n_lay_z)
    z_int: np.ma.MaskedArray = z_int[sort_mask,:].reshape(n_col_x, n_col_y, n_lev_z) # Layer interfaces [m]; (n_col_x, n_col_y, n_lev_z)

    ### NOTE: The number of points in the vertical acceleration grid "should"
    ### be between 1/10 and 1/20 of n_lay_z.
    ngrid_z: int = np.ceil(n_lay_z / 10)

    ### Create the scattered point array for interpolating variables from
    XYZ_mid: np.ma.MaskedArray = np.concatenate((np.tile(np.expand_dims(cols, axis = 2), (1, 1, n_lay_z, 1)), np.expand_dims(z_mid, axis = 3)), axis = 3) # (n_col_x, n_col_y, n_lay_z, 3)
    XYZ_mid: np.ma.MaskedArray = XYZ_mid.reshape(n_col_x * n_col_y * n_lay_z, 3)
    XYZ_int: np.ma.MaskedArray = np.concatenate((np.tile(np.expand_dims(cols, axis = 2), (1, 1, n_lev_z, 1)), np.expand_dims(z_int, axis = 3)), axis = 3) # (n_col_x, n_col_y, n_lev_z, 3)
    XYZ_int: np.ma.MaskedArray = XYZ_int.reshape(n_col_x * n_col_y * n_lev_z, 3)

    ### Create a regularly-spaced grid for interpolating variables to.
    z_min: np_float = z_int.min()
    z_max: np_float = z_int.max()

    z_lev: np.ndarray = np.linspace(z_min, z_max, n_lev_z) # Regularly-spaced layer interfaces [m]; (n_lev_z)
    z_lay: np.ndarray = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced layer midpoints [m]; (n_lay_z)

    XX_lev: np.ndarray
    YY_lev: np.ndarray
    ZZ_lev: np.ndarray
    XX_lev, YY_lev, ZZ_lev = np.meshgrid(x, y, z_lev, indexing = "ij")

    XX_lay: np.ndarray
    YY_lay: np.ndarray
    ZZ_lay: np.ndarray
    XX_lay, YY_lay, ZZ_lay = np.meshgrid(x, y, z_lay, indexing = "ij")

    interp_method: str = "nearest"

    ## Interpolate the pressure to the regularly-spaced grid.
    ### NOTE: We should interweave p_mid and p_int and interpolate that to 
    ### the regularly-spaced grid. As a first pass, we do these individually.
    ### NOTE: The surface pressure is stored in a different variable, but that
    ### would add an extra level relative to the other state variables.
    p_mid: np.ma.MaskedArray = np.squeeze(nc_input.variables["p_mid"][:].astype(np_float), axis = 0) # Pressure at layer midpoints [Pa]; (ncol, n_lay_z)
    p_mid: np.ma.MaskedArray = p_mid[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)
    
    p_int: np.ma.MaskedArray = np.squeeze(nc_input.variables["p_int"][:].astype(np_float), axis = 0) # Pressure at layer interfaces [Pa]; (ncol, n_lev_z)
    p_int: np.ma.MaskedArray = p_int[sort_mask,:].reshape(n_col_x, n_col_y, n_lev_z) # (n_col_x, n_col_y, n_lev_z)

    p_lay: np.ndarray = griddata(XYZ_mid, p_mid.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Pressure at regular layer midpoints [Pa]; (n_col_x, n_col_y, n_lay_z)
    p_lay: np.ndarray = np.transpose(p_lay, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    p_lev: np.ndarray = griddata(XYZ_int, p_int.reshape(n_col_x * n_col_y * n_lev_z), 
                                 (XX_lev, YY_lev, ZZ_lev), method = interp_method) # Pressure at regular layer interfaces [Pa]; (n_col_x, n_col_y, n_lev_z)
    p_lev: np.ndarray = np.transpose(p_lev, axes = (2, 1, 0)) # (n_lev_z, n_col_y, n_col_x)
    
    ## Interpolate the temperature to the regularly-spaced grid.
    ### NOTE: We don't seem to have T_int, so we interpolate T_mid to T_lev.
    T_mid: np.ma.MaskedArray = np.squeeze(nc_input.variables["T_mid"][:].astype(np_float), axis = 0) # Temperature at layer midpoints [K]; (ncol, n_lay_z)
    T_mid: np.ma.MaskedArray = T_mid[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    T_lay: np.ndarray = griddata(XYZ_mid, T_mid.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Temperature at regular layer midpoints [K]; (n_col_x, n_col_y, n_lay_z)
    T_lay: np.ndarray = np.transpose(T_lay, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    T_lev: np.ndarray = griddata(XYZ_mid, T_mid.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lev, YY_lev, ZZ_lev), method = interp_method) # Temperature at regular layer interfaces [K]; (n_col_x, n_col_y, n_lev_z)
    T_lev: np.ndarray = np.transpose(T_lev, axes = (2, 1, 0)) # (n_lev_z, n_col_y, n_col_x)

    ## Interpolate the relative humidity to the regularly-spaced grid.
    RelativeHumidity: np.ma.MaskedArray = np.squeeze(nc_input.variables["RelativeHumidity"][:].astype(np_float), axis = 0) # Relative humidity at layer midpoints [N/A]; (ncol, n_lay_z)
    RelativeHumidity: np.ma.MaskedArray = RelativeHumidity[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    rh_lay: np.ndarray = griddata(XYZ_mid, RelativeHumidity.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Relative humidity at regular layer midpoints; (n_col_x, n_col_y, n_lay_z)
    rh_lay: np.ma.MaskedArray = np.transpose(rh_lay, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    ## Interpolate specific humidity to the regularly-spaced grid.
    ### NOTE: I assume that "humidity mixing ratio" is the same as "specific humidity"
    qv: np.ma.MaskedArray = np.squeeze(nc_input.variables["qv"][:].astype(np_float), axis = 0) # Specific humidity at layer midpoints [N/A]; (ncol, n_lay_z)
    qv: np.ma.MaskedArray = qv[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    q_lay: np.ndarray = griddata(XYZ_mid, qv.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Specific humidity at regular layer midpoints; (n_col_x, n_col_y, n_lay_z)
    q_lay: np.ma.MaskedArray = np.transpose(q_lay, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    ## Obtain the number of shortwave and longwave bands.
    swband: np.ma.MaskedArray = nc_input.variables["swband"][:].astype(np_float) # Shortwave bands [cm^(-1)]; (n_bnd_sw)
    lwband: np.ma.MaskedArray = nc_input.variables["lwband"][:].astype(np_float) # Longwave bands [cm^(-1)]; (n_bnd_lw)

    n_bnd_sw: int = swband.size
    n_bnd_lw: int = lwband.size

    ## Calculate the lwp, iwp on the regularly-spaced grid.
    qc: np.ma.MaskedArray = np.squeeze(nc_input.variables["qc"][:].astype(np_float), axis = 0) # Cloud liquid water mixing ratio at layer midpoints [N/A]; (ncol, n_lay_z)
    qc: np.ma.MaskedArray = qc[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    qi: np.ma.MaskedArray = np.squeeze(nc_input.variables["qi"][:].astype(np_float), axis = 0) # Cloud ice water mixing ratio at layer midpoints [N/A]; (ncol, n_lay_z)
    qi: np.ma.MaskedArray = qi[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    dp: np.ma.MaskedArray = p_int[:,:,1:] - p_int[:,:,:-1] # Layer pressure thickness [Pa]; (n_col_x, n_col_y, n_lay_z)

    lwp_mid: np.ma.MaskedArray = qc * dp / g # Liquid water path [kg m^(-2)]; (n_col_x, n_col_y, n_lay_z)
    lwp: np.ndarray = griddata(XYZ_mid, lwp_mid.reshape(n_col_x * n_col_y * n_lay_z), 
                                   (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Liquid water path at regular layer midpoints; (n_col_x, n_col_y, n_lay_z)
    lwp: np.ma.MaskedArray = np.transpose(lwp, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    iwp_mid: np.ma.MaskedArray = qi * dp / g # Ice water path [kg m^(-2)]; (n_col_x, n_col_y, n_lay_z)
    iwp: np.ndarray = griddata(XYZ_mid, iwp_mid.reshape(n_col_x * n_col_y * n_lay_z), 
                                   (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Ice water path at regular layer midpoints; (n_col_x, n_col_y, n_lay_z)
    iwp: np.ma.MaskedArray = np.transpose(iwp, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    ## Interpolate effective liquid water radius to the regularly-spaced grid.
    eff_radius_qr: np.ma.MaskedArray = np.squeeze(nc_input.variables["eff_radius_qr"][:].astype(np_float), axis = 0) # Effective raidus of cloud rain particles [μm]; (ncol, n_lay_z)
    eff_radius_qr: np.ma.MaskedArray = eff_radius_qr[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    rel: np.ndarray = griddata(XYZ_mid, eff_radius_qr.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Liquid water effective radius at regular layer midpoints [μm]; (n_col_x, n_col_y, n_lay_z)
    rel: np.ma.MaskedArray = np.transpose(rel, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    ## Interpolate effective ice water diameter to the regularly-spaced grid.
    eff_radius_qi: np.ma.MaskedArray = np.squeeze(nc_input.variables["eff_radius_qi"][:].astype(np_float), axis = 0) # Effective raidus of cloud ice particles [μm]; (ncol, n_lay_z)
    eff_radius_qi: np.ma.MaskedArray = eff_radius_qi[sort_mask,:].reshape(n_col_x, n_col_y, n_lay_z) # (n_col_x, n_col_y, n_lay_z)

    dei: np.ndarray = griddata(XYZ_mid, 2. * eff_radius_qi.reshape(n_col_x * n_col_y * n_lay_z), 
                                 (XX_lay, YY_lay, ZZ_lay), method = interp_method) # Ice water effective diameter at regular layer midpoints [μm]; (n_col_x, n_col_y, n_lay_z)
    dei: np.ma.MaskedArray = np.transpose(dei, axes = (2, 1, 0)) # (n_lay_z, n_col_y, n_col_x)

    ## Convert water from specific humidity (q) to volume mixing ratio (vmr)
    h2o: np.ndarray = q_lay / ((R_d / R_v) * (1. - q_lay)) # [kg kg^(-1)]

    ## Set other VMRs
    co2: float = 348.e-6  # Carbon Dioxide [kg kg^(-1)]
    ch4: float = 1650.e-9 # Methane [kg kg^(-1)]
    n2o: float = 306.e-9  # Nitrous Oxide [kg kg^(-1)]
    n2: float = 0.7808    # Nitrogen [kg kg^(-1)]
    o2: float = 0.2095    # Oxygen [kg kg^(-1)]
    co: float = 0.0       # Carbon Monoxide [kg kg^(-1)]
    ccl4: float = 0.0     # Carbon Tetrachloride [kg kg^(-1)]
    cfc11: float = 0.0    # Trichlorofluoromethane (CFC-11) [kg kg^(-1)]
    cfc12: float = 0.0    # Dichlorodifluoromethane (CFC-12) [kg kg^(-1)]
    cfc22: float = 0.0    # Chlorodifluoromethane (HCFC-22) [kg kg^(-1)]
    hfc143a: float = 0.0  # 1,1,1-Trifluoroethane (HFC-143a) [kg kg^(-1)]
    hfc125: float = 0.0   # Pentafluoroethane (HFC-125) [kg kg^(-1)]
    hfc32: float = 0.0    # Difluoromethane (HFC-32) [kg kg^(-1)]
    hfc23: float = 0.0    # Trifluoromethane (HFC-23) [kg kg^(-1)]
    hfc134a: float = 0.0  # 1,1,1,2-Tetrafluoroethane (HFC-134a) [kg kg^(-1)]
    cf4: float = 0.0      # Carbon Tetrafluoride (CF₄) [kg kg^(-1)]
    no2: float = 0.0      # Nitrogen Dioxide [kg kg^(-1)]

    ## Set O3 VMR
    g1: float = 3.6478
    g2: float = 0.83209
    g3: float = 11.3515
    p_hpa: np.ndarray = p_lay / 100. # [hPa]

    o3_min: float = 1.e-13 # RRTMGP in Single Precision will fail with lower ozone concentrations
    o3: np.ndarray = np.maximum(o3_min, g1 * p_hpa**g2 * np.exp(-p_hpa / g3) * 1.e-6)

    ## Aerosol mixing ratios
    aermr01: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.03 - 0.5 µm)
    aermr02: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (0.5 - 5 µm)
    aermr03: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sea salt aerosol (5 - 20 µm)
    aermr04: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.03 - 0.55 µm)
    aermr05: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.55 - 0.9 µm)
    aermr06: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Dust aerosol (0.9 - 20 µm)
    aermr07: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophilic Organic Matter Aerosol
    aermr08: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophobic Organic Matter Aerosol
    aermr09: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophilic Black Carbon Aerosol
    aermr10: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Hydrophobic Black Carbon Aerosol
    aermr11: np.ndarray = np.zeros((n_lay_z, n_col_y, n_col_x)) # Sulfate Aerosol

    ## Longwave boundary conditions
    emis_sfc: float = 1. # Surface emissivity [N/A]
    t_sfc: float = 300.  # Surface temperature [K]

    ## Shortwave boundary conditions
    solar_zenith_angle: np.float64 = np.deg2rad(60.00) # [N/A]
    mu0: np.float64 = np.cos(solar_zenith_angle) # [N/A]

    sfc_alb_dir: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Direct
    sfc_alb_dif: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Diffuse

    total_solar_irradiance: float = 551.58 # [W m^(-2)]

    azi: float = 1.834 # Azimuthal Angle [Radians]

    ## Write input to file
    nc_float: str = "f8"
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
    nc_ngrid_x: nc._netCDF4.Variable = nc_file.createVariable("ngrid_x", nc_float)
    nc_ngrid_x.description = "No. 'acceleration grid' points in the x-direction"
    nc_ngrid_x[:] = ngrid_x

    nc_ngrid_y: nc._netCDF4.Variable = nc_file.createVariable("ngrid_y", nc_float)
    nc_ngrid_y.description = "No. 'acceleration grid' points in the y-direction"
    nc_ngrid_y[:] = ngrid_y

    nc_ngrid_z: nc._netCDF4.Variable = nc_file.createVariable("ngrid_z", nc_float)
    nc_ngrid_z.description = "No. 'acceleration grid' points in the z-direction"
    nc_ngrid_z[:] = ngrid_z

    nc_x: nc._netCDF4.Variable = nc_file.createVariable("x", nc_float, "x")
    nc_x.description = "x-dimension - column midpoints"
    nc_x.units = "m"
    nc_x[:] = x

    nc_y: nc._netCDF4.Variable = nc_file.createVariable("y", nc_float, "y")
    nc_y.description = "y-dimension - column midpoints"
    nc_y.units = "m"
    nc_y[:] = y

    nc_z: nc._netCDF4.Variable = nc_file.createVariable("z", nc_float, "z")
    nc_z.description = "z-dimension - layer midpoints"
    nc_z.units = "m"
    nc_z[:] = z_lay

    nc_xh: nc._netCDF4.Variable = nc_file.createVariable("xh", nc_float, "xh")
    nc_xh.description = "x-dimension - column interfaces"
    nc_xh.units = "m"
    nc_xh[:] = xh

    nc_yh: nc._netCDF4.Variable = nc_file.createVariable("yh", nc_float, "yh")
    nc_yh.description = "y-dimension - column interfaces"
    nc_yh.units = "m"
    nc_yh[:] = yh

    nc_zh: nc._netCDF4.Variable = nc_file.createVariable("zh", nc_float, "zh")
    nc_zh.description = "z-dimension - layer interfaces"
    nc_zh.units = "m"
    nc_zh[:] = z_lev

    nc_z_lay: nc._netCDF4.Variable = nc_file.createVariable("z_lay", nc_float, ("lay"))
    nc_z_lay.description = "z-dimension - layer midpoints"
    nc_z_lay.units = "m"
    nc_z_lay[:] = z_lay

    nc_z_lev: nc._netCDF4.Variable = nc_file.createVariable("z_lev", nc_float, ("lev"))
    nc_z_lev.description = "z-dimension - layer interfaces"
    nc_z_lev.units = "m"
    nc_z_lev[:] = z_lev

    ## Pressure
    nc_p_lay: nc._netCDF4.Variable = nc_file.createVariable("p_lay", nc_float, ("lay", "y", "x"))
    nc_p_lay.description = "pressure - layer midpoints"
    nc_p_lay.units = "Pa"
    nc_p_lay[:,:,:] = p_lay

    nc_p_lev: nc._netCDF4.Variable = nc_file.createVariable("p_lev", nc_float, ("lev", "y", "x"))
    nc_p_lev.description = "pressure - layer interfaces"
    nc_p_lev.units = "Pa"
    nc_p_lev[:,:,:] = p_lev

    ## Temperature
    nc_T_lay: nc._netCDF4.Variable = nc_file.createVariable("t_lay", nc_float, ("lay", "y", "x"))
    nc_T_lay.description = "Temperature - layer midpoints"
    nc_T_lay.units = "K"
    nc_T_lay[:,:,:] = T_lay

    nc_T_lev: nc._netCDF4.Variable = nc_file.createVariable("t_lev", nc_float, ("lev", "y", "x"))
    nc_T_lev.description = "Temperature - layer interfaces"
    nc_T_lev.units = "K"
    nc_T_lev[:,:,:] = T_lev

    ## Relative humidity
    nc_rh_lay: nc._netCDF4.Variable = nc_file.createVariable("rh", nc_float, ("lay", "y", "x"))
    nc_rh_lay.description = "Relative humidity - layer midpoints"
    nc_rh_lay.units = "Pa Pa^(-1)"
    nc_rh_lay[:,:,:] = rh_lay

    ## Gas volume mixing ratios
    nc_co2: nc._netCDF4.Variable = nc_file.createVariable("vmr_co2", nc_float)
    nc_co2.description = "Volume mixing ratio - Carbon Dioxide (CO2)"
    nc_co2.units = "kg kg^(-1)"
    nc_co2[:] = co2
    
    nc_ch4: nc._netCDF4.Variable = nc_file.createVariable("vmr_ch4", nc_float)
    nc_ch4.description = "Volume mixing ratio - Methane (CH4)"
    nc_ch4.units = "kg kg^(-1)"
    nc_ch4[:] = ch4
    
    nc_n2o: nc._netCDF4.Variable = nc_file.createVariable("vmr_n2o", nc_float)
    nc_n2o.description = "Volume mixing ratio - Nitrous Oxide (N2O)"
    nc_n2o.units = "kg kg^(-1)"
    nc_n2o[:] = n2o
    
    nc_o3: nc._netCDF4.Variable = nc_file.createVariable("vmr_o3", nc_float, ("lay", "y", "x"))
    nc_o3.description = "Volume mixing ratio - Ozone (O3)"
    nc_o3.units = "kg kg^(-1)"
    nc_o3[:] = o3
    
    nc_h2o: nc._netCDF4.Variable = nc_file.createVariable("vmr_h2o", nc_float, ("lay", "y", "x"))
    nc_h2o.description = "Volume mixing ratio - Water Vapor (H2O)"
    nc_h2o.units = "kg kg^(-1)"
    nc_h2o[:] = h2o
    
    nc_n2: nc._netCDF4.Variable = nc_file.createVariable("vmr_n2", nc_float)
    nc_n2.description = "Volume mixing ratio - Nitrogen (N2)"
    nc_n2.units = "kg kg^(-1)"
    nc_n2[:] = n2
    
    nc_o2: nc._netCDF4.Variable = nc_file.createVariable("vmr_o2", nc_float)
    nc_o2.description = "Volume mixing ratio - Oxygen (O2)"
    nc_o2.units = "kg kg^(-1)"
    nc_o2[:] = o2
    
    nc_co: nc._netCDF4.Variable = nc_file.createVariable("vmr_co", nc_float)
    nc_co.description = "Volume mixing ratio - Carbon Monoxide (CO)"
    nc_co.units = "kg kg^(-1)"
    nc_co[:] = co
    
    nc_ccl4: nc._netCDF4.Variable = nc_file.createVariable("vmr_ccl4", nc_float)
    nc_ccl4.description = "Volume mixing ratio - Carbon Tetrachloride (CCl4)"
    nc_ccl4.units = "kg kg^(-1)"
    nc_ccl4[:] = ccl4
    
    nc_cfc11: nc._netCDF4.Variable = nc_file.createVariable("vmr_cfc11", nc_float)
    nc_cfc11.description = "Volume mixing ratio - Trichlorofluoromethane (CFC-11)"
    nc_cfc11.units = "kg kg^(-1)"
    nc_cfc11[:] = cfc11
    
    nc_cfc12: nc._netCDF4.Variable = nc_file.createVariable("vmr_cfc12", nc_float)
    nc_cfc12.description = "Volume mixing ratio - Dichlorodifluoromethane (CFC-12)"
    nc_cfc12.units = "kg kg^(-1)"
    nc_cfc12[:] = cfc12
    
    nc_cfc22: nc._netCDF4.Variable = nc_file.createVariable("vmr_cfc22", nc_float)
    nc_cfc22.description = "Volume mixing ratio - Chlorodifluoromethane (CFC-22)"
    nc_cfc22.units = "kg kg^(-1)"
    nc_cfc22[:] = cfc22
    
    nc_hfc143a: nc._netCDF4.Variable = nc_file.createVariable("vmr_hfc143a", nc_float)
    nc_hfc143a.description = "Volume mixing ratio - 1,1,1-Trifluoroethane (HFC-143a)"
    nc_hfc143a.units = "kg kg^(-1)"
    nc_hfc143a[:] = hfc143a
    
    nc_hfc125: nc._netCDF4.Variable = nc_file.createVariable("vmr_hfc125", nc_float)
    nc_hfc125.description = "Volume mixing ratio - Pentafluoroethane (HFC-125)"
    nc_hfc125.units = "kg kg^(-1)"
    nc_hfc125[:] = hfc125
    
    nc_hfc23: nc._netCDF4.Variable = nc_file.createVariable("vmr_hfc23", nc_float)
    nc_hfc23.description = "Volume mixing ratio - Trifluoromethane (HFC-23)"
    nc_hfc23.units = "kg kg^(-1)"
    nc_hfc23[:] = hfc23
    
    nc_hfc32: nc._netCDF4.Variable = nc_file.createVariable("vmr_hfc32", nc_float)
    nc_hfc32.description = "Volume mixing ratio - Difluoromethane (HFC-32)"
    nc_hfc32.units = "kg kg^(-1)"
    nc_hfc32[:] = hfc32
    
    nc_hfc134a: nc._netCDF4.Variable = nc_file.createVariable("vmr_hfc134a", nc_float)
    nc_hfc134a.description = "Volume mixing ratio - 1,1,1,2-Tetrafluoroethane (HFC-134a)"
    nc_hfc134a.units = "kg kg^(-1)"
    nc_hfc134a[:] = hfc134a
    
    nc_cf4: nc._netCDF4.Variable = nc_file.createVariable("vmr_cf4", nc_float)
    nc_cf4.description = "Volume mixing ratio - Carbon Tetrafluoride (CF4)"
    nc_cf4.units = "kg kg^(-1)"
    nc_cf4[:] = cf4
    
    nc_no2: nc._netCDF4.Variable = nc_file.createVariable("vmr_no2", nc_float)
    nc_no2.description = "Volume mixing ratio - Nitrogen Dioxide (NO2)"
    nc_no2.units = "kg kg^(-1)"
    nc_no2[:] = no2

    ## Aerosols mixing ratios
    nc_aermr01: nc._netCDF4.Variable = nc_file.createVariable("aermr01", nc_float, ("lay", "y", "x"))
    nc_aermr01.description = "Aerosol mixing ratio - Sea salt aerosol (0.03 - 0.5 µm)"
    nc_aermr01.units = "kg kg^(-1)"
    nc_aermr01[:] = aermr01

    nc_aermr02: nc._netCDF4.Variable = nc_file.createVariable("aermr02", nc_float, ("lay", "y", "x"))
    nc_aermr02.description = "Aerosol mixing ratio - Sea salt aerosol (0.5 - 5 µm)"
    nc_aermr02.units = "kg kg^(-1)"
    nc_aermr02[:] = aermr02

    nc_aermr03: nc._netCDF4.Variable = nc_file.createVariable("aermr03", nc_float, ("lay", "y", "x"))
    nc_aermr03.description = "Aerosol mixing ratio - Sea salt aerosol (5 - 20 µm)"
    nc_aermr03.units = "kg kg^(-1)"
    nc_aermr03[:] = aermr03

    nc_aermr04: nc._netCDF4.Variable = nc_file.createVariable("aermr04", nc_float, ("lay", "y", "x"))
    nc_aermr04.description = "Aerosol mixing ratio - Dust aerosol (0.03 - 0.55 µm)"
    nc_aermr04.units = "kg kg^(-1)"
    nc_aermr04[:] = aermr04

    nc_aermr05: nc._netCDF4.Variable = nc_file.createVariable("aermr05", nc_float, ("lay", "y", "x"))
    nc_aermr05.description = "Aerosol mixing ratio - Dust aerosol (0.55 - 0.9 µm)"
    nc_aermr05.units = "kg kg^(-1)"
    nc_aermr05[:] = aermr05

    nc_aermr06: nc._netCDF4.Variable = nc_file.createVariable("aermr06", nc_float, ("lay", "y", "x"))
    nc_aermr06.description = "Aerosol mixing ratio - Dust aerosol (0.9 - 20 µm)"
    nc_aermr06.units = "kg kg^(-1)"
    nc_aermr06[:] = aermr06

    nc_aermr07: nc._netCDF4.Variable = nc_file.createVariable("aermr07", nc_float, ("lay", "y", "x"))
    nc_aermr07.description = "Aerosol mixing ratio - Hydrophilic organic matter aerosol"
    nc_aermr07.units = "kg kg^(-1)"
    nc_aermr07[:] = aermr07

    nc_aermr08: nc._netCDF4.Variable = nc_file.createVariable("aermr08", nc_float, ("lay", "y", "x"))
    nc_aermr08.description = "Aerosol mixing ratio - Hydrophobic organic matter aerosol"
    nc_aermr08.units = "kg kg^(-1)"
    nc_aermr08[:] = aermr08

    nc_aermr09: nc._netCDF4.Variable = nc_file.createVariable("aermr09", nc_float, ("lay", "y", "x"))
    nc_aermr09.description = "Aerosol mixing ratio - Hydrophilic black carbon aerosol"
    nc_aermr09.units = "kg kg^(-1)"
    nc_aermr09[:] = aermr09

    nc_aermr10: nc._netCDF4.Variable = nc_file.createVariable("aermr10", nc_float, ("lay", "y", "x"))
    nc_aermr10.description = "Aerosol mixing ratio - Hydrophobic black carbon aerosol"
    nc_aermr10.units = "kg kg^(-1)"
    nc_aermr10[:] = aermr10

    nc_aermr11: nc._netCDF4.Variable = nc_file.createVariable("aermr11", nc_float, ("lay", "y", "x"))
    nc_aermr11.description = "Aerosol mixing ratio - Sulphate aerosol"
    nc_aermr11.units = "kg kg^(-1)"
    nc_aermr11[:] = aermr11

    ## Longwave boundary conditions
    nc_emis_sfc: nc._netCDF4.Variable = nc_file.createVariable("emis_sfc" , nc_float, ("y", "x", "band_lw"))
    nc_emis_sfc.description = "Surface emissivity"
    nc_emis_sfc.units = "N/A"
    nc_emis_sfc[:,:,:] = emis_sfc

    nc_t_sfc: nc._netCDF4.Variable = nc_file.createVariable("t_sfc" , nc_float, ("y", "x"))
    nc_t_sfc.description = "Surface temperature"
    nc_t_sfc.units = "K"
    nc_t_sfc[:,:] = t_sfc

    ## Shortwave boundary conditions
    nc_mu0: nc._netCDF4.Variable = nc_file.createVariable("mu0", nc_float, ("y", "x"))
    nc_mu0.description = "Cosine of solar zenith angle"
    nc_mu0.units = "N/A"
    nc_mu0[:,:] = mu0

    nc_sfc_alb_dir: nc._netCDF4.Variable = nc_file.createVariable("sfc_alb_dir", nc_float, ("y", "x", "band_sw"))
    nc_sfc_alb_dir.description = "Surface albedo - direct"
    nc_sfc_alb_dir.units = "N/A"
    nc_sfc_alb_dir[:,:,:] = sfc_alb_dir

    nc_sfc_alb_dif: nc._netCDF4.Variable = nc_file.createVariable("sfc_alb_dif", nc_float, ("y", "x", "band_sw"))
    nc_sfc_alb_dif.description = "Surface albedo - diffuse"
    nc_sfc_alb_dif.units = "N/A"
    nc_sfc_alb_dif[:,:,:] = sfc_alb_dif

    nc_tsi: nc._netCDF4.Variable = nc_file.createVariable("tsi", nc_float, ("y", "x"))
    nc_tsi.description = "Total solar irradiance"
    nc_tsi.units = "W m^(-2)"
    nc_tsi[:,:] = total_solar_irradiance

    nc_azi: nc._netCDF4.Variable = nc_file.createVariable("azi", nc_float, ("y", "x"))
    nc_azi.description = "Azimuthal angle"
    nc_azi.units = "radians"
    nc_azi[:,:] = azi

    ## Clouds
    nc_lwp: nc._netCDF4.Variable = nc_file.createVariable("lwp", nc_float, ("lay", "y", "x"))
    nc_lwp.description = "Liquid water path"
    nc_lwp.units = "kg m^(-2)"
    nc_lwp[:,:,:] = lwp

    nc_iwp: nc._netCDF4.Variable = nc_file.createVariable("iwp", nc_float, ("lay", "y", "x"))
    nc_iwp.description = "Ice water path"
    nc_iwp.units = "kg m^(-2)"
    nc_iwp[:,:,:] = iwp

    nc_rel: nc._netCDF4.Variable = nc_file.createVariable("rel", nc_float, ("lay", "y", "x"))
    nc_rel.description = "Liquid water effective radius"
    nc_rel.units = "μm"
    nc_rel[:,:,:] = rel

    nc_dei: nc._netCDF4.Variable = nc_file.createVariable("dei", nc_float, ("lay", "y", "x"))
    nc_dei.description = "Ice water effective diameter"
    nc_dei.units = "μm"
    nc_dei[:,:,:] = dei

    nc_file.close()

if __name__ == "__main__":
    main()

