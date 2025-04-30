"""
Generates a NetCDF file (rte_rrtmgp_input.nc) that contains atmospheric and
surface properties used for RTE-RRTMGP. It defines vertical profiles of pressure,
temperature, and specific humidity, as well as various atmospheric and surface
parameters.
"""

# Strandard Library Imports

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from consts import R_d, R_v, L_v, g, p_0

def main():
    nc_float: str = "f8"

    ## Spatial grids
    n_col_x: int = 256 # Number of columns in x
    n_col_y: int = 256 # Number of columns in y
    n_lay_z: int = 128 # Number of layers in z
    n_lev_z: int = n_lay_z + 1 # Number of levels in z

    ngrid_x: int = 48 # Number of grid points in x
    ngrid_y: int = 48 # Number of grid points in y
    ngrid_z: int = 32 # Number of grid points in z

    x_min: float = 0. # [m]
    x_max: float = 6670. # [m]
    dx: float = (x_max - x_min) / (n_col_x - 1) # [m]
    x: np.ndarray = np.linspace(x_min, x_max, n_col_x) # [m]
    xh: np.ndarray = np.linspace(x_min, x_max + dx, n_col_x + 1) # [m]

    y_min: float = 0. # [m]
    y_max: float = 6670. # [m]
    dy: float = (y_max - y_min) / (n_col_y - 1) # [m]
    y: np.ndarray = np.linspace(y_min, y_max, n_col_y) # [m]
    yh: np.ndarray = np.linspace(y_min, y_max + dy, n_col_y + 1) # [m]

    z_min: float = 0. # [m]
    z_max: float = 1440. # [m]
    z_lev: np.ndarray = np.linspace(z_min, z_max, n_lev_z) # [m]
    z_lay: np.ndarray = (z_lev[1:] + z_lev[:-1]) / 2. # [m]

    ## Spectral grids
    n_bnd_lw: int = 16 # Number of bands in longwave
    n_bnd_sw: int = 14 # Number of bands in shortwave

    ## Pressure (p), specific humidity (q), and temperature (T)
    p_lay: np.ndarray # [Pa]; (nlay)
    q_lay: np.ndarray # [kg kg^(-1)]; (nlay)
    T_lay: np.ndarray # [K]; (nlay)
    rh_lay: np.ndarray # [Pa Pa^(-1)]
    p_lay, q_lay, T_lay, rh_lay = calc_p_q_T_rh(z_lay)

    p_lev: np.ndarray # [Pa]; (nlev)
    T_lev: np.ndarray # [K]; (nlev)
    p_lev, _, T_lev, _ = calc_p_q_T_rh(z_lev)

    ## Clouds
    lwp: np.ndarray # Liquid Water Path [kg m^(-2)]; (nlay, nx, ny)
    iwp: np.ndarray # Ice Water Path [kg m^(-2)]; (nlay, nx, ny)
    rel: np.ndarray # Liquid Water Effective Radius [μm]; (nlay, nx, ny)
    dei: np.ndarray # Ice Water Effective Diameter [μm]; (nlay, nx, ny)
    lwp, iwp, rel, dei = calc_cloud(x, y, z_lay)

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
    solar_zenith_angle: np.float64 = np.deg2rad(42.05) # [N/A]
    mu0: np.float64 = np.cos(solar_zenith_angle) # [N/A]

    sfc_alb_dir: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Direct
    sfc_alb_dif: np.ndarray = np.ones((n_col_y, n_col_x, n_bnd_sw)) * 0.07 # Surface Albedo - Diffuse

    total_solar_irradiance: float = 551.58 # [W m^(-2)]

    azi: float = 1.834 # Azimuthal Angle [Radians]

    ## Write input to file
    nc_file: nc.Dataset = nc.Dataset("rte_rrtmgp_input.nc", mode = "w",
                                     datamodel = "NETCDF4", clobber = True)

    nc_file.createDimension("x", n_col_x)
    nc_file.createDimension("y", n_col_y)
    nc_file.createDimension("lay", p_lay.size)
    nc_file.createDimension("lev", p_lev.size)
    nc_file.createDimension("z", n_lay_z)
    nc_file.createDimension("xh", n_col_x + 1)
    nc_file.createDimension("yh", n_col_y + 1)
    nc_file.createDimension("zh", n_lev_z)
    nc_file.createDimension("band_lw", n_bnd_lw)
    nc_file.createDimension("band_sw", n_bnd_sw)

    ## Spatial grid
    nc_ngrid_x: nc._netCDF4.Variable = nc_file.createVariable("ngrid_x", nc_float)
    nc_ngrid_x.description = "Number of grid points in the x-direction"
    nc_ngrid_x[:] = ngrid_x

    nc_ngrid_y: nc._netCDF4.Variable = nc_file.createVariable("ngrid_y", nc_float)
    nc_ngrid_y.description = "Number of grid points in the y-direction"
    nc_ngrid_y[:] = ngrid_y

    nc_ngrid_z: nc._netCDF4.Variable = nc_file.createVariable("ngrid_z", nc_float)
    nc_ngrid_z.description = "Number of grid points in the z-direction"
    nc_ngrid_z[:] = ngrid_z

    nc_x: nc._netCDF4.Variable = nc_file.createVariable("x", nc_float, "x")
    nc_x.description = "x-dimension"
    nc_x.units = "m"
    nc_x[:] = x

    nc_y: nc._netCDF4.Variable = nc_file.createVariable("y", nc_float, "y")
    nc_y.description = "y-dimension"
    nc_y.units = "m"
    nc_y[:] = y

    nc_z: nc._netCDF4.Variable = nc_file.createVariable("z", nc_float, "z")
    nc_z.description = "z-dimension - layer midpoints"
    nc_z.units = "m"
    nc_z[:] = z_lay

    nc_xh: nc._netCDF4.Variable = nc_file.createVariable("xh", nc_float, "xh")
    nc_xh.description = "x-dimension - augmented grid (?)"
    nc_xh.units = "m"
    nc_xh[:] = xh

    nc_yh: nc._netCDF4.Variable = nc_file.createVariable("yh", nc_float, "yh")
    nc_yh.description = "y-dimension - augmented grid (?)"
    nc_yh.units = "m"
    nc_yh[:] = yh

    nc_zh: nc._netCDF4.Variable = nc_file.createVariable("zh", nc_float, "zh")
    nc_zh.description = "z-dimension - layer interfaces, augmented grid (?)"
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
    nc_p_lay[:,:,:] = np.tile(p_lay[:, None, None], (1, n_col_y, n_col_x))

    nc_p_lev: nc._netCDF4.Variable = nc_file.createVariable("p_lev", nc_float, ("lev", "y", "x"))
    nc_p_lev.description = "pressure - layer interfaces"
    nc_p_lev.units = "Pa"
    nc_p_lev[:,:,:] = np.tile(p_lev[:, None, None], (1, n_col_y, n_col_x))

    ## Temperature
    nc_T_lay: nc._netCDF4.Variable = nc_file.createVariable("t_lay", nc_float, ("lay", "y", "x"))
    nc_T_lay.description = "Temperature - layer midpoints"
    nc_T_lay.units = "K"
    nc_T_lay[:,:,:] = np.tile(T_lay[:, None, None], (1, n_col_y, n_col_x))

    nc_T_lev: nc._netCDF4.Variable = nc_file.createVariable("t_lev", nc_float, ("lev", "y", "x"))
    nc_T_lev.description = "Temperature - layer interfaces"
    nc_T_lev.units = "K"
    nc_T_lev[:,:,:] = np.tile(T_lev[:, None, None], (1, n_col_y, n_col_x))

    ## Relative humidity
    nc_rh_lay: nc._netCDF4.Variable = nc_file.createVariable("rh", nc_float, ("lay", "y", "x"))
    nc_rh_lay.description = "Relative humidity - layer midpoints"
    nc_rh_lay.units = "Pa Pa^(-1)"
    nc_rh_lay[:,:,:] = np.tile(rh_lay[:, None, None], (1, n_col_y, n_col_x))

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
    nc_o3[:] = np.tile(o3[:, None, None], (1, n_col_y, n_col_x))
    
    nc_h2o: nc._netCDF4.Variable = nc_file.createVariable("vmr_h2o", nc_float, ("lay", "y", "x"))
    nc_h2o.description = "Volume mixing ratio - Water Vapor (H2O)"
    nc_h2o.units = "kg kg^(-1)"
    nc_h2o[:] = np.tile(h2o[:, None, None], (1, n_col_y, n_col_x))
    
    nc_n2: nc._netCDF4.Variable = nc_file.createVariable("vmr_n2", nc_float)
    nc_n2.description = "Volume mixing ratio - Nitrogen (N2)"
    nc_n2.units = "kg kg^(-1)"
    nc_n2[:] = n2
    
    nc_o2: nc._netCDF4.Variable = nc_file.createVariable("vmr_o2", nc_float)
    nc_o2.description = "Volume mixing ratio - Oxygen (O2)"
    nc_o2.units = "kg kg^(-1)"
    nc_o2[:] = o2
    
    nc_co = nc_file.createVariable("vmr_co", nc_float)
    nc_co.description = "Volume mixing ratio - Carbon Monoxide (CO)"
    nc_co.units = "kg kg^(-1)"
    nc_co[:] = co
    
    nc_ccl4 = nc_file.createVariable("vmr_ccl4", nc_float)
    nc_ccl4.description = "Volume mixing ratio - Carbon Tetrachloride (CCl4)"
    nc_ccl4.units = "kg kg^(-1)"
    nc_ccl4[:] = ccl4
    
    nc_cfc11 = nc_file.createVariable("vmr_cfc11", nc_float)
    nc_cfc11.description = "Volume mixing ratio - Trichlorofluoromethane (CFC-11)"
    nc_cfc11.units = "kg kg^(-1)"
    nc_cfc11[:] = cfc11
    
    nc_cfc12 = nc_file.createVariable("vmr_cfc12", nc_float)
    nc_cfc12.description = "Volume mixing ratio - Dichlorodifluoromethane (CFC-12)"
    nc_cfc12.units = "kg kg^(-1)"
    nc_cfc12[:] = cfc12
    
    nc_cfc22 = nc_file.createVariable("vmr_cfc22", nc_float)
    nc_cfc22.description = "Volume mixing ratio - Chlorodifluoromethane (CFC-22)"
    nc_cfc22.units = "kg kg^(-1)"
    nc_cfc22[:] = cfc22
    
    nc_hfc143a = nc_file.createVariable("vmr_hfc143a", nc_float)
    nc_hfc143a.description = "Volume mixing ratio - 1,1,1-Trifluoroethane (HFC-143a)"
    nc_hfc143a.units = "kg kg^(-1)"
    nc_hfc143a[:] = hfc143a
    
    nc_hfc125 = nc_file.createVariable("vmr_hfc125", nc_float)
    nc_hfc125.description = "Volume mixing ratio - Pentafluoroethane (HFC-125)"
    nc_hfc125.units = "kg kg^(-1)"
    nc_hfc125[:] = hfc125
    
    nc_hfc23 = nc_file.createVariable("vmr_hfc23", nc_float)
    nc_hfc23.description = "Volume mixing ratio - Trifluoromethane (HFC-23)"
    nc_hfc23.units = "kg kg^(-1)"
    nc_hfc23[:] = hfc23
    
    nc_hfc32 = nc_file.createVariable("vmr_hfc32", nc_float)
    nc_hfc32.description = "Volume mixing ratio - Difluoromethane (HFC-32)"
    nc_hfc32.units = "kg kg^(-1)"
    nc_hfc32[:] = hfc32
    
    nc_hfc134a = nc_file.createVariable("vmr_hfc134a", nc_float)
    nc_hfc134a.description = "Volume mixing ratio - 1,1,1,2-Tetrafluoroethane (HFC-134a)"
    nc_hfc134a.units = "kg kg^(-1)"
    nc_hfc134a[:] = hfc134a
    
    nc_cf4 = nc_file.createVariable("vmr_cf4", nc_float)
    nc_cf4.description = "Volume mixing ratio - Carbon Tetrafluoride (CF4)"
    nc_cf4.units = "kg kg^(-1)"
    nc_cf4[:] = cf4
    
    nc_no2 = nc_file.createVariable("vmr_no2", nc_float)
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
    nc_sfc_alb_dir[:,:,:] = sfc_alb_dir[:,:]

    nc_sfc_alb_dif: nc._netCDF4.Variable = nc_file.createVariable("sfc_alb_dif", nc_float, ("y", "x", "band_sw"))
    nc_sfc_alb_dif.description = "Surface albedo - diffuse"
    nc_sfc_alb_dif.units = "N/A"
    nc_sfc_alb_dif[:,:,:] = sfc_alb_dif[:,:]

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

# Local Function Definitions
def calc_p_q_T_rh(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the pressure, specific humidity, and temperature profiles based on altitude.

    Parameters:
    z (np.ndarray): Altitude levels [m]

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]:
        - Pressure [Pa]
        - Specific humidity [kg kg^(-1)]
        - Temperature [K]
    """


    ## Calculate specific humidity vertical profile
    q_0: float = 0.01864 # Surface specific humidity (300 K SST); [kg kg^(-1)]
    z_q1: float = 4.0e3 # Scale height for exponential decay of specific humidity; [m]
    z_q2: float = 7.5e3 # Scale height for Gaussian decay of specific humidity; [m]
    z_t: float = 15.e3 # Transition height above which humidity and temperature are capped/flattened; [m]
    q_t: float = 1.e-14 # Minimum specific humidity value above transition height; [kg kg^(-1)]

    q: np.ndarray = q_0 * np.exp(-z / z_q1) * np.exp(-(z / z_q2)**2) # Specific humidity (nz); [kg kg^(-1)]

    # CvH hack to remove moisture jump.
    q_t: np.ndarray = q_0 * np.exp(-z_t/z_q1) * np.exp(-(z_t/z_q2)**2) # [kg kg^(-1)]

    i_above_zt: tuple[np.ndarray] = np.where(z > z_t)
    q[i_above_zt] = q_t

    ## Calculate temperature vertical profile
    T_0: float = 300. # SST; [K]
    gamma: float = 6.7e-3 # Temperature lapse rate; [K m^(-1)]
    Tv_0 : float= (1. + (R_v / R_d - 1) * q_0) * T_0 # Virtual temperature at surface; [K]
    Tv: np.ndarray = Tv_0 - gamma * z # Virtual temperature (nz); [K]
    Tv_t: float = Tv_0 - gamma * z_t # Virtual temperature at transition height; [K]
    
    Tv[i_above_zt] = Tv_t
    T: np.ndarray = Tv / (1. + (R_v / R_d - 1) * q) # Temperature (nz); [K]

    ## Calculate pressure vertical profile
    p: np.ndarray = p_0 * (Tv / Tv_0)**(g / (R_d * gamma)) # [Pa]

    p_tmp: np.ndarray = p_0 * (Tv_t / Tv_0)**(g / (R_d * gamma)) \
                        * np.exp( -( (g * (z - z_t)) / (R_d * Tv_t) ) ) # [Pa]

    p[i_above_zt] = p_tmp[i_above_zt]

    ## Calculate relative humidity profile using Clausius-Clapeyron [Vallis Textbook]
    vap_pres: np.ndarray = (q * p) / (q + (R_d / R_v) * (1. - q)) # Vapor pressure (nz); [Pa]

    e0: float = 6.12e2 # [Pa]
    T0: float = 273. # [K]

    sat_vap_pres: np.ndarray = e0 * np.exp(L_v / R_v * ((1. / T0) - (1. / T))) # Saturation vapor pressure (nz); [Pa]

    rh = vap_pres / sat_vap_pres # Relative humidity [Pa Pa^(-1)]

    return p, q, T, rh

def calc_cloud(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rel_min: float = 2.5 # Minimum Liquid Water Effective Radius [μm]
    rel_max: float = 21.5 # Maximum Liquid Water Effective Radius [μm]

    dei_min: float = 10. # Minimum Ice Water Effective Diameter [μm]
    dei_max: float = 180. # Maximum Ice Water Effective Diameter [μm]

    rel_val: float = (rel_min + rel_max) / 2. #  [μm]
    dei_val: float = (dei_min + dei_max) / 2. #  [μm]

    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    Z, X, Y = np.meshgrid(z, x, y, indexing = "ij")

    x_mid: np.float64 = (x.max() + x.min()) / 2.
    x_width: np.float64 = x.max() - x.min()

    y_mid: np.float64 = (y.max() + y.min()) / 2.
    y_width: np.float64 = y.max() - y.min()

    z_mid: np.float64 = (z.max() + z.min()) / 2.
    z_width: np.float64 = z.max() - z.min()

    ## Set cloud mask
    cloud_mask_X: np.ndarray = (np.abs(X - x_mid) <= 0.1 * x_width)
    cloud_mask_Y: np.ndarray = (np.abs(Y - y_mid) <= 0.1 * y_width)
    cloud_mask_Z: np.ndarray = (490. <= Z) & (Z <= 690.)

    cloud_mask: np.ndarray = cloud_mask_X & cloud_mask_Y & cloud_mask_Z

    lwp: np.ndarray = np.where(cloud_mask, 10., 0.) # Liquid Water Path [kg m^(-2)]
    iwp: np.ndarray = np.where(cloud_mask, 10., 0.) # Ice Water Path [kg m^(-2)]
    rel: np.ndarray = np.where(lwp[:,:,:] > 0., rel_val, 0.) # Liquid Water Effective Radius [μm]
    dei: np.ndarray = np.where(iwp[:,:,:] > 0., dei_val, 0.) # Ice Water Effective Diameter [μm]

    return lwp, iwp, rel, dei

if __name__ == "__main__":
    main()
