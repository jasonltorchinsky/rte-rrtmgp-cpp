# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from plot_profiles import plot_profiles_1d, plot_profile_2d

def main():
    ## Load the input, output, and optics data
    input_file: str = "rte_rrtmgp_input.nc"
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file)

    optics_file: str = "aerosol_optics.nc"
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file)

    output_file: str = "rte_rrtmgp_output.nc"
    nc_output: nc._netCDF4.Dataset = nc.Dataset(output_file)

    ## Extract the spatial variables
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # [m]

    XX: np.ndarray
    YY: np.ndarray
    XX, YY = np.meshgrid(x, y, indexing = "ij")

    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # [m]
    z_lev: np.ma.MaskedArray = nc_input.variables["z_lev"][:] # [m]

    nx: int = np.size(x)
    ny: int = np.size(y)

    nlay: int = np.size(z_lay)
    nlev: int = np.size(z_lev)
    nz: int = nlay + nlev

    z: np.ndarray = np.empty(nz, dtype = z_lev.dtype) # [m]
    z[0::2] = z_lev
    z[1::2] = z_lay

    ## Extract the wavenumber information
    wavenumber1_lw: np.ma.MaskedArray = nc_optics.variables["wavenumber1_lw"][:] # (band_lw), [cm^(-1)]
    wavenumber2_lw: np.ma.MaskedArray = nc_optics.variables["wavenumber2_lw"][:] # (band_lw), [cm^(-1)]

    wavenumber1_sw: np.ma.MaskedArray = nc_optics.variables["wavenumber1_sw"][:] # (band_sw), [cm^(-1)]
    wavenumber2_sw: np.ma.MaskedArray = nc_optics.variables["wavenumber2_sw"][:] # (band_sw), [cm^(-1)]

    band_lw: int = np.size(wavenumber1_lw)
    band_sw: int = np.size(wavenumber1_sw)
    
    ### Bin edges - ASSUME: wavenumber1_Xw looks like lower bin bounds, e.g., [0, 1, 2, 3, 4, 5]
    ### and wavenumber2_Xw looks like upper bin bounds, e.g., [1, 2, 3, 4, 5, 6]
    wavenumber_lw: np.ndarray = np.empty(band_lw + 1, dtype = wavenumber1_lw.dtype) # [cm^(-1)]
    wavenumber_lw[0:-1] = wavenumber1_lw
    wavenumber_lw[-1] = wavenumber2_lw[-1]

    wavenumber_sw: np.ndarray = np.empty(band_sw + 1, dtype = wavenumber1_sw.dtype) # [cm^(-1)]
    wavenumber_sw[0:-1] = wavenumber1_sw
    wavenumber_sw[-1] = wavenumber2_sw[-1]

    ## Plot the zonally- and meridionally-averaged vertical shortwave fluxes profiles (TwoStream Solver)
    sw_flux_up: np.ma.MaskedArray = nc_output.variables["sw_flux_up"][:] # (lev, y, x); [W m^(-2)]
    sw_flux_dn: np.ma.MaskedArray = nc_output.variables["sw_flux_dn"][:] # (lev, y, x); [W m^(-2)]
    sw_flux_net: np.ma.MaskedArray = nc_output.variables["sw_flux_net"][:] # (lev, y, x); [W m^(-2)]

    sw_flux_up_z: np.ndarray = np.nanmean(sw_flux_up, axis = (1, 2)) # (lev)
    sw_flux_dn_z: np.ndarray = np.nanmean(sw_flux_dn, axis = (1, 2)) # (lev)
    sw_flux_net_z: np.ndarray = np.nanmean(sw_flux_net, axis = (1, 2)) # (lev)

    profiles: list = [sw_flux_up_z, sw_flux_dn_z, sw_flux_net_z]
    profile_labels: list = [r"Upwelling", r"Downwelling", r"Net"]

    plot_profiles_1d(z_lev / 1000., profiles, "sw_flux.png",
                     profile_labels = profile_labels,
                     title = "TwoStream Solver",
                     xlabel = r"Shortwave Flux $[W m^{-2}]$", ylabel = r"z $[km]$",
                     coord_axis = "y")

    ## Plot the zonally- and meridionally-averaged vertical absorbed shortwave fluxes profiles (Monte Carlo Ray Tracer)
    rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
    rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]

    rt_flux_abs_dir_z: np.ndarray = np.nanmean(rt_flux_abs_dir, axis = (1, 2)) # (lay)
    rt_flux_abs_dif_z: np.ndarray = np.nanmean(rt_flux_abs_dif, axis = (1, 2)) # (lay)

    profiles: list = [rt_flux_abs_dir_z, rt_flux_abs_dif_z]
    profile_labels: list = [r"Direct", r"Diffuse"]

    plot_profiles_1d(z_lay / 1000., profiles, "rt_flux_abs.png",
                     title = "Monte Carlo Ray Tracer",
                     xlabel = r"Absorbed Shortwave Flux $[W m^{-3}]$", ylabel = r"z $[km]$",
                     coord_axis = "y")

    ## Plot the upwelling shortwave top-of-domain flux (Monte Carlo Ray Tracer)
    rt_flux_tod_up: np.ma.MaskedArray = nc_output.variables["rt_flux_tod_up"][:] # (y, x); [W m^(-2)]

    plot_profile_2d([XX / 1000., YY / 1000.], rt_flux_tod_up, "rt_flux_tod_up.png",
                    title = "Monte Carlo Ray Tracer", 
                    xlabel = r"x [$km$]", ylabel = r"y [$km$]",
                    cbarlabel = r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]")

    ## Plot the upwelling shortwave surface flux (Monte Carlo Ray Tracer)
    rt_flux_sfc_up: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_up"][:] # (y, x); [W m^(-2)]

    plot_profile_2d([XX / 1000., YY / 1000.], rt_flux_sfc_up, "rt_flux_sfc_up.png",
                    title = "Monte Carlo Ray Tracer", 
                    xlabel = r"x [$km$]", ylabel = r"y [$km$]",
                    cbarlabel = r"Upwelling Shortwave Surface Flux [$W m^{-2}$]")

if __name__ == "__main__":
    main()
