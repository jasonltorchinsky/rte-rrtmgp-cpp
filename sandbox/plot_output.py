# Standard Library Imports
import os

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

    ## Create the output directories
    ts_dir_name: str = "ts"
    ts_dir_path: str = os.path.join(os.getcwd(), ts_dir_name)
    rt_dir_name: str = "rt"
    rt_dir_path: str = os.path.join(os.getcwd(), rt_dir_name)

    if not os.path.exists(ts_dir_path):
        os.mkdir(ts_dir_path)

    if not os.path.exists(rt_dir_path):
        os.mkdir(rt_dir_path)

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


    # Two Stream Solver
    ## Plot the zonally- and meridionally-averaged vertical shortwave fluxes profiles (TwoStream Solver)
    sw_flux_up: np.ma.MaskedArray = nc_output.variables["sw_flux_up"][:] # (lev, y, x); [W m^(-2)]
    sw_flux_dn: np.ma.MaskedArray = nc_output.variables["sw_flux_dn"][:] # (lev, y, x); [W m^(-2)]
    sw_flux_net: np.ma.MaskedArray = nc_output.variables["sw_flux_net"][:] # (lev, y, x); [W m^(-2)]

    sw_flux_up_z: np.ndarray = np.nanmean(sw_flux_up, axis = (1, 2)) # (lev)
    sw_flux_dn_z: np.ndarray = np.nanmean(sw_flux_dn, axis = (1, 2)) # (lev)
    sw_flux_net_z: np.ndarray = np.nanmean(sw_flux_net, axis = (1, 2)) # (lev)

    coord: np.ndarray = z_lev / 1000. # (lev); [km]
    profiles: list = [sw_flux_up_z, sw_flux_dn_z, sw_flux_net_z]
    profile_labels: list = [r"Upwelling", r"Downwelling", r"Net"]
    file_path: str = os.path.join(ts_dir_path, "sw_flux.png")
    title: str = "Two Stream Solver"
    xlabel: str = r"Shortwave Flux $[W m^{-2}]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)
    
    ## Plot the zonally- and meridionally-averaged vertical absorbed shortwave fluxes profiles (Two Stream Solver)
    ts_flux_abs: np.ma.MaskedArray = ((sw_flux_dn[1:] + sw_flux_up[:-1]) - (sw_flux_dn[:-1] + sw_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (lay, y, x); [W m^(-3)]

    ts_flux_abs_z: np.ndarray = np.nanmean(ts_flux_abs, axis = (1, 2)) # (lay); [W m^(-3)]

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    profiles: list = [ts_flux_abs_z]
    profile_labels: list = [r"Total"]
    file_path: str = os.path.join(ts_dir_path, "ts_flux_abs.png")
    title: str = "Two Stream Solver"
    xlabel: str = r"Absorbed Shortwave Flux $[W m^{-3}]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Plot the upwelling shortwave top-of-domain flux (Two Stream Solver)
    ts_flux_tod_up: np.ma.MaskedArray = sw_flux_up[-1,:,:] # (y, x); [W m^(-2)]

    meshgrid: tuple = [XX / 1000., YY / 1000.]
    profile: np.ndarray = ts_flux_tod_up
    file_path: str = os.path.join(ts_dir_path, "ts_flux_tod_up.png")
    title: str = "Two Stream Solver"
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]"

    plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
                    ylabel = ylabel, cbarlabel = cbarlabel)

    ## Plot the upwelling shortwave surface flux (Two Stream Solver)
    ts_flux_sfc_up: np.ma.MaskedArray = sw_flux_up[1,:,:] # (y, x); [W m^(-2)]

    meshgrid: tuple = [XX / 1000., YY / 1000.]
    profile: np.ndarray = ts_flux_sfc_up
    file_path: str = os.path.join(ts_dir_path, "ts_flux_sfc_up.png")
    title: str = "Two Stream Solver"
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Upwelling Shortwave Surface Flux [$W m^{-2}$]"

    plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
                    ylabel = ylabel, cbarlabel = cbarlabel)

    # Monte Carlo Ray Tracer
    ## Plot the zonally- and meridionally-averaged vertical absorbed shortwave fluxes profiles (Monte Carlo Ray Tracer)
    rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
    rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]

    rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dir + rt_flux_abs_dif # (lay, y, x); [W m^(-3)]

    rt_flux_abs_dir_z: np.ndarray = np.nanmean(rt_flux_abs_dir, axis = (1, 2)) # (lay); [W m^(-3)]
    rt_flux_abs_dif_z: np.ndarray = np.nanmean(rt_flux_abs_dif, axis = (1, 2)) # (lay); [W m^(-3)]
    rt_flux_abs_z: np.ndarray = np.nanmean(rt_flux_abs, axis = (1, 2)) # (lay); [W m^(-3)]

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    profiles: list = [rt_flux_abs_dir_z, rt_flux_abs_dif_z, rt_flux_abs_z]
    profile_labels: list = [r"Direct", r"Diffuse", r"Total"]
    file_path: str = os.path.join(rt_dir_path, "rt_flux_abs.png")
    title: str = "Monte Carlo Ray Tracer"
    xlabel: str = r"Absorbed Shortwave Flux $[W m^{-3}]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Plot the upwelling shortwave top-of-domain flux (Monte Carlo Ray Tracer)
    rt_flux_tod_up: np.ma.MaskedArray = nc_output.variables["rt_flux_tod_up"][:] # (y, x); [W m^(-2)]

    meshgrid: tuple = [XX / 1000., YY / 1000.]
    profile: np.ndarray = rt_flux_tod_up
    file_path: str = os.path.join(rt_dir_path, "rt_flux_tod_up.png")
    title: str = "Monte Carlo Ray Tracer"
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]"

    plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
                    ylabel = ylabel, cbarlabel = cbarlabel)

    ## Plot the upwelling shortwave surface flux (Monte Carlo Ray Tracer)
    rt_flux_sfc_up: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_up"][:] # (y, x); [W m^(-2)]

    meshgrid: tuple = [XX / 1000., YY / 1000.]
    profile: np.ndarray = rt_flux_sfc_up
    file_path: str = os.path.join(rt_dir_path, "rt_flux_sfc_up.png")
    title: str = "Monte Carlo Ray Tracer"
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Upwelling Shortwave Surface Flux [$W m^{-2}$]"

    plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
                    ylabel = ylabel, cbarlabel = cbarlabel)

if __name__ == "__main__":
    main()
