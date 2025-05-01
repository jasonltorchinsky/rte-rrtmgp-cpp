# Standard Library Imports
import os

# Third-Party Library Imports
import netCDF4 as nc
import numpy as np

# Local Library Imports
from plot_profiles import plot_profiles_1d, plot_profile_3d

def main():
    ## Load the input and optics data
    input_file: str = "rte_rrtmgp_input.nc"
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file)

    optics_file: str = "aerosol_optics.nc"
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file)

    ## Create the output directories
    input_dir_name: str = "input"
    input_dir_path: str = os.path.join(os.getcwd(), input_dir_name)

    if not os.path.exists(input_dir_path):
        os.mkdir(input_dir_path)

    ## Extract the spatial variables
    x: np.ma.MaskedArray = nc_input.variables["x"][:]
    y: np.ma.MaskedArray = nc_input.variables["y"][:]

    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:]

    XX_lay: np.ndarray
    YY_lay: np.ndarray
    ZZ_lay: np.ndarray
    XX_lay, YY_lay, ZZ_lay = np.meshgrid(x, y, z_lay, indexing = "ij")

    z_lev: np.ma.MaskedArray = nc_input.variables["z_lev"][:]

    XX_lev: np.ndarray
    YY_lev: np.ndarray
    ZZ_lev: np.ndarray
    XX_lev, YY_lev, ZZ_lev = np.meshgrid(x, y, z_lev, indexing = "ij")

    nx: int = np.size(x)
    ny: int = np.size(y)

    nlay: int = np.size(z_lay)
    nlev: int = np.size(z_lev)
    nz: int = nlay + nlev

    z: np.ndarray = np.empty(nz, dtype = z_lev.dtype) # [m]
    z[0::2] = z_lev
    z[1::2] = z_lay

    XX: np.ndarray
    YY: np.ndarray
    ZZ: np.ndarray
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing = "ij")

    ## Extract the wavenumber information
    wavenumber1_lw: np.ma.MaskedArray = nc_optics.variables["wavenumber1_lw"][:] # (band_lw); [cm^(-1)]
    wavenumber2_lw: np.ma.MaskedArray = nc_optics.variables["wavenumber2_lw"][:] # (band_lw); [cm^(-1)]

    wavenumber1_sw: np.ma.MaskedArray = nc_optics.variables["wavenumber1_sw"][:] # (band_sw); [cm^(-1)]
    wavenumber2_sw: np.ma.MaskedArray = nc_optics.variables["wavenumber2_sw"][:] # (band_sw); [cm^(-1)]

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

    ## Plot the zonally- and meridionally-averaged vertical pressure profile
    p_lay: np.ma.MaskedArray = nc_input.variables["p_lay"][:] # (lay, y, x); [Pa]
    p_lev: np.ma.MaskedArray = nc_input.variables["p_lev"][:] # (lev, y, x); [Pa]

    p: np.ndarray = np.empty([nz, ny, nx], dtype = p_lev.dtype)
    p[0::2,...] = p_lev
    p[1::2,...] = p_lay

    p_z: np.ndarray = np.nanmean(p, axis = (1, 2))

    coord: np.ndarray = z / 1000. # (nz) [km]
    profiles: list = [p_z] # (nz) [K]
    file_path: str = os.path.join(input_dir_path, "pressure.png")
    xlabel: str = r"Pressure $[hPa]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Plot the zonally- and meridionally-averaged vertical temperture profile
    t_lay: np.ma.MaskedArray = nc_input.variables["t_lay"][:] # (lay, y, x); [K]
    t_lev: np.ma.MaskedArray = nc_input.variables["t_lev"][:] # (lev, y, x); [K]

    t: np.ndarray = np.empty([nz, ny, nx], dtype = t_lev.dtype) # (nz); [K]
    t[0::2,...] = t_lev
    t[1::2,...] = t_lay

    t_z: np.ndarray = np.nanmean(t, axis = (1, 2))

    coord: np.ndarray = z / 1000. # (nz) [km]
    profiles: list = [t_z] # (nz) [K]
    file_path: str = os.path.join(input_dir_path, "temperature.png")
    xlabel: str = r"Temperature $[K]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Plot the zonally- and meridionally-averaged vertical volume mixing ratio profiles
    vmr_co2: np.ma.MaskedArray = nc_input.variables["vmr_co2"][:]
    vmr_ch4: np.ma.MaskedArray = nc_input.variables["vmr_ch4"][:]
    vmr_n2o: np.ma.MaskedArray = nc_input.variables["vmr_n2o"][:]
    vmr_o3_lay: np.ma.MaskedArray = nc_input.variables["vmr_o3"][:] # (lay, y, x)
    vmr_h2o_lay: np.ma.MaskedArray = nc_input.variables["vmr_h2o"][:] # (lay, y, x)
    vmr_n2: np.ma.MaskedArray = nc_input.variables["vmr_n2"][:]
    vmr_o2: np.ma.MaskedArray = nc_input.variables["vmr_o2"][:]

    vmr_co2_z: np.ndarray = np.tile(vmr_co2, (nlay))
    vmr_ch4_z: np.ndarray = np.tile(vmr_ch4, (nlay))
    vmr_n2o_z: np.ndarray = np.tile(vmr_n2o, (nlay))
    vmr_o3_z: np.ndarray = np.nanmean(vmr_o3_lay, axis = (1, 2))
    vmr_h2o_z: np.ndarray = np.nanmean(vmr_h2o_lay, axis = (1, 2))
    vmr_n2_z: np.ndarray = np.tile(vmr_n2, (nlay))
    vmr_o2_z: np.ndarray = np.tile(vmr_o2, (nlay))

    coord: np.ndarray = z_lay / 1000. # (nlay); [km]
    profiles: list = [vmr_co2_z, vmr_ch4_z, vmr_n2o_z, vmr_o3_z, vmr_h2o_z, vmr_n2_z, vmr_o2_z]
    file_path: str = os.path.join(input_dir_path, "vmr.png")
    profile_labels: list = [r"$C O_2$", r"$C H_4$", r"$N_2 O$", r"$O_3$", r"$H_2 O$", r"$N_2$", r"$O_2$"]
    xlabel: str = r"Volume Mixing Ratio"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"
    xscale: str = "log"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels, 
                     xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis,
                     xscale = xscale)

    ## Plot the zonally- and meridionally-averaged surface emissivity spectrum
    emis_sfc: np.ma.MaskedArray = nc_input.variables["emis_sfc"][:] # (y, x, band_lw)

    emis_sfc_spec: np.ndarray = np.nanmean(emis_sfc, axis = (0, 1)) # (band_lw)

    ### Repeat the last value for the step plot
    emis_sfc_spec: np.ndarray = np.concatenate((emis_sfc_spec, np.array([emis_sfc_spec[-1]]))) # (band_lw + 1)

    coord: np.ndarray = wavenumber_lw # (band_lw); [cm^(-1)]
    profiles: list = [emis_sfc_spec]
    file_path: str = os.path.join(input_dir_path, "emis_sfc.png")
    xlabel: str = r"Wavenumber [$cm^{-1}$]"
    ylabel: str = r"Surface Emissivity - Longwave"
    coord_axis: str = "x"
    draw_style: str = "steps-post"

    plot_profiles_1d(coord, profiles, file_path, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis, draw_style = draw_style)

    ## Plot the zonally- and meridionally-averaged surface albedo (direct and diffuse)
    sfc_alb_dir: np.ma.MaskedArray = nc_input.variables["sfc_alb_dir"][:] # (y, x, band_sw)
    sfc_alb_dif: np.ma.MaskedArray = nc_input.variables["sfc_alb_dif"][:] # (y, x, band_sw)

    sfc_alb_dir_spec: np.ndarray = np.nanmean(sfc_alb_dir, axis = (0, 1)) # (band_sw); [cm^(-1)]
    sfc_alb_dif_spec: np.ndarray = np.nanmean(sfc_alb_dif, axis = (0, 1)) # (band_sw); [cm^(-1)]

    ### Repeat the last value for the step plot
    sfc_alb_dir_spec: np.ndarray = np.concatenate((sfc_alb_dir_spec, np.array([sfc_alb_dir_spec[-1]]))) # (band_sw + 1)
    sfc_alb_dif_spec: np.ndarray = np.concatenate((sfc_alb_dif_spec, np.array([sfc_alb_dif_spec[-1]]))) # (band_sw + 1)

    coord: np.ndarray = wavenumber_sw # (band_sw); [cm^(-1)]
    profiles: list = [sfc_alb_dir_spec, sfc_alb_dif_spec]
    file_path: str = os.path.join(input_dir_path, "sfc_alb.png")
    xlabel: str = r"Wavenumber [$cm^{-1}$]"
    ylabel: str = r"Surface Albedo - Shortwave"
    coord_axis: str = "x"
    draw_style: str = "steps-post"

    plot_profiles_1d(coord, profiles, file_path, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis, draw_style = draw_style)

    # Plot the liquid water path
    tol: float = 0.1
    lwp: np.ma.MaskedArray = nc_input.variables["lwp"][:] # (lay, y, x); [kg m^(-2)]
    lwp_npts: np.int64 = np.sum((lwp > tol * lwp.max()))
    if (lwp_npts <= 100000):
        meshgrid: np.ndarray = [XX_lay / 1000., YY_lay / 1000., ZZ_lay / 1000.] #  [km]
        profile: list = np.transpose(lwp, axes = (2, 1, 0))
        file_path: str = os.path.join(input_dir_path, "lwp.png")
        xlabel: str = r"x [$km$]"
        ylabel: str = r"y [$km$]"
        zlabel: str = r"z [$km$]"
        title: str = r"Liquid Water Path [$kg\,m^{-2}$]"
        cmap: str = "Blues"

        plot_profile_3d(meshgrid, profile, file_path, xlabel = xlabel, ylabel = ylabel,
                        zlabel = zlabel, title = title, cmap = cmap, tol = tol)

    # Plot the ice water path
    tol: float = 0.1
    iwp: np.ma.MaskedArray = nc_input.variables["iwp"][:] # (lay, y, x); [kg m^(-2)]
    iwp_npts: np.int64 = np.sum((iwp > tol * iwp.max()))
    if (iwp_npts <= 100000):
        meshgrid: np.ndarray = [XX_lay / 1000., YY_lay / 1000., ZZ_lay / 1000.] #  [km]
        profile: list = np.transpose(iwp, axes = (2, 1, 0))
        file_path: str = os.path.join(input_dir_path, "iwp.png")
        xlabel: str = r"x [$km$]"
        ylabel: str = r"y [$km$]"
        zlabel: str = r"z [$km$]"
        title: str = r"Ice Water Path [$kg\,m^{-2}$]"
        cmap: str = "Purples"

        plot_profile_3d(meshgrid, profile, file_path, xlabel = xlabel, ylabel = ylabel,
                        zlabel = zlabel, title = title, cmap = cmap, tol = tol)

if __name__ == "__main__":
    main()

