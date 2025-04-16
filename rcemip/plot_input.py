# Standard Library Imports

# Third-Party Library Imports
import netCDF4 as nc
import numpy as np

# Local Library Imports
from plot_profiles import plot_profiles_1d

def main():
    ## Load the input and optics data
    input_file: str = "rte_rrtmgp_input.nc"
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file)

    optics_file: str = "aerosol_optics.nc"
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file)

    ## Extract the spatial variables
    x: np.ma.MaskedArray = nc_input.variables["x"][:]
    y: np.ma.MaskedArray = nc_input.variables["y"][:]

    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:]
    z_lev: np.ma.MaskedArray = nc_input.variables["z_lev"][:]

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

    ## Plot the zonally- and meridionally-averaged vertical pressure profile
    p_lay: np.ma.MaskedArray = nc_input.variables["p_lay"][:] # (lay, y, x); [Pa]
    p_lev: np.ma.MaskedArray = nc_input.variables["p_lev"][:] # (lev, y, x); [Pa]

    p: np.ndarray = np.empty([nz, ny, nx], dtype = p_lev.dtype)
    p[0::2,...] = p_lev
    p[1::2,...] = p_lay

    p_z: np.ndarray = np.nanmean(p, axis = (1, 2))
    plot_profiles_1d(z / 1000., [p_z / 100.], "pressure.png",
                xlabel = r"Pressure $[hPa]$", ylabel = r"z $[km]$",
                coord_axis = "y")

    ## Plot the zonally- and meridionally-averaged vertical temperture profile
    t_lay: np.ma.MaskedArray = nc_input.variables["t_lay"][:] # (lay, y, x); [K]
    t_lev: np.ma.MaskedArray = nc_input.variables["t_lev"][:] # (lev, y, x); [K]

    t: np.ndarray = np.empty([nz, ny, nx], dtype = t_lev.dtype)
    t[0::2,...] = t_lev
    t[1::2,...] = t_lay

    t_z: np.ndarray = np.nanmean(t, axis = (1, 2))
    plot_profiles_1d(z / 1000., [t_z], "temperature.png",
                xlabel = r"Temperature $[K]$", ylabel = r"z $[km]$",
                coord_axis = "y")

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

    vmr_zs: list = [vmr_co2_z, vmr_ch4_z, vmr_n2o_z, vmr_o3_z, vmr_h2o_z, vmr_n2_z, vmr_o2_z]
    vmr_labels: list = [r"$C O_2$", r"$C H_4$", r"$N_2 O$", r"$O_3$", r"$H_2 O$", r"$N_2$", r"$O_2$"]

    plot_profiles_1d(z_lay / 1000., vmr_zs, "vmr.png",
                  xlabel = r"Volume Mixing Ratio", ylabel = r"z $[km]$",
                  profile_labels = vmr_labels, xscale = "log", coord_axis = "y")

    ## Plot the zonally- and meridionally-averaged surface emissivity spectrum
    emis_sfc: np.ma.MaskedArray = nc_input.variables["emis_sfc"][:] # (y, x, band_lw)

    emis_sfc_spec: np.ndarray = np.nanmean(emis_sfc, axis = (0, 1)) # (band_lw)

    ### Repeat the last value for the step plot
    emis_sfc_spec: np.ndarray = np.concatenate((emis_sfc_spec, np.array([emis_sfc_spec[-1]]))) # (band_lw + 1)

    plot_profiles_1d(wavenumber_lw, [emis_sfc_spec], "emis_sfc.png",
                  xlabel = r"Wavenumber [$cm^{-1}$]", ylabel = r"Surface Emissivity - Longwave",
                  coord_axis = "x", drawstyle = "steps-post")

    ## Plot the zonally- and meridionally-averaged surface albedo (direct and diffuse)
    sfc_alb_dir: np.ma.MaskedArray = nc_input.variables["sfc_alb_dir"][:] # (y, x, band_sw)
    sfc_alb_dif: np.ma.MaskedArray = nc_input.variables["sfc_alb_dif"][:] # (y, x, band_sw)

    sfc_alb_dir_spec: np.ndarray = np.nanmean(sfc_alb_dir, axis = (0, 1)) # (band_sw)
    sfc_alb_dif_spec: np.ndarray = np.nanmean(sfc_alb_dif, axis = (0, 1)) # (band_sw)

    ### Repeat the last value for the step plot
    sfc_alb_dir_spec: np.ndarray = np.concatenate((sfc_alb_dir_spec, np.array([sfc_alb_dir_spec[-1]]))) # (band_sw + 1)
    sfc_alb_dif_spec: np.ndarray = np.concatenate((sfc_alb_dif_spec, np.array([sfc_alb_dif_spec[-1]]))) # (band_sw + 1)


    plot_profiles_1d(wavenumber_sw, [sfc_alb_dir_spec, sfc_alb_dif_spec], "sfc_alb.png",
                  xlabel = r"Wavenumber [$cm^{-1}$]", ylabel = r"Surface Albedo - Longwave",
                  profile_labels = [r"Direct", r"Diffuse"],
                  coord_axis = "x", drawstyle = "steps-post")

if __name__ == "__main__":
    main()

