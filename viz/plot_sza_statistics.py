# Standard Library Imports
import argparse
import ast
import os

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from consts import np_float, np_INF
from plot_profiles import plot_profiles_1d, plot_profile_2d, plot_profiles_2d_3d

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_output",
        description = "Plots the output of RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Path to RTE-RRTMGP-CPP input file.")
    
    parser.add_argument("--szas",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Solar zenith angles.")

    parser.add_argument("--output",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Path to RTE-RRTMGP-CPP output file.")
    
    parser.add_argument("--optics",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = False,
                        default = ["aerosol_optics.nc"],
                        help = "Path to aerosol optics file.")

    parser.add_argument("--outdir",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = False,
                        default = ["output"],
                        help = "Path to output generated plots.")
    
    args: argparse.Namespace = parser.parse_args()

    input_file_path: str = os.path.normpath(args.input[0])
    szas: np.ndarray = np.array(ast.literal_eval(args.szas[0]), dtype = np_float).flatten()
    nszas: int = int(szas.size)
    output_file_path_base: str = os.path.normpath(args.output[0])
    optics_file_path: str = os.path.normpath(args.optics[0])
    out_dir_path: str = os.path.normpath(args.outdir[0])

    ## Load the input, output, and optics data
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file_path)

    ## Create the output directories
    out_dir_path: str = os.path.join(os.getcwd(), out_dir_path)
    sza_dir_name: str = "sza"
    sza_dir_path: str = os.path.join(out_dir_path, sza_dir_name)
    stats_dir_name: str = "stats"
    stats_dir_path: str = os.path.join(sza_dir_path, stats_dir_name)

    dir_paths: list[str] = [out_dir_path, sza_dir_path, stats_dir_path]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    ## Extract the spatial variables
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # [m]

    XX: np.ndarray
    YY: np.ndarray
    XX, YY = np.meshgrid(x, y, indexing = "ij")

    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # [m]
    
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

    ## Store values for each SZA
    ts_flux_tod_up_dict: dict = {}
    ts_flux_sfc_up_dict: dict = {}
    ts_flux_sfc_dn_dict: dict = {}
    ts_flux_abs_dict: dict = {}

    rt_flux_tod_up_dict: dict = {}
    rt_flux_sfc_up_dict: dict = {}
    rt_flux_sfc_dn_dict: dict = {}
    rt_flux_abs_dict: dict = {}

    for sza in szas:
        sza_str: str = "{:04.0f}".format(sza)
        output_file_path: str = output_file_path_base + "." + sza_str + ".out.nc"
        nc_output: nc._netCDF4.Dataset = nc.Dataset(output_file_path)

        ## Two-Stream
        ts_sw_flux_up: np.ma.MaskedArray = nc_output.variables["sw_flux_up"][:] # (lev, y, x); [W m^(-2)]
        ts_sw_flux_dn: np.ma.MaskedArray = nc_output.variables["sw_flux_dn"][:] # (lev, y, x); [W m^(-2)]
        ts_flux_abs: np.ma.MaskedArray = ((ts_sw_flux_dn[1:] + ts_sw_flux_up[:-1]) - (ts_sw_flux_dn[:-1] + ts_sw_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (lay, y, x); [W m^(-3)]
        
        ts_flux_tod_up_dict[sza]: np.ma.MaskedArray = ts_sw_flux_up[-1, ...] # (y, x); [W m^(-2)]
        ts_flux_sfc_up_dict[sza]: np.ma.MaskedArray = ts_sw_flux_up[0, ...] # (y, x); [W m^(-2)]
        ts_flux_sfc_dn_dict[sza]: np.ma.MaskedArray = ts_sw_flux_dn[0, ...] # (y, x); [W m^(-2)]
        ts_flux_abs_dict[sza]: np.ma.MaskedArray = ts_flux_abs

        ## Ray-Tracer
        rt_flux_tod_up: np.ma.MaskedArray = nc_output.variables["rt_flux_tod_up"][:] # (y, x); [W m^{-2}]
        rt_flux_sfc_up: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_up"][:] # (y, x); [W m^{-2}]

        rt_flux_sfc_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_dir"][:] # (y, x); [W m^{-2}]
        rt_flux_sfc_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_dif"][:] # (y, x); [W m^{-2}]
        rt_flux_sfc_dn: np.ma.MaskedArray =  rt_flux_sfc_dir +  rt_flux_sfc_dif # (y, x); [W m^{-2}]

        rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
        rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dir + rt_flux_abs_dif # (lay, y, x); [W m^(-3)]
        
        rt_flux_tod_up_dict[sza]: np.ma.MaskedArray = rt_flux_tod_up # (y, x); [W m^(-2)]
        rt_flux_sfc_up_dict[sza]: np.ma.MaskedArray = rt_flux_sfc_up # (y, x); [W m^(-2)]
        rt_flux_sfc_dn_dict[sza]: np.ma.MaskedArray = rt_flux_sfc_dn # (y, x); [W m^(-2)]
        rt_flux_abs_dict[sza]: np.ma.MaskedArray = rt_flux_abs

    ## Bulk Mean 3D Signal - ToD Upward, Surface Upward, Surface Downward, and Absorbed Flux
    profile_short_names: list[str] = ["flux_tod_up", "flux_sfc_up", "flux_sfc_dn", "flux_abs"]
    profiles: list[np.ma.MaskedArray] = []
    profile_labels: list[str] = [r"Top-of-Domain Upward Shortwave Flux $[W m^{-2}]$", 
        r"Surface Upward Shortwave Flux $[W m^{-2}]$", r"Surface Downward Shortwave Flux $[W m^{-2}]$",
        r"Absorbed Shortwave Flux $[W m^{-3}]$"]
    for profile_short_name in profile_short_names:
        profile: np.ndarray = np.zeros([nszas])
        for ii in range(nszas):
            sza: np.float64 = szas[ii]

            if profile_short_name == "flux_tod_up":
                ts_flux: np.ma.MaskedArray = ts_flux_tod_up_dict[sza] # (y, x); [W m^(-2)]
                rt_flux: np.ma.MaskedArray = rt_flux_tod_up_dict[sza] # (y, x); [W m^(-2)]
            elif profile_short_name == "flux_sfc_up":
                ts_flux: np.ma.MaskedArray = ts_flux_sfc_up_dict[sza] # (y, x); [W m^(-2)]
                rt_flux: np.ma.MaskedArray = rt_flux_sfc_up_dict[sza] # (y, x); [W m^(-2)]
            elif profile_short_name == "flux_sfc_dn":
                ts_flux: np.ma.MaskedArray = ts_flux_sfc_dn_dict[sza] # (y, x); [W m^(-2)]
                rt_flux: np.ma.MaskedArray = rt_flux_sfc_dn_dict[sza] # (y, x); [W m^(-2)]
            elif profile_short_name == "flux_abs":
                ts_flux: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
                rt_flux: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

            diff_flux: np.ma.MaskedArray = (rt_flux - ts_flux) / np.abs(rt_flux).max()
            profile[ii]: np.ma.MaskedArray = np.nanmean(diff_flux)
        if not np.any(np.isnan(profile)):
            profiles += [profile]

    coord: np.ndarray = szas # (nszas)
    file_path: str = os.path.join(stats_dir_path, "bulk_mean_3d.png")
    title: str = "(RT - TS) / max(|RT|) - Bulk Mean"
    xlabel: str = r"Solar Zenith Angle $(^{\circ})$"
    coord_axis: str = "x"
    viz: str = "difference"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, coord_axis = coord_axis, viz = viz)

    ## Absolute Mean 3D Signal - ToD Upward, Surface Upward, Surface Downward, and Absorbed Flux
    profile_short_names: list[str] = ["flux_tod_up", "flux_sfc_up", "flux_sfc_dn", "flux_abs"]
    profiles: list[np.ma.MaskedArray] = []
    profile_labels: list[str] = [r"Top-of-Domain Upward Shortwave Flux $[W m^{-2}]$", 
        r"Surface Upward Shortwave Flux $[W m^{-2}]$", r"Surface Downward Shortwave Flux $[W m^{-2}]$",
        r"Absorbed Shortwave Flux $[W m^{-3}]$"]
    for profile_short_name in profile_short_names:
        profile: np.ndarray = np.zeros([nszas])
        for ii in range(nszas):
            sza: np.float64 = szas[ii]

            if profile_short_name == "flux_tod_up":
                ts_flux: np.ma.MaskedArray = ts_flux_tod_up_dict[sza] # (y, x); [W m^(-2)]
                rt_flux: np.ma.MaskedArray = rt_flux_tod_up_dict[sza] # (y, x); [W m^(-2)]
            elif profile_short_name == "flux_sfc_up":
                ts_flux: np.ma.MaskedArray = ts_flux_sfc_up_dict[sza] # (y, x); [W m^(-2)]
                rt_flux: np.ma.MaskedArray = rt_flux_sfc_up_dict[sza] # (y, x); [W m^(-2)]
            elif profile_short_name == "flux_sfc_dn":
                ts_flux: np.ma.MaskedArray = ts_flux_sfc_dn_dict[sza] # (y, x); [W m^(-2)]
                rt_flux: np.ma.MaskedArray = rt_flux_sfc_dn_dict[sza] # (y, x); [W m^(-2)]
            elif profile_short_name == "flux_abs":
                ts_flux: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
                rt_flux: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

            diff_flux: np.ma.MaskedArray = np.abs(rt_flux - ts_flux) / np.abs(rt_flux).max()
            profile[ii]: np.ma.MaskedArray = np.nanmean(diff_flux)
        if not np.any(np.isnan(profile)):
            profiles += [profile]

    coord: np.ndarray = szas # (nszas)
    file_path: str = os.path.join(stats_dir_path, "abs_mean_3d.png")
    title: str = "|RT - TS| / max(|RT|) - Absolute Mean"
    xlabel: str = r"Solar Zenith Angle $(^{\circ})$"
    coord_axis: str = "x"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, coord_axis = coord_axis)
    
if __name__ == "__main__":
    main()

