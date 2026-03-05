# Standard Library Imports
import argparse
import ast
import os

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from consts import np_float, np_INF
from plot_tools import plot_profiles_1d, plot_profile_2d, plot_profiles_2d_3d

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
    diff_dir_name: str = "diff"
    diff_dir_path: str = os.path.join(sza_dir_path, diff_dir_name)
    ts_dir_name: str = "ts"
    ts_dir_path: str = os.path.join(sza_dir_path, ts_dir_name)
    rt_dir_name: str = "rt"
    rt_dir_path: str = os.path.join(sza_dir_path, rt_dir_name)
    err_dir_name: str = "err"
    err_dir_path: str = os.path.join(sza_dir_path, err_dir_name)

    dir_paths: list[str] = [out_dir_path, sza_dir_path, diff_dir_path, err_dir_path, 
        ts_dir_path, rt_dir_path]
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

    ## Store Total Absorbed Shortwave Flux for each SZA
    ts_flux_abs_dict: dict = {}
    rt_flux_abs_dict: dict = {}

    ts_flux_abs_min: np.float64 = np.float64(np_INF)
    ts_flux_abs_max: np.float64 = np.float64(-np_INF)

    rt_flux_abs_min: np.float64 = np.float64(np_INF)
    rt_flux_abs_max: np.float64 = np.float64(-np_INF)

    diff_flux_abs_max: np.float64 = np.float64(-np_INF)

    for sza in szas:
        sza_str: str = "{:04.0f}".format(sza)
        output_file_path: str = output_file_path_base + "." + sza_str + ".out.nc"
        nc_output: nc._netCDF4.Dataset = nc.Dataset(output_file_path)

        ## Two-Stream
        ts_sw_flux_up: np.ma.MaskedArray = nc_output.variables["sw_flux_up"][:] # (lev, y, x); [W m^(-2)]
        ts_sw_flux_dn: np.ma.MaskedArray = nc_output.variables["sw_flux_dn"][:] # (lev, y, x); [W m^(-2)]
        ts_flux_abs: np.ma.MaskedArray = ((ts_sw_flux_dn[1:] + ts_sw_flux_up[:-1]) - (ts_sw_flux_dn[:-1] + ts_sw_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (lay, y, x); [W m^(-3)]
        ts_flux_abs_dict[sza]: np.ma.MaskedArray = ts_flux_abs

        ts_flux_abs_min: np.float64 = min(ts_flux_abs_min, ts_flux_abs.min())
        ts_flux_abs_max: np.float64 = max(ts_flux_abs_max, ts_flux_abs.max())

        ## Ray-Tracer
        rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
        rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dir + rt_flux_abs_dif # (lay, y, x); [W m^(-3)]
        rt_flux_abs_dict[sza]: np.ma.MaskedArray = rt_flux_abs

        rt_flux_abs_min: np.float64 = min(rt_flux_abs_min, rt_flux_abs.min())
        rt_flux_abs_max: np.float64 = max(rt_flux_abs_max, rt_flux_abs.max())

        ## Difference
        diff_flux_abs: np.float64 = (rt_flux_abs - ts_flux_abs) / np.abs(rt_flux_abs).max()
        diff_flux_abs_max: np.float64 = max(diff_flux_abs_max, np.abs(diff_flux_abs).max())

    ## Vertical profile of total absorbed shortwave flux for varying SZA (Two Stream)
    profiles: list[np.ma.MaskedArray] = []
    profile_labels: list[str] = []
    for sza in szas:
        ts_flux_abs: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

        ts_flux_abs: np.ma.MaskedArray = np.nanmean(ts_flux_abs, axis = (1, 2)) # (lay); [W m^(-3)]

        if not np.any(np.isnan(ts_flux_abs.data)):
            profiles += [ts_flux_abs]
            profile_labels += [r"${:.1f}^\circ$".format(sza)]

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    file_path: str = os.path.join(ts_dir_path, "ts_flux_abs.png")
    title: str = "Two Stream"
    xlabel: str = r"Absorbed Shortwave Flux $[W m^{-3}]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Vertical profile of total absorbed shortwave flux for varying SZA (Ray Tracer)
    profiles: list[np.ma.MaskedArray] = []
    profile_labels: list[str] = []
    for sza in szas:
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

        rt_flux_abs: np.ma.MaskedArray = np.nanmean(rt_flux_abs, axis = (1, 2)) # (lay); [W m^(-3)]

        if not np.any(np.isnan(rt_flux_abs.data)):
            profiles += [rt_flux_abs]
            profile_labels += [r"${:.1f}^\circ$".format(sza)]

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    file_path: str = os.path.join(rt_dir_path, "rt_flux_abs.png")
    title: str = "Ray Tracer"
    xlabel: str = r"Absorbed Shortwave Flux $[W m^{-3}]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Vertical slice profile of total absorbed shortwave flux for varying SZA (Two Stream)
    for sza in szas:
        ts_flux_abs: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

        ts_flux_abs: np.ma.MaskedArray = np.nanmean(ts_flux_abs, axis = (2)) # (lay, y); [W m^(-3)]

        ts_flux_abs: np.ma.MaskedArray = np.transpose(ts_flux_abs, axes = (1, 0)) # (y, lay); [W m^(-3)]

        if not np.any(np.isnan(ts_flux_abs.data)):
            sza_str: str = "{:04.0f}".format(sza)
            file_name: str = "ts_flux_abs_" + sza_str + ".png"

            profile: np.ndarray = ts_flux_abs.data
            meshgrid: tuple[np.ndarray] = [YY_lay[0,...] / 1000., ZZ_lay[0,...] / 1000.]
            file_path: str = os.path.join(ts_dir_path, file_name)
            title: str = "Two Stream - " + r"${:.1f}^\circ$".format(sza)
            xlabel: str = r"y $\left[ km \right]$"
            ylabel: str = r"z $\left[ km \right]$"
            cbarlabel: str = r"Absorbed Shortwave Flux [$W m^{-3}$]"
            cmin: float = min(ts_flux_abs_min, rt_flux_abs_min)
            cmax: float = max(ts_flux_abs_max, rt_flux_abs_max)

            plot_profile_2d(meshgrid, profile, file_path, title = title, 
                            xlabel = xlabel, ylabel = ylabel,
                            cbarlabel = cbarlabel, cmin = cmin, cmax = cmax)

    ## Vertical slice profile of total absorbed shortwave flux for varying SZA (Ray Tracer)
    for sza in szas:
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

        rt_flux_abs: np.ma.MaskedArray = np.nanmean(rt_flux_abs, axis = (2)) # (lay, y); [W m^(-3)]

        rt_flux_abs: np.ma.MaskedArray = np.transpose(rt_flux_abs, axes = (1, 0)) # (y, lay); [W m^(-3)]

        if not np.any(np.isnan(rt_flux_abs.data)):
            sza_str: str = "{:04.0f}".format(sza)
            file_name: str = "rt_flux_abs_" + sza_str + ".png"

            profile: np.ndarray = rt_flux_abs.data
            meshgrid: tuple[np.ndarray] = [YY_lay[0,...] / 1000., ZZ_lay[0,...] / 1000.]
            file_path: str = os.path.join(rt_dir_path, file_name)
            title: str = "Ray Tracer - " + r"${:.1f}^\circ$".format(sza)
            xlabel: str = r"y $\left[ km \right]$"
            ylabel: str = r"z $\left[ km \right]$"
            cbarlabel: str = r"Absorbed Shortwave Flux [$W m^{-3}$]"
            cmin: float = min(ts_flux_abs_min, rt_flux_abs_min)
            cmax: float = max(ts_flux_abs_max, rt_flux_abs_max)

            plot_profile_2d(meshgrid, profile, file_path, title = title, 
                            xlabel = xlabel, ylabel = ylabel,
                            cbarlabel = cbarlabel, cmin = cmin, cmax = cmax)

    ## Vertical profile of difference of total absorbed shortwave flux for varying SZA
    profiles: list[np.ma.MaskedArray] = []
    profile_labels: list[str] = []
    for sza in szas:
        ts_flux_abs: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

        diff_flux_abs: np.ma.MaskedArray = (rt_flux_abs - ts_flux_abs) / np.abs(rt_flux_abs).max() # (lay, y, x); [W m^(-3)]
        diff_flux_abs: np.ma.MaskedArray = np.nanmean(diff_flux_abs, axis = (1, 2)) # (lay); [W m^(-3)]

        if not np.any(np.isnan(diff_flux_abs.data)):
            profiles += [diff_flux_abs]
            profile_labels += [r"${:.1f}^\circ$".format(sza)]

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    file_path: str = os.path.join(diff_dir_path, "diff_flux_abs.png")
    title: str = "(RT - TS) / max(|RT|)"
    xlabel: str = r"Absorbed Shortwave Flux $\left[ W m^{-3} \right]$"
    ylabel: str = r"z $\left[ km \right]$"
    coord_axis: str = "y"
    viz: str = "difference"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis, viz = viz)

    ## Vertical slice profile of difference of total absorbed shortwave flux for varying SZA
    for sza in szas:
        ts_flux_abs: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
        
        diff_flux_abs: np.ma.MaskedArray = (rt_flux_abs - ts_flux_abs) / np.abs(rt_flux_abs).max() # (lay, y, x); [W m^(-3)]
        diff_flux_abs: np.ma.MaskedArray = np.nanmean(diff_flux_abs, axis = (2)) # (lay, y); [W m^(-3)]
        diff_flux_abs: np.ma.MaskedArray = np.transpose(diff_flux_abs, axes = (1, 0)) # (y, lay); [W m^(-3)]

        if not np.any(np.isnan(diff_flux_abs.data)):
            sza_str: str = "{:04.0f}".format(sza)
            file_name: str = "diff_flux_abs_" + sza_str + ".png"

            profile: np.ndarray = diff_flux_abs.data
            meshgrid: tuple[np.ndarray] = [YY_lay[0,...] / 1000., ZZ_lay[0,...] / 1000.]
            file_path: str = os.path.join(diff_dir_path, file_name)
            title: str = "(RT - TS) / max(|RT|) - " + r"${:.1f}^\circ$".format(sza)
            xlabel: str = r"y $\left[ km \right]$"
            ylabel: str = r"z $\left[ km \right]$"
            cbarlabel: str = r"Absorbed Shortwave Flux [$W m^{-3}$]"
            cmin: float = -diff_flux_abs_max
            cmax: float = diff_flux_abs_max
            cmap: str = "bwr"
            cscale: str = "difference"

            plot_profile_2d(meshgrid, profile, file_path, title = title, 
                            xlabel = xlabel, ylabel = ylabel,
                            cbarlabel = cbarlabel, cmin = cmin, cmax = cmax,
                            cmap = cmap, cscale = cscale)

    ## Vertical profile of RMSE of total absorbed shortwave flux for varying SZA
    profiles: list[np.ma.MaskedArray] = []
    profile_labels: list[str] = []
    for sza in szas:
        ts_flux_abs: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]

        err_flux_abs: np.ma.MaskedArray = np.power((rt_flux_abs - ts_flux_abs) / np.abs(rt_flux_abs).max(), 2) # (lay, y, x); [W m^(-3)]
        err_flux_abs: np.ma.MaskedArray = np.nanmean(err_flux_abs, axis = (1, 2)) # (lay); [W m^(-3)]
        err_flux_abs: np.ma.MaskedArray = np.sqrt(err_flux_abs) # (lay); [W m^(-3)]

        if not np.any(np.isnan(err_flux_abs.data)):
            profiles += [err_flux_abs]
            profile_labels += [r"${:.1f}^\circ$".format(sza)]

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    file_path: str = os.path.join(err_dir_path, "err_flux_abs.png")
    title: str = "Relative Root-Mean-Square Error"
    xlabel: str = r"Absorbed Shortwave Flux $\left[ W m^{-3} \right]$"
    ylabel: str = r"z $\left[ km \right]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel,
                     coord_axis = coord_axis)

    ## Vertical slice profile of RMSE of total absorbed shortwave flux for varying SZA
    for sza in szas:
        ts_flux_abs: np.ma.MaskedArray = ts_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
        rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dict[sza] # (lay, y, x); [W m^(-3)]
        
        err_flux_abs: np.ma.MaskedArray = np.power((rt_flux_abs - ts_flux_abs) / np.abs(rt_flux_abs).max(), 2) # (lay, y, x); [W m^(-3)]
        err_flux_abs: np.ma.MaskedArray = np.nanmean(err_flux_abs, axis = (2)) # (lay, y); [W m^(-3)]
        err_flux_abs: np.ma.MaskedArray = np.sqrt(err_flux_abs) # (lay, y); [W m^(-3)]

        err_flux_abs: np.ma.MaskedArray = np.transpose(err_flux_abs, axes = (1, 0)) # (y, lay); [W m^(-3)]

        if not np.any(np.isnan(err_flux_abs.data)):
            sza_str: str = "{:04.0f}".format(sza)
            file_name: str = "err_flux_abs_" + sza_str + ".png"

            profile: np.ndarray = err_flux_abs.data
            meshgrid: tuple[np.ndarray] = [YY_lay[0,...] / 1000., ZZ_lay[0,...] / 1000.]
            file_path: str = os.path.join(err_dir_path, file_name)
            title: str = "Relative Root-Mean-Square Error - " + r"${:.1f}^\circ$".format(sza)
            xlabel: str = r"y $\left[ km \right]$"
            ylabel: str = r"z $\left[ km \right]$"
            cbarlabel: str = r"Absorbed Shortwave Flux [$W m^{-3}$]"
            cmin: float = 0.0
            cmax: float = diff_flux_abs_max
            cmap: str = "Reds"

            plot_profile_2d(meshgrid, profile, file_path, title = title, 
                            xlabel = xlabel, ylabel = ylabel,
                            cbarlabel = cbarlabel, cmin = cmin, cmax = cmax,
                            cmap = cmap)
    
if __name__ == "__main__":
    main()

