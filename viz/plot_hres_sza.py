# Standard Library Imports
import argparse
import ast
import os

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from consts import np_float, np_INF
from plot_tools import plot_profiles_1d, plot_profiles_1d_grid, \
    plot_profile_2d_grid, plot_scatter_grid

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_output",
        description = "Plots the output of RTE-RRTMGP-CPP.")
    
    parser.add_argument("--rteprefix",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Prefix of RTE-RRTMGP-CPP file, e.g., 'path/scream_dpxx_RICO_doSW'.")
    
    parser.add_argument("--rtesuffix",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Suffix of RTE-RRTMGP-CPP file, e.g., 'scream.INSTANT.nhours_x1.2004-12-16-00000'.")
    
    parser.add_argument("--nxs",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Number of horizontal elements in x (equivalently in y).")

    parser.add_argument("--szas",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Solar zenith angles.")
    
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

    rte_prefix: str = os.path.normpath(args.rteprefix[0])
    rte_suffix: str = os.path.normpath(args.rtesuffix[0])
    nelemxs: np.ndarray = np.array(ast.literal_eval(args.nxs[0]), dtype = np_float).flatten()
    szas: np.ndarray = np.array(ast.literal_eval(args.szas[0]), dtype = np_float).flatten()
    optics_file_path: str = os.path.normpath(args.optics[0])
    out_dir_path: str = os.path.normpath(args.outdir[0])

    ## Set maximum elevation for plots
    z_max: np_float = 10000. # [m]
    
    ## Derived quantities from input
    hress: np.ndarray = 50 / (nelemxs * 2) # Domain size of simulation is 50 km X 50 km; resolution is x / (2 * nx)

    ## Load the input, output, and optics data
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file_path)

    ## Create the output directories
    out_dir_path: str = os.path.join(os.getcwd(), out_dir_path)
    hres_sza_dir_name: str = "hres_sza"
    hres_sza_dir_path: str = os.path.join(out_dir_path, hres_sza_dir_name)

    dir_paths: list[str] = [out_dir_path, hres_sza_dir_path]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    nnelemx: int = np.size(nelemxs)
    nsza: int = np.size(szas)

    ## Create several dicts into which we put varying tensors so we're not
    ## constantly re-reading the files
    ## Keys are of the form (sza, nx).
    x_dict: dict = dict()
    y_dict: dict = dict()
    z_lev_full_dict: dict = dict()
    z_lay_full_dict: dict = dict()
    
    z_full_dict: dict = dict()

    z_lev_dict: dict = dict()
    z_lay_dict: dict = dict()
    z_dict: dict = dict()

    XX_lay_dict: dict = dict()
    YY_lay_dict: dict = dict()
    ZZ_lay_dict: dict = dict()

    hres_dict: dict = dict()

    ts_flux_abs_dict: dict = dict()
    rt_flux_abs_dict: dict = dict()

    ts_flux_sfc_up_dict: dict = dict()
    rt_flux_sfc_up_dict: dict = dict()

    ts_flux_tod_up_dict: dict = dict()
    rt_flux_tod_up_dict: dict = dict()

    rzmsre_flux_abs_dict: dict = dict()
    rhmsre_flux_abs_dict: dict = dict()
    rmsre_flux_abs_dict: dict = dict()

    ## Extract profiles from files
    for ii in range(0, nnelemx):
        nelemx: np_float = nelemxs[ii]
        hres: np_float = hress[ii]

        nelemx_str: str = "{:02.0f}x{:02.0f}".format(nelemx, nelemx)
        hres_str: str = "{:02.0f} km".format(hres)

        ## Input file is same for all nxs, choose a dummy SZA to open file and read it
        dummy_sza: np_float = szas[0]
        dummy_sza_str: str = "{:04.0f}".format(dummy_sza)

        ## Read input file
        input_file_path: str = rte_prefix + "_" + nelemx_str + "." + rte_suffix + "." + dummy_sza_str + ".in.nc"
        nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)

        ## Extract the spatial variables
        x: np.ma.MaskedArray = nc_input.variables["x"][:] # [m]
        y: np.ma.MaskedArray = nc_input.variables["y"][:] # [m]

        nx: int = np.size(x)
        ny: int = np.size(y)

        z_lay_full: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # [m]
        z_lev_full: np.ma.MaskedArray = nc_input.variables["z_lev"][:] # [m]

        nlay_full: int = np.size(z_lay_full)
        nlev_full: int = np.size(z_lev_full)
        nz_full: int = nlay_full + nlev_full

        z_full: np.ndarray = np.empty(nz_full, dtype = z_lev_full.dtype) # [m]
        z_full[0::2] = z_lev_full
        z_full[1::2] = z_lay_full

        z_lay: np.ma.MaskedArray = z_lay_full[z_lay_full <= z_max]
        z_lev: np.ma.MaskedArray = z_lev_full[z_lev_full <= z_max]

        XX_lay: np.ma.MaskedAray
        YY_lay: np.ma.MaskedArray
        ZZ_lay: np.ma.MaskedArray
        XX_lay, YY_lay, ZZ_lay = np.meshgrid(x / 1000., y / 1000., z_lay / 1000., indexing = "ij") # (x, y, lay) [km], (x, y, lay) [km], (x, y, lay) [km]

        nlay: int = np.size(z_lay)
        nlev: int = np.size(z_lev)
        nz: int = nlay + nlev

        z: np.ndarray = np.empty(nz, dtype = z_lev.dtype) # [m]
        z[0::2] = z_lev
        z[1::2] = z_lay

        for jj in range(0, nsza):
            sza: np.float = szas[jj]
            sza_str: str = "{:04.0f}".format(sza)
            
            ## Read output file
            output_file_path: str = rte_prefix + "_" + nelemx_str + "." + rte_suffix + "." + sza_str + ".out.nc"
            nc_output: nc._netCDF4.Dataset = nc.Dataset(output_file_path)

            ## Two-Stream Absorbed Shortwave Flux
            ts_sw_flux_up: np.ma.MaskedArray = nc_output.variables["sw_flux_up"][:] # (lev, y, x); [W m^(-2)]
            ts_sw_flux_dn: np.ma.MaskedArray = nc_output.variables["sw_flux_dn"][:] # (lev, y, x); [W m^(-2)]
            
            ts_flux_abs: np.ma.MaskedArray = ((ts_sw_flux_dn[1:] + ts_sw_flux_up[:-1]) - (ts_sw_flux_dn[:-1] + ts_sw_flux_up[1:])) / np.expand_dims(z_lev_full[1:] - z_lev_full[:-1], [1, 2]) # (lay, y, x); [W m^(-3)]

            ## Ray-Tracer Absorbed Shortwave Flux
            rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
            rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]
            
            rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dir + rt_flux_abs_dif # (lay, y, x); [W m^(-3)]

            ## Two-Stream Upwelling Surface Shortwave Flux
            ts_flux_sfc_up: np.ma.MaskedArray = ts_sw_flux_up[0,...] # (y, x); [W m^(-2)]

            ## Ray-Tracer Upwelling Surface Shortwave Flux
            rt_flux_sfc_up: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_up"][:] # (y, x); [W m^(-2)]

            ## Two-Stream Upwelling Top-of-Domain Shortwave Flux
            ts_flux_tod_up: np.ma.MaskedArray = ts_sw_flux_up[-1,...] # (y, x); [W m^(-2)]

            ## Ray-Tracer Upwelling Top-of-Domain Shortwave Flux
            rt_flux_tod_up: np.ma.MaskedArray = nc_output.variables["rt_flux_tod_up"][:] # (y, x); [W m^(-2)]

            ## Relative Difference Absorbed Shortwave Flux (rd)
            rd_flux_abs: np.ma.MaskedArray = (rt_flux_abs - ts_flux_abs) / np.abs(rt_flux_abs).max() # (lay, y, x); [W m^(-3)]

            ## Root-Zonal-Mean-Square Relative Error Absorbed Shortwave Flux (rzmsre)
            rzmsre_flux_abs: np.ma.MaskedArray = np.sqrt(np.nanmean(np.power(rd_flux_abs, 2), axis = (2))) # [W m^(-3)]
            rzmsre_flux_abs: np.ma.MaskedArray = rzmsre_flux_abs[:z_lay.size,...]

            ## Root-Horizontal-Mean-Square Relative Error Absorbed Shortwave Flux (rhmsre)
            rhmsre_flux_abs: np.ma.MaskedArray = np.sqrt(np.nanmean(np.power(rd_flux_abs, 2), axis = (1, 2))) # (lay); [W m^(-3)]
            rhmsre_flux_abs: np.ma.MaskedArray = rhmsre_flux_abs[:z_lay.size]

            ## Root-Mean-Square Relative Error Absorbed Shortwave Flux (rmsre)
            rmsre_flux_abs: np.ma.MaskedArray = np.sqrt(np.nanmean(np.power(rd_flux_abs, 2), axis = (0, 1, 2))) # [W m^(-3)]

            ## Fill in dicts
            x_dict[(sza, nelemx)] = x
            y_dict[(sza, nelemx)] = y
            z_lev_full_dict[(sza, nelemx)] = z_lev_full
            z_lay_full_dict[(sza, nelemx)] = z_lay_full

            z_full_dict[(sza, nelemx)] = z_full

            z_lev_dict[(sza, nelemx)] = z_lev
            z_lay_dict[(sza, nelemx)] = z_lay
            z_dict[(sza, nelemx)] = z

            XX_lay_dict[(sza, nelemx)] = XX_lay
            YY_lay_dict[(sza, nelemx)] = YY_lay
            ZZ_lay_dict[(sza, nelemx)] = ZZ_lay

            hres_dict[(sza, nelemx)] = hres

            ts_flux_abs_dict[(sza, nelemx)] = ts_flux_abs
            rt_flux_abs_dict[(sza, nelemx)] = rt_flux_abs

            ts_flux_sfc_up_dict[(sza, nelemx)] = ts_flux_sfc_up
            rt_flux_sfc_up_dict[(sza, nelemx)] = rt_flux_sfc_up

            ts_flux_tod_up_dict[(sza, nelemx)] = ts_flux_tod_up
            rt_flux_tod_up_dict[(sza, nelemx)] = rt_flux_tod_up

            rzmsre_flux_abs_dict[(sza, nelemx)] = rzmsre_flux_abs
            rhmsre_flux_abs_dict[(sza, nelemx)] = rhmsre_flux_abs
            rmsre_flux_abs_dict[(sza, nelemx)] = rmsre_flux_abs

    ## flux_sfc_up - Normalized
    xdata_grid: tuple[tuple[tuple[np.ma.MaskedArray]]] = [[[[] for ii in range(0, nnelemx)] for jj in range(0, nsza)]]
    ydata_grid: tuple[tuple[tuple[np.ma.MaskedArray]]] = [[[[] for ii in range(0, nnelemx)] for jj in range(0, nsza)]]
    subplot_label_grid: tuple[tuple[str]] = [[[] for jj in range(0, nsza)]]
    data_labels_grid: tuple[tuple[tuple[str]]] = [[[[] for ii in range(0, nnelemx)] for jj in range(0, nsza)]]
    for jj in range(0, nsza):
        sza: np_float = szas[jj]

        subplot_label_grid[0][jj] = r"${:.1f}^\circ$".format(sza)

        ## Store the maximum to normalize
        flux_sfc_up_min: np_float =  np_INF
        flux_sfc_up_max: np_float = -np_INF

        for ii in range(0, nnelemx):
            nelemx: np_float = nelemxs[ii]

            flux_sfc_up_min: np_float = min(flux_sfc_up_min,
                ts_flux_sfc_up_dict[(sza, nelemx)].min(),
                rt_flux_sfc_up_dict[(sza, nelemx)].min())
            flux_sfc_up_max: np_float = max(flux_sfc_up_max,
                ts_flux_sfc_up_dict[(sza, nelemx)].max(),
                rt_flux_sfc_up_dict[(sza, nelemx)].max())

        for ii in range(0, nnelemx):
            nelemx: np_float = nelemxs[-1 - ii]
            hres: np_float = hress[-1 - ii]

            ts_flux_sfc_up: np.ma.MaskedArray = ts_flux_sfc_up_dict[(sza, nelemx)].flatten()
            rt_flux_sfc_up: np.ma.MaskedArray = rt_flux_sfc_up_dict[(sza, nelemx)].flatten()

            xdata_grid[0][jj][ii] = (ts_flux_sfc_up - flux_sfc_up_min) / (flux_sfc_up_max - flux_sfc_up_min)
            ydata_grid[0][jj][ii] = (rt_flux_sfc_up - flux_sfc_up_min) / (flux_sfc_up_max - flux_sfc_up_min)

            data_labels_grid[0][jj][ii] = r"${:.2f}$ km".format(hres)

    file_name: str = "flux_sfc_up.png"

    xdata_grid: tuple[tuple[np.ma.MaskedArray]] = xdata_grid
    ydata_grid: tuple[tuple[np.ma.MaskedArray]] = ydata_grid
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    title: str = r"Normalized Upwelling Shortwave Surface Flux"
    xlabel: str = r"Two-Stream"
    ylabel: str = r"Ray Tracer"
    subplot_label_grid: tuple[tuple[str]] = subplot_label_grid
    data_labels_grid: tuple[tuple[tuple[str]]] = data_labels_grid
    xlim: list[float] = (0.0, 1.0)
    ylim: list[float] = (0.0, 1.0)
    show_identity: bool = True
    figsize: list = (26, 6)

    plot_scatter_grid(xdata_grid, ydata_grid, file_path, 
        title = title, xlabel = xlabel, ylabel = ylabel, 
        subplot_label_grid = subplot_label_grid,
        data_labels_grid = data_labels_grid, xlim = xlim, ylim = ylim,
        show_identity = show_identity,
        figsize = figsize)

    ## flux_tod_up - Normalized
    xdata_grid: tuple[tuple[tuple[np.ma.MaskedArray]]] = [[[[] for ii in range(0, nnelemx)] for jj in range(0, nsza)]]
    ydata_grid: tuple[tuple[tuple[np.ma.MaskedArray]]] = [[[[] for ii in range(0, nnelemx)] for jj in range(0, nsza)]]
    subplot_label_grid: tuple[tuple[str]] = [[[] for jj in range(0, nsza)]]
    data_labels_grid: tuple[tuple[tuple[str]]] = [[[[] for ii in range(0, nnelemx)] for jj in range(0, nsza)]]
    for jj in range(0, nsza):
        sza: np_float = szas[jj]

        subplot_label_grid[0][jj] = r"${:.1f}^\circ$".format(sza)

        ## Stgore the maximum to normalize
        flux_tod_up_min: np_float =  np_INF
        flux_tod_up_max: np_float = -np_INF

        for ii in range(0, nnelemx):
            nelemx: np_float = nelemxs[ii]

            flux_tod_up_min: np_float = min(flux_tod_up_min,
                ts_flux_tod_up_dict[(sza, nelemx)].min(),
                rt_flux_tod_up_dict[(sza, nelemx)].min())
            flux_tod_up_max: np_float = max(flux_tod_up_max,
                ts_flux_tod_up_dict[(sza, nelemx)].max(),
                rt_flux_tod_up_dict[(sza, nelemx)].max())

        for ii in range(0, nnelemx):
            nelemx: np_float = nelemxs[-1 - ii]
            hres: np_float = hress[-1 - ii]

            ts_flux_tod_up: np.ma.MaskedArray = ts_flux_tod_up_dict[(sza, nelemx)].flatten()
            rt_flux_tod_up: np.ma.MaskedArray = rt_flux_tod_up_dict[(sza, nelemx)].flatten()

            xdata_grid[0][jj][ii] = (ts_flux_tod_up - flux_tod_up_min) / (flux_tod_up_max - flux_tod_up_min)
            ydata_grid[0][jj][ii] = (rt_flux_tod_up - flux_tod_up_min) / (flux_tod_up_max - flux_tod_up_min)

            data_labels_grid[0][jj][ii] = r"${:.2f}$ km".format(hres)

    file_name: str = "flux_tod_up.png"

    xdata_grid: tuple[tuple[np.ma.MaskedArray]] = xdata_grid
    ydata_grid: tuple[tuple[np.ma.MaskedArray]] = ydata_grid
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    title: str = r"Normalized Upwelling Shortwave Top-of-Domain Flux"
    xlabel: str = r"Two-Stream"
    ylabel: str = r"Ray Tracer"
    subplot_label_grid: tuple[tuple[str]] = subplot_label_grid
    data_labels_grid: tuple[tuple[tuple[str]]] = data_labels_grid
    xlim: list[float] = (0.0, 1.0)
    ylim: list[float] = (0.0, 1.0)
    show_identity: bool = True
    figsize: list = (26, 6)

    plot_scatter_grid(xdata_grid, ydata_grid, file_path, 
        title = title, xlabel = xlabel, ylabel = ylabel, 
        subplot_label_grid = subplot_label_grid,
        data_labels_grid = data_labels_grid, xlim = xlim, ylim = ylim,
        show_identity = show_identity,
        figsize = figsize)

    ## RZMSRE
    meshgrid_grid: tuple[tuple[np.ma.MaskedArray]] = [[[] for jj in range(0, nsza)] for ii in range(0, nnelemx)]
    profile_grid: tuple[tuple[np.ma.MaskedArray]] = [[[] for jj in range(0, nsza)] for ii in range(0, nnelemx)]
    profile_label_grid: tuple[tuple[str]] = [[[] for jj in range(0, nsza)] for ii in range(0, nnelemx)]
    cmax: np_float = -np_INF
    for ii in range(0, nnelemx):
        nelemx: np_float = nelemxs[ii]
        hres: np_float = hress[ii]
        for jj in range(0, nsza):
            sza: np_float = szas[jj]

            meshgrid_grid[ii][jj] = (YY_lay_dict[(sza, nelemx)][0,...], ZZ_lay_dict[(sza, nelemx)][0,...])
            profile_grid[ii][jj] = np.transpose(rzmsre_flux_abs_dict[(sza, nelemx)], axes = (1, 0))

            profile_label_grid[ii][jj] = r"${:.2f}$ km - ${:.1f}^\circ$".format(hres, sza)

            cmax: np_float = max(cmax, np.abs(profile_grid[ii][jj]).max())
    cmin: np_float = 0.0

    file_name: str = "rzmsre_grid.png"

    meshgrid_grid: list = meshgrid_grid
    profile_grid: list = profile_grid
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    title: str = "Root-Zonal-Mean-Square Relative Error"
    xlabel: str = r"$y$ [km]"
    ylabel: str = r"$z$ [km]"
    cbarlabel: str = r"$(RT - TS) / max(|RT|)$"
    profile_label_grid: list = profile_label_grid
    cmin: np_float = cmin
    cmax: np_float = cmax
    cmap: str = "Reds"
    figsize: list = (13, 6.5)

    plot_profile_2d_grid(meshgrid_grid, profile_grid, file_path, 
        title = title, xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
        profile_label_grid = profile_label_grid, cmin = cmin, cmax = cmax, 
        cmap = cmap, figsize = figsize)

    ## RHMSRE
    coord_grid: tuple[np.ma.MaskedArray] = [[[] for ii in range(0, nnelemx)]]
    profiles_grid: list[list[np.ma.MaskedArray]] = [[[[] for jj in range(0, nsza)] for ii in range(0, nnelemx)]]
    profile_labels_grid: list[list[str]] = [[[[] for jj in range(0, nsza)] for ii in range(0, nnelemx)]]
    title_grid: tuple[np.ma.MaskedArray] = [[[] for ii in range(0, nnelemx)]]
    for ii in range(0, nnelemx):
        nelemx: np_float = nelemxs[ii]
        hres: np_float = hress[ii]

        coord_grid[0][ii] = z_lay / 1000. # [km]
        title_grid[0][ii] = (r"${:.2f}$ km").format(hres)

        for jj in range(0, nsza):
            sza: np_float = szas[jj]

            profiles_grid[0][ii][jj] = rhmsre_flux_abs_dict[(sza, nelemx)]

            profile_labels_grid[0][ii][jj] = (r"${:.1f}^\circ$").format(sza)

    file_name: str = "rhmsre_grid.png"

    coord_grid: list = coord_grid
    profiles_grid: list = profiles_grid
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    title: str = "Root-Horizontal-Mean-Square Relative Error"
    xlabel: str = r"$(RT - TS) / max(|RT|)$"
    ylabel: str = r"$z$ [km]"
    profile_labels_grid: list = profile_labels_grid
    title_grid: tuple[str] = title_grid
    coord_axis: str = "y"

    plot_profiles_1d_grid(coord_grid, profiles_grid, file_path, 
        title = title, xlabel = xlabel, ylabel = ylabel, 
        profile_labels_grid = profile_labels_grid, title_grid = title_grid,
        coord_axis = coord_axis)

    ## RMSRE - horizontal axis is SZA
    profiles: list[np.ndarray] = [np.zeros(nsza, dtype = np_float) for ii in range(0, nnelemx)]
    profile_labels: list[str] = ["" for ii in range(0, nnelemx)]
    for ii in range(0, nnelemx):
        nelemx: np_float = nelemxs[ii]

        hres: np_float = hress[ii]
        hres_str: str = "{:.2f} km".format(hres)
        profile_labels[ii] = hres_str

        for jj in range(0, nsza):
            sza: np_float = szas[jj]

            profiles[ii][jj] = rmsre_flux_abs_dict[(sza, nelemx)]

    file_name: str = "rmsre_sza_hres.png"

    coord: np.ndarray = szas # (nsza)
    profiles: list[np.ndarray] = profiles
    profile_labels: list[str] = profile_labels
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    title: str = "Root-Mean-Square Relative Error"
    xlabel: str = r"Solar Zenith Angle $(^{\circ})$"
    ylabel: str = r"$(RT - TS) / max(|RT|)$"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel)

    ## RMSRE - horizontal axis is horizontal resolution
    profiles: list[np.ndarray] = [np.zeros(nnelemx, dtype = np_float) for jj in range(0, nsza)]
    profile_labels: list[str] = ["" for jj in range(0, nsza)]
    for jj in range(0, nsza):
        sza: np_float = szas[jj]
        
        sza_str: str = r"${:.1f}^\circ$".format(sza)
        profile_labels[jj] = sza_str

        for ii in range(0, nnelemx):
            nelemx: np_float = nelemxs[ii]

            profiles[jj][ii] = rmsre_flux_abs_dict[(sza, nelemx)]

    file_name: str = "rmsre_hres_sza.png"

    coord: np.ndarray = hress # (nhres)
    profiles: list[np.ndarray] = profiles
    profile_labels: list[str] = profile_labels
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    title: str = "Root-Mean-Square Relative Error"
    xlabel: str = r"Horizontal Resolution $(km)$"
    ylabel: str = r"$(RT - TS) / max(|RT|)$"

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
                     title = title, xlabel = xlabel, ylabel = ylabel)
    
if __name__ == "__main__":
    main()

