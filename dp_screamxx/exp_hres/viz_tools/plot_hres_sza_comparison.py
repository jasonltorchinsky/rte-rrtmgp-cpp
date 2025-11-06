# Standard Library Imports
import argparse
import ast
import os

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from consts import np_float, np_INF
from plot_profiles import plot_profiles_1d, plot_profile_2d, plot_profile_grid_2d, plot_profiles_2d_3d

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
    nxs: np.ndarray = np.array(ast.literal_eval(args.nxs[0]), dtype = np_float).flatten()
    szas: np.ndarray = np.array(ast.literal_eval(args.szas[0]), dtype = np_float).flatten()
    optics_file_path: str = os.path.normpath(args.optics[0])
    out_dir_path: str = os.path.normpath(args.outdir[0])

    ## Load the optics data
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file_path)

    ## Create the output directories
    out_dir_path: str = os.path.join(os.getcwd(), out_dir_path)
    hres_sza_dir_name: str = "hres_sza"
    hres_sza_dir_path: str = os.path.join(out_dir_path, hres_sza_dir_name)

    dir_paths: list[str] = [out_dir_path, hres_sza_dir_path]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    nnx: int = np.size(nxs)
    nsza: int = np.size(szas)

    ## Create list of arrays to plot in a grid of plots
    meshgrid_list: list = [[[] for jj in range(0, nsza)] for ii in range(0, nnx)]
    nx_list: list = [[[] for jj in range(0, nsza)] for ii in range(0, nnx)]
    hres_list: list = [[[] for jj in range(0, nsza)] for ii in range(0, nnx)]
    sza_list: list = [[[] for jj in range(0, nsza)] for ii in range(0, nnx)]
    rt_flux_abs_list: list = [[[] for jj in range(0, nsza)] for ii in range(0, nnx)]
    profile_label_list: list = [[[] for jj in range(0, nsza)] for ii in range(0, nnx)]
    rt_flux_abs_min: np_float
    rt_flux_abs_max: np_float
    [rt_flux_abs_min, rt_flux_abs_max] = [np.float64(np_INF), -np.float64(np_INF)]

    ## Set maximum elevation for plots
    z_max: np_float = 15000. # [m]

    for ii in range(0, nnx):
        nx: np_float = nxs[ii]
        hres: np_float = 50 / (nx * 2) # Domain size of simulation is 50 km X 50 km; resolution is x / (2 * nx)

        nx_str: str = "{:02.0f}x{:02.0f}".format(nx, nx)
        hres_str: str = "{:02.0f} km".format(hres)

        ## Input file is same for all nxs, choose a dummy SZA to open file and read it
        dummy_sza: np_float = szas[0]
        dummy_sza_str: str = "{:04.0f}".format(dummy_sza)

        ## Read input file
        input_file_path: str = rte_prefix + "_" + nx_str + "." + rte_suffix + "." + dummy_sza_str + ".in.nc"
        nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)

        ## Extract the spatial variables
        x: np.ma.MaskedArray = nc_input.variables["x"][:] # [m]
        y: np.ma.MaskedArray = nc_input.variables["y"][:] # [m]

        z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # [m]
        z_lev: np.ma.MaskedArray = nc_input.variables["z_lev"][:] # [m]

        z_lay: np.ma.MaskedArray = z_lay[z_lay <= z_max]
        z_lev: np.ma.MaskedArray = z_lev[z_lev <= z_max]

        YY_lay: np.ndarray
        ZZ_lay: np.ndarray
        YY_lay, ZZ_lay = np.meshgrid(y / 1000., z_lay / 1000., indexing = "ij") # (y, lay) [km], (y, lay) [km]

        nx: int = np.size(x)
        ny: int = np.size(y)

        nlay: int = np.size(z_lay)
        nlev: int = np.size(z_lev)
        nz: int = nlay + nlev

        z: np.ndarray = np.empty(nz, dtype = z_lev.dtype) # [m]
        z[0::2] = z_lev
        z[1::2] = z_lay

        for jj in range(0, nsza):
            sza: np.float = szas[jj]
            sza_str: str = "{:04.0f}".format(sza)

            ## Fill in meshgrid_list, nx_list, sza_list
            meshgrid_list[ii][jj] = [YY_lay, ZZ_lay]
            nx_list[ii][jj] = nx
            sza_list[ii][jj] = sza
            
            ## Read output file
            output_file_path: str = rte_prefix + "_" + nx_str + "." + rte_suffix + "." + sza_str + ".out.nc"
            nc_output: nc._netCDF4.Dataset = nc.Dataset(output_file_path)

            ## Ray-Tracer Absorbed Flux
            rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
            rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]
            rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dir + rt_flux_abs_dif # (lay, y, x); [W m^(-3)]

            rt_flux_abs_2d: np.ma.MaskedArray = np.nanmean(rt_flux_abs, axis = (2)) # (lay, y)
            rt_flux_abs_2d: np.ma.MaskedArray = np.transpose(rt_flux_abs_2d) # (y, lay)
            rt_flux_abs_2d: np.ma.MaskedArray = rt_flux_abs_2d[:, :z_lay.size]

            rt_flux_abs_min: np.float64 = min(rt_flux_abs_min, rt_flux_abs_2d.min())
            rt_flux_abs_max: np.float64 = max(rt_flux_abs_max, rt_flux_abs_2d.max())

            rt_flux_abs_list[ii][jj] = rt_flux_abs_2d

            ## Profile label
            profile_label_list[ii][jj] = (r"${:.2f}$ km - ${:.1f}^\circ$").format(hres, sza)

    file_name: str = "rt_flux_abs.png"

    meshgrids: list = meshgrid_list
    profiles: list = rt_flux_abs_list
    file_path: str = os.path.join(hres_sza_dir_path, file_name)
    xlabel: str = r"$y$ [km]"
    ylabel: str = r"$z$ [km]"
    cbarlabel: str = r"Absorbed Shortwave Flux $[W m^{-3}]$"
    profile_labels: list = profile_label_list
    cmin: np_float = rt_flux_abs_min
    cmax: np_float = rt_flux_abs_max
    
    plot_profile_grid_2d(meshgrids, profiles, file_path, xlabel = xlabel, ylabel = ylabel,
        cbarlabel = cbarlabel, profile_labels = profile_labels, cmin = cmin, cmax = cmax)
    
if __name__ == "__main__":
    main()

