# Standard Library Imports
import argparse
import os

# Third-Party Library Imports
import numpy as np
import netCDF4 as nc

# Local Library Imports
from plot_profiles import plot_profiles_1d, plot_profile_2d
from consts import np_EPS

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_output",
        description = "Plots comparisons of the two stream and ray tracer solvers of of RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Path to RTE-RRTMGP-CPP input file.")
    
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
                        default = ["comparison"],
                        help = "Path to output generated plots.")
    
    args: argparse.Namespace = parser.parse_args()

    input_file_path: str = os.path.normpath(args.input[0])
    output_file_path: str = os.path.normpath(args.output[0])
    optics_file_path: str = os.path.normpath(args.optics[0])
    out_dir_path: str = os.path.normpath(args.outdir[0])

    ## Load the input, output, and optics data
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)
    nc_output: nc._netCDF4.Dataset = nc.Dataset(output_file_path)
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file_path)

    ## Create the output directories
    out_dir_path: str = os.path.join(os.getcwd(), out_dir_path)

    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

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

    # COMPARE: ABSORBED SHORTWAVE FLUX [W m^(-3)]
    ## Calculate the absorbed shortwave fluxes profiles (TwoStream Solver)
    ts_flux_up: np.ma.MaskedArray = nc_output.variables["sw_flux_up"][:] # (lev, y, x); [W m^(-2)]
    ts_flux_dn: np.ma.MaskedArray = nc_output.variables["sw_flux_dn"][:] # (lev, y, x); [W m^(-2)]

    ts_flux_abs: np.ma.MaskedArray = ((ts_flux_dn[1:] + ts_flux_up[:-1]) - (ts_flux_dn[:-1] + ts_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (lay, y, x); [W m^(-3)]
    ### Filter out values that are super small
    ts_flux_abs[np.abs(ts_flux_abs) <= np_EPS] = 0.

    ## Get the absorbed shortwave fluxes profiles (Monte Carlo Ray Tracer)
    rt_flux_abs_dir: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dir"][:] # (lay, y, x); [W m^(-3)]
    rt_flux_abs_dif: np.ma.MaskedArray = nc_output.variables["rt_flux_abs_dif"][:] # (lay, y, x); [W m^(-3)]

    rt_flux_abs: np.ma.MaskedArray = rt_flux_abs_dir + rt_flux_abs_dif # (lay, y, x); [W m^(-3)]
    ### Filter out values that are super small
    rt_flux_abs[np.abs(rt_flux_abs) <= np_EPS] = 0.

    ## Plot the relative difference of absorbed shortwave fluxes in each layer
    if (np.max(np.abs(rt_flux_abs)) > 0.):
        flux_abs_diff: np.ma.MaskedArray = (ts_flux_abs - rt_flux_abs) / np.max(np.abs(rt_flux_abs)) # (lay, x, y)
        flux_abs_diff_z: np.ndarray = np.nanmean(flux_abs_diff, axis = (1, 2)) # (lay)
        title: str = r"Relative Difference: $\left( TS - MC \right) / max\left( |MC| \right)$"
        xlabel: str = r"Absorbed Shortwave Flux"
    else:
        flux_abs_diff: np.ma.MaskedArray = (ts_flux_abs - rt_flux_abs) # (lay, x, y); [W m^(-3)]
        flux_abs_diff_z: np.ndarray = np.nanmean(flux_abs_diff, axis = (1, 2)) # (lay); [W m^(-3)]
        title: str = r"Absolute Difference: $\left( TS - MC \right)$"
        xlabel: str = r"Absorbed Shortwave Flux $[W m^{-3}]$"

    coord: np.ndarray = z_lay / 1000. # (lay); [km]
    profiles: list = [flux_abs_diff_z]
    file_path: str = os.path.join(out_dir_path, "flux_abs.png")
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"
    viz: str = "difference"

    plot_profiles_1d(coord, profiles, file_path, title = title, xlabel = xlabel,
                     ylabel = ylabel, coord_axis = coord_axis, viz = viz)

    # COMPARE: UPWELLING SHORTWAVE SURFACE FLUXES [W m^(-2)]
    ts_flux_sfc_up: np.ma.MaskedArray = ts_flux_up[0, ...] # (y, x); [W m^(-2)]
    rt_flux_sfc_up: np.ma.MaskedArray = nc_output.variables["rt_flux_sfc_up"][:] # (y, x); [W m^(-2)]

    ## Filter out values that are super small
    ts_flux_sfc_up[np.abs(ts_flux_sfc_up) <= np_EPS] = 0.
    rt_flux_sfc_up[np.abs(rt_flux_sfc_up) <= np_EPS] = 0.

    ## Plot the relative difference of upwelling shortwave surface flux
    if (np.max(np.abs(rt_flux_sfc_up)) > 0.):
        flux_sfc_up_diff: np.ma.MaskedArray = (ts_flux_sfc_up - rt_flux_sfc_up) / np.max(np.abs(rt_flux_sfc_up)) # (x, y)
        title: str = r"Relative Difference: $\left( TS - MC \right) / max\left( |MC| \right)$"
        cbarlabel: str = r"Upwelling Shortwave Surface Flux"
    else:
        flux_sfc_up_diff: np.ma.MaskedArray = (ts_flux_sfc_up - rt_flux_sfc_up) # (x, y); [W m^(-2)]
        title: str = r"Absolute Difference: $\left( TS - MC \right)$"
        cbarlabel: str = r"Upwelling Shortwave Surface Flux $[W m^{-3}]$"

    meshgrid: tuple = [XX / 1000., YY / 1000.]
    profile: np.ndarray = np.transpose(flux_sfc_up_diff, axes = (1, 0))
    file_path: str = os.path.join(out_dir_path, "flux_sfc_up.png")
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cmap: str = "bwr"
    cscale: str = "difference"

    plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
                    ylabel = ylabel, cbarlabel = cbarlabel, cmap = cmap,
                    cscale = cscale)

    # COMPARE: UPWELLING SHORTWAVE TOP-OF-DOMAIN FLUXES [W m^(-2)]
    ts_flux_tod_up: np.ma.MaskedArray = ts_flux_up[-1, ...] # (y, x); [W m^(-2)]
    rt_flux_tod_up: np.ma.MaskedArray = nc_output.variables["rt_flux_tod_up"][:] # (y, x); [W m^(-2)]

    ## Filter out values that are super small
    ts_flux_tod_up[np.abs(ts_flux_tod_up) <= np_EPS] = 0.
    rt_flux_tod_up[np.abs(rt_flux_tod_up) <= np_EPS] = 0.

    ## Plot the relative difference of upwelling shortwave top-of-domain flux
    if (np.max(np.abs(rt_flux_tod_up)) > 0.):
        flux_tod_up_diff: np.ma.MaskedArray = (ts_flux_tod_up - rt_flux_tod_up) / np.max(np.abs(rt_flux_tod_up)) # (x, y)
        title: str = r"Relative Difference: $\left( TS - MC \right) / max\left( |MC| \right)$"
        cbarlabel: str = r"Upwelling Shortwave Top-of-Domain Flux"
    else:
        flux_tod_up_diff: np.ma.MaskedArray = (ts_flux_tod_up - rt_flux_tod_up) # (x, y); [W m^(-2)]
        title: str = r"Absolute Difference: $\left( TS - MC \right)$"
        cbarlabel: str = r"Upwelling Shortwave Top-of-Domain Flux $[W m^{-3}]$"

    meshgrid: tuple = [XX / 1000., YY / 1000.]
    profile: np.ndarray = np.transpose(flux_tod_up_diff, axes = (1, 0))
    file_path: str = os.path.join(out_dir_path, "flux_tod_up.png")
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cmap: str = "bwr"
    cscale: str = "difference"

    plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
                    ylabel = ylabel, cbarlabel = cbarlabel, cmap = cmap,
                    cscale = cscale)


if __name__ == "__main__":
    main()
