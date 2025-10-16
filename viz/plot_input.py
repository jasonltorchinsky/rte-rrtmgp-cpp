# Standard Library Imports
import argparse
import os

# Third-Party Library Imports
import netCDF4 as nc
import numpy as np

# Local Library Imports
from plot_tools import plot_profiles_1d, plot_profile_2d, plot_profile_3d, plot_distribution

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_input",
        description = "Plots the input of RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input",
                        action = "store",
                        nargs = 1,
                        type = str,
                        required = True,
                        help = "Path to RTE-RRTMGP-CPP input file.")
    
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
                        default = ["input"],
                        help = "Path to output generated plots.")
    
    args: argparse.Namespace = parser.parse_args()

    input_file_path: str = os.path.normpath(args.input[0])
    optics_file_path: str = os.path.normpath(args.optics[0])
    out_dir_path: str = os.path.normpath(args.outdir[0])

    ## Load the input and optics data
    nc_input: nc._netCDF4.Dataset = nc.Dataset(input_file_path)
    nc_optics: nc._netCDF4.Dataset = nc.Dataset(optics_file_path)

    ## Create the output directories
    out_dir_path: str = os.path.join(os.getcwd(), out_dir_path)

    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    plot_pressure(nc_input, out_dir_path)
    plot_temperature(nc_input, out_dir_path)
    plot_vmr(nc_input, out_dir_path)
    plot_lwp(nc_input, out_dir_path)
    plot_rel(nc_input, out_dir_path)
    plot_iwp(nc_input, out_dir_path)
    plot_dei(nc_input, out_dir_path)
        

def plot_pressure(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # (y); [m]

    nx: int = np.size(x)
    ny: int = np.size(y)

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]
    z_lev: np.ma.MaskedArray = nc_input.variables["z_lev"][:] # (lev); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]
    z_lev: np.ma.MaskedArray = z_lev / 1000. # (lev); [km]

    nlay: int = np.size(z_lay)
    nlev: int = np.size(z_lev)
    nz: int = nlay + nlev

    z: np.ndarray = np.empty(nz, dtype = z_lay.dtype) # (nz); [km]
    z[0::2] = z_lev
    z[1::2] = z_lay

    # Obtain horizontally-averaged pressure profile
    p_lay: np.ma.MaskedArray = nc_input.variables["p_lay"][:] # (lay, y, x); [Pa]
    p_lev: np.ma.MaskedArray = nc_input.variables["p_lev"][:] # (lev, y, x); [Pa]

    p: np.ndarray = np.empty([nz, ny, nx], dtype = p_lay.dtype) # (z, y, x); [Pa]
    p[0::2,...] = p_lev
    p[1::2,...] = p_lay

    p_z: np.ndarray = np.nanmean(p, axis = (1, 2)) # (z); [Pa]

    ## Plot horizontally-averaged pressure profile
    file_name: str = "pressure_z.png"

    coord: np.ndarray = z
    profiles: list = [p_z]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Pressure $[Pa]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, 
        xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis)

def plot_temperature(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]
    z_lev: np.ma.MaskedArray = nc_input.variables["z_lev"][:] # (lev); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]
    z_lev: np.ma.MaskedArray = z_lev / 1000. # (lev); [km]

    nlay: int = np.size(z_lay)
    nlev: int = np.size(z_lev)
    nz: int = nlay + nlev

    z: np.ndarray = np.empty(nz, dtype = z_lay.dtype) # (nz); [km]
    z[0::2] = z_lev
    z[1::2] = z_lay

    # Obtain horizontally-averaged temperature profile
    t_lay: np.ma.MaskedArray = nc_input.variables["t_lay"][:] # (lay, y, x); [K]
    t_lev: np.ma.MaskedArray = nc_input.variables["t_lev"][:] # (lev, y, x); [K]

    t: np.ndarray = np.empty([nz, ny, nx], dtype = t_lev.dtype) # (nz); [K]
    t[0::2,...] = t_lev
    t[1::2,...] = t_lay

    t_z: np.ndarray = np.nanmean(t, axis = (1, 2)) # (nz); [K]

    ## Plot horizontally-averaged temperature profile
    file_name: str = "temperature_z.png"

    coord: np.ndarray = z
    profiles: list = [t_z]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Temperature $[K]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, 
        xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis)

    # Obtain surface temperature profile
    t_sfc: np.ma.MaskedArray = nc_input.variables["t_sfc"][:] # (y, x); [K]

    t_sfc: np.ndarray = np.transpose(t_sfc, axes = (1, 0)) # (y, x); [K]

    ## Plot surface temperature profile
    file_name: str = "temperature_sfc.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = t_sfc
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Surface Temperature $[K]$"
    cmin: float = t_sfc.min()
    cmax: float = t_sfc.max()

    if (cmax > cmin):
        plot_profile_2d(meshgrid, profile, file_path, xlabel = xlabel,
                        ylabel = ylabel, cbarlabel = cbarlabel, cmin = cmin, cmax = cmax)


def plot_vmr(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the horizontally-averaged VMR profiles
    gas_codes: list = ["ch4", "co", "co2", "h2o", "n2", "n2o", "o2", "o3", 
                       "ccl4", "cfc11", "cfc12", "cfc22", "hfc143a", "hfc125",
                       "hfc23", "hfc32", "hfc134a", "cf4", "no2"]
    gas_names: list = [r"$C H_4$", r"$C O$", r"$C O_2$", r"$H_2 O$", r"$N_2$", 
                       r"$N_2 O$", r"$O_2$", r"$O_3$", r"$C Cl_4", r"$CFC-11$",
                       r"$CFC-12$", r"$CFC-22$", r"$HFC-143a$", r"$HFC-125$",
                       r"$HFC-23$", r"$HFC-32$", r"$HFC-134a$", r"$C F_4$", r"$N O_2$"]
    ngases: int = len(gas_codes)
    profiles: list = []
    profile_labels: list = []
    
    for ii in range(0, ngases):
        if ("vmr_" + gas_codes[ii]) in nc_input.variables.keys():
            vmr: np.ma.MaskedArray = nc_input.variables["vmr_" + gas_codes[ii]][:]
            assert(vmr.min() >= 0.0)
            assert(vmr.max() <= 1.0)
            if vmr.max() > 0.: # If non-zero, then plot it
                if vmr.ndim == 0: # Constant across domain
                    vmr_z: np.ma.MaskedArray = vmr * np.ones((nlay)) # (lay); [N/A]
                elif vmr.ndim == 1: # Constant across domain
                    vmr_z: np.ma.MaskedArray = np.tile(vmr, (nlay)) # (lay); [N/A]
                elif vmr.ndim == 3:
                    vmr_z: np.ndarray = np.nanmean(vmr, axis = (1, 2)) # (lay); [N/A]

                profiles.append(vmr_z)
                profile_labels.append(gas_names[ii])

    ## Plot horizontally-averaged VMR profiles
    file_name: str = "vmr_z.png"

    coord: np.ndarray = z_lay
    profiles: list = profiles
    profile_labels: list = profile_labels
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Volume Mixing Ratio"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"
    xscale: str = "log"

    plot_profiles_1d(coord, profiles, file_path, 
        profile_labels = profile_labels, xlabel = xlabel, ylabel = ylabel,
        coord_axis = coord_axis, xscale = xscale)

def plot_lwp(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged lwp profiles
    lwp: np.ma.MaskedArray = nc_input.variables["lwp"][:] # (lay, y, x); [kg m^(-2)]

    lwp_xy: np.ma.MaskedArray = np.sum(lwp, axis = (0)) # (y, x); [kg m^(-2)]
    lwp_xy: np.ma.MaskedArray = np.transpose(lwp_xy, axes = (1, 0)) # (x, y); [kg m^(-2)]

    lwp_lay: np.ma.MaskedArray = np.nanmean(lwp, axis = (1, 2)) # (lay); [kg m^(-2)]

    ## Plot vertically-integrated lwp profile
    file_name: str = "lwp_xy.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = lwp_xy
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Vertically-Integrated Liquid Water Path $[kg\,m^{-2}]$"
    cmin: float = 0.0
    cmax: float = lwp_xy.max()
    cmap: str = "Blues"
    cscale: float = "normal"

    if (lwp_xy.max() > lwp_xy.min()):
        plot_profile_2d(meshgrid, profile, file_path, 
            xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
            cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale)

    ## Plot horizontally-averaged lwp profile
    file_name: str = "lwp_z.png"

    coord: np.ndarray = z_lay
    profiles: list = [lwp_lay]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Liquid Water Path $[kg\,m^{-2}]$"
    ylabel: str = r"z $[km]$"
    xscale: str = "linear"
    coord_axis: str = "y"

    if (lwp_lay.max() > lwp_lay.min()):
        plot_profiles_1d(coord, profiles, file_path, 
            xlabel = xlabel, ylabel = ylabel, xscale = xscale,
            coord_axis = coord_axis)

def plot_rel(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged rel profiles
    rel: np.ma.MaskedArray = nc_input.variables["rel"][:] # (lay, y, x); [μm]

    rel_xy: np.ma.MaskedArray = np.sum(rel, axis = (0)) # (y, x); [μm]
    rel_xy: np.ma.MaskedArray = np.transpose(rel_xy, axes = (1, 0)) # (x, y); [μm]

    rel_lay: np.ma.MaskedArray = np.nanmean(rel, axis = (1, 2)) # (lay); [μm]

    ## Plot vertically-integrated rel profile
    file_name: str = "rel_xy.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = rel_xy
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Vertically-Integrated Liquid Water Effective Radius [$\mu m$]"
    cmin: float = rel_xy.min()
    cmax: float = rel_xy.max()
    cmap: str = "winter_r"
    cscale: float = "normal"

    if (rel_xy.max() > rel_xy.min()):
        plot_profile_2d(meshgrid, profile, file_path, 
            xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
            cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale)

    ## Plot horizontally-averaged rel profile
    file_name: str = "rel_z.png"

    coord: np.ndarray = z_lay
    profiles: list = [rel_lay]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Liquid Water Effective Radius [$\mu m$]"
    ylabel: str = r"z $[km]$"
    xscale: str = "linear"
    coord_axis: str = "y"

    if (rel_lay.max() > rel_lay.min()):
        plot_profiles_1d(coord, profiles, file_path, 
            xlabel = xlabel, ylabel = ylabel, xscale = xscale,
            coord_axis = coord_axis)

    ## Plot rel distribution
    file_name: str = "rel_dist.png"

    profile: np.ndarray = rel
    file_path: str = os.path.join(out_dir_path, file_name)
    nbins: int = 256
    xlabel: str = r"Liquid Water Effective Radius [$\mu m$]"
    ylabel: str = "Counts"
    title: str = r"Liquid Water Effective Radius Distribution"
    xscale: str = "linear"
    yscale: str = "linear"

    if (rel.max() > rel.min()):
        plot_distribution(profile, file_path, nbins = nbins, title = title,
            xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale)

def plot_iwp(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged iwp profiles
    iwp: np.ma.MaskedArray = nc_input.variables["iwp"][:] # (lay, y, x); [kg m^(-2)]

    iwp_xy: np.ma.MaskedArray = np.sum(iwp, axis = (0)) # (y, x); [kg m^(-2)]
    iwp_xy: np.ma.MaskedArray = np.transpose(iwp_xy, axes = (1, 0)) # (x, y); [kg m^(-2)]

    iwp_lay: np.ma.MaskedArray = np.nanmean(iwp, axis = (1, 2)) # (lay); [kg m^(-2)]

    ## Plot vertically-integrated iwp profile
    file_name: str = "iwp_xy.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = iwp_xy
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Vertically-Integrated Ice Water Path $[kg\,m^{-2}]$"
    cmin: float = 0.0
    cmax: float = iwp_xy.max()
    cmap: str = "Purples"
    cscale: float = "normal"

    if (iwp_xy.max() > iwp_xy.min()):
        plot_profile_2d(meshgrid, profile, file_path, 
            xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
            cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale)

    ## Plot horizontally-averaged iwp profile
    file_name: str = "iwp_z.png"

    coord: np.ndarray = z_lay
    profiles: list = [iwp_lay]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Ice Water Path $[kg\,m^{-2}]$"
    ylabel: str = r"z $[km]$"
    xscale: str = "linear"
    coord_axis: str = "y"

    if (iwp_lay.max() > iwp_lay.min()):
        plot_profiles_1d(coord, profiles, file_path, 
            xlabel = xlabel, ylabel = ylabel, xscale = xscale,
            coord_axis = coord_axis)

def plot_dei(nc_input: nc._netCDF4.Dataset, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = nc_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = nc_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = nc_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged dei profiles
    dei: np.ma.MaskedArray = nc_input.variables["dei"][:] # (lay, y, x); [μm]

    dei_xy: np.ma.MaskedArray = np.sum(dei, axis = (0)) # (y, x); [μm]
    dei_xy: np.ma.MaskedArray = np.transpose(dei_xy, axes = (1, 0)) # (x, y); [μm]

    dei_lay: np.ma.MaskedArray = np.nanmean(dei, axis = (1, 2)) # (lay); [μm]

    ## Plot vertically-integrated dei profile
    file_name: str = "dei_xy.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = dei_xy
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Vertically-Integrated Ice Water Effective Diameter [$\mu m$]"
    cmin: float = dei_xy.min()
    cmax: float = dei_xy.max()
    cmap: str = "summer_r"
    cscale: float = "normal"

    if (dei_xy.max() > dei_xy.min()):
        plot_profile_2d(meshgrid, profile, file_path, 
            xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
            cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale)

    ## Plot horizontally-averaged dei profile
    file_name: str = "dei_z.png"

    coord: np.ndarray = z_lay
    profiles: list = [dei_lay]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Ice Water Effective Diameter [$\mu m$]"
    ylabel: str = r"z $[km]$"
    xscale: str = "linear"
    coord_axis: str = "y"

    if (dei_lay.max() > dei_lay.min()):
        plot_profiles_1d(coord, profiles, file_path, 
            xlabel = xlabel, ylabel = ylabel, xscale = xscale,
            coord_axis = coord_axis)

    ## Plot dei distribution
    file_name: str = "dei_dist.png"

    profile: np.ndarray = dei
    file_path: str = os.path.join(out_dir_path, file_name)
    nbins: int = 256
    xlabel: str = r"Ice Water Effective Diameter [$\mu m$]"
    ylabel: str = "Counts"
    title: str = r"Ice Water Effective Diameter Distribution"
    xscale: str = "linear"
    yscale: str = "linear"

    if (dei.max() > dei.min()):
        plot_distribution(profile, file_path, nbins = nbins, title = title,
            xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale)

if __name__ == "__main__":
    main()