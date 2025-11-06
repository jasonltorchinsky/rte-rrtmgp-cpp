# Append the 'exp_hres' directory to the PYTHONPATH for future imports
import os, sys
src_dir: str = os.path.normpath( \
    os.path.join(os.path.dirname(__file__), os.pardir))
if src_dir not in sys.path:
    sys.path.append(src_dir)
    
# Standard Library Imports
import argparse
import os

# Third-Party Library Imports
import netCDF4 as nc
import numpy as np
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET
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
    xr_input: XR_DATASET = xr.open_dataset(input_file_path,
        engine = "netcdf4", decode_timedelta = False)
    xr_optics: XR_DATASET = xr.open_dataset(optics_file_path,
        engine = "netcdf4", decode_timedelta = False)

    ## Create the output directories
    out_dir_path: str = os.path.join(os.getcwd(), out_dir_path)

    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    #plot_pressure(xr_input, out_dir_path)
    #plot_temperature(xr_input, out_dir_path)
    #plot_vmr(xr_input, out_dir_path)
    #plot_lwp(xr_input, out_dir_path)
    #plot_rel(xr_input, out_dir_path)
    #plot_iwp(xr_input, out_dir_path)
    #plot_dei(xr_input, out_dir_path)

    horz_average_kwargs: dict = {
        "p" : {"file_name" : "p_havg.png",
            "xlabel" : r"Horizontally-Averaged Pressure $[Pa]$",
            "xscale" : "linear"},
        "t" : {"file_name" : "t_havg.png",
            "xlabel" : r"Horizontally-Averaged Temperature $[K]$",
            "xscale" : "linear"},
        "vmr" : {"file_name" : "vmr_havg.png",
            "xlabel" : r"Horizontally-Averaged Volume Mixing Ratio",
            "gas_codes" : ["ch4", "co", "co2", "h2o", "n2", "n2o", "o2", "o3", 
                "ccl4", "cfc11", "cfc12", "cfc22", "hfc143a", "hfc125", "hfc23", 
                "hfc32", "hfc134a", "cf4", "no2"],
            "gas_names" : [r"$C H_4$", r"$C O$", r"$C O_2$", r"$H_2 O$",
                r"$N_2$", r"$N_2 O$", r"$O_2$", r"$O_3$", r"$C Cl_4",
                r"$CFC-11$", r"$CFC-12$", r"$CFC-22$", r"$HFC-143a$",
                r"$HFC-125$", r"$HFC-23$", r"$HFC-32$", r"$HFC-134a$", 
                r"$C F_4$", r"$N O_2$"],
                "xscale" : "log"},
        "lwp" : {"file_name" : "lwp_havg.png",
            "xlabel" : r"Horizontally-Averaged Liquid Water Path $[kg\,m^{-2}]$",
            "xscale" : "linear"},
        "iwp" : {"file_name" : "iwp_havg.png",
            "xlabel" : r"Horizontally-Averaged Ice Water Path $[kg\,m^{-2}]$",
            "xscale" : "linear"},
        "rel" : {"file_name" : "rel_havg.png",
            "xlabel" : r"Horizontally-Averaged Liquid Water Effective Radius [$\mu m$]",
            "xscale" : "linear"},
        "dei" : {"file_name" : "dei_havg.png",
            "xlabel" : r"Horizontally-Averaged Ice Water Effective Diameter [$\mu m$]",
            "xscale" : "linear"}
    }

    #for key, kwargs in horz_average_kwargs.items():
    #    plot_horz_average(xr_input, out_dir_path, key, kwargs)

    vert_integral_kwargs: dict = {
        "lwp" : {"file_name" : "lwp_vint.png",
            "cbarlabel" : r"Vertically-Integrated Liquid Water Path $[kg\,m^{-2}]$",
            "cmap" : "Blues"},
        "iwp" : {"file_name" : "iwp_vint.png",
            "cbarlabel" : r"Vertically-Integrated Ice Water Path $[kg\,m^{-2}]$",
            "cmap" : "Purples"},
        "rel" : {"file_name" : "rel_vint.png",
            "cbarlabel" : r"Vertically-Integrated Liquid Water Effective Radius [$\mu m$]",
            "cmap" : "Blues"},
        "dei" : {"file_name" : "dei_vint.png",
            "cbarlabel" : r"Vertically-Integrated Ice Water Effective Diameter [$\mu m$]",
            "cmap" : "Purples"}
    }

    #for key, kwargs in vert_integral_kwargs.items():
    #    plot_vert_integral(xr_input, out_dir_path, key, kwargs)

    x_integral_kwargs: dict = {
        "lwp" : {"file_name" : "lwp_xint.png",
            "cbarlabel" : r"$x$-Integrated Liquid Water Path $[kg\,m^{-2}]$",
            "cmap" : "Blues"},
        "iwp" : {"file_name" : "iwp_xint.png",
            "cbarlabel" : r"$x$-Integrated Ice Water Path $[kg\,m^{-2}]$",
            "cmap" : "Purples"},
        "rel" : {"file_name" : "rel_xint.png",
            "cbarlabel" : r"$x$-Integrated Liquid Water Effective Radius [$\mu m$]",
            "cmap" : "Blues"},
        "dei" : {"file_name" : "dei_xint.png",
            "cbarlabel" : r"$x$-Integrated Ice Water Effective Diameter [$\mu m$]",
            "cmap" : "Purples"}
    }

    for key, kwargs in x_integral_kwargs.items():
        plot_x_integral(xr_input, out_dir_path, key, kwargs)

def plot_horz_average(xr_input: XR_DATASET, out_dir_path: str, key: str, kwargs: dict) -> None:
    # Obtain horizontal and vertical grid information
    nx: NP_INT = NP_INT(xr_input.sizes["x"])
    ny: NP_INT = NP_INT(xr_input.sizes["y"])
    nlay: NP_INT = NP_INT(xr_input.sizes["z"])
    nlev: NP_INT = NP_INT(xr_input.sizes["zh"])
    nz: NP_INT = nlay + nlev

    z_lay: NP_ARRAY[NP_REAL] = xr_input["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
    z_lev: NP_ARRAY[NP_REAL] = xr_input["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

    z: NP_ARRAY[NP_REAL] = np.empty(nz, dtype = NP_REAL) # (nz); [m]
    z[0::2] = z_lev
    z[1::2] = z_lay

    # Obtain horizontally-averaged profiles
    ## Exceptions
    if key == "vmr":
        ngases: NP_INT = NP_INT(len(kwargs["gas_codes"]))
        profiles: list[NP_ARRAY[NP_REAL]] = []
        profile_labels: Optional[list[str]] = []

        for ii in range(0, ngases):
            gas_key: str = "vmr_" + kwargs["gas_codes"][ii]
            if gas_key in xr_input.keys():
                field: NP_ARRAY[NP_REAL] = xr_input[gas_key].values.astype(NP_REAL) # (lay, y, x)
                assert((field.min() >= 0.0) and (field.max() <= 1.0))
                if field.max() > 0.: # If non-zero, then plot it
                    if field.ndim == 0: # Constant across domain
                        field_z: NP_ARRAY[NP_REAL] = field * np.ones((nlay), NP_REAL) # (lay); [N/A]
                    elif field.ndim == 1: # Constant across domain
                        field_z: NP_ARRAY[NP_REAL] = np.tile(field, (nlay)) # (lay); [N/A]
                    elif field.ndim == 3:
                        field_z: NP_ARRAY[NP_REAL] = np.nanmean(field, axis = (1, 2)) # (lay); [N/A]

                    profiles.append(field_z)
                    profile_labels.append(kwargs["gas_names"][ii])

        coord: NP_ARRAY[NP_REAL] = z_lay / 1000. # (nlay); [km]

    else:
        profile_labels: Optional[list[str]] = None
        lay_key: str = key + "_lay"
        lev_key: str = key + "_lev"
        field_lay: Optional[NP_ARRAY[NP_REAL]] = None
        field_lev: Optional[NP_ARRAY[NP_REAL]] = None
        assert((key in xr_input.keys()) or
            (lay_key in xr_input.keys()) or
            (lev_key in xr_input.keys()))
        if key in xr_input.keys():
            field_lay = xr_input[key].values.astype(NP_REAL) # (lay, y, x)
        else:
            if lay_key in xr_input.keys():
                field_lay = xr_input[lay_key].values.astype(NP_REAL) # (lay, y, x)
            if lev_key in xr_input.keys():
                field_lev = xr_input[lev_key].values.astype(NP_REAL) # (lev, y, x)

        assert((field_lay is not None) or (field_lev is not None))

        if (field_lay is not None) and (field_lev is not None):
            field: NP_ARRAY[NP_REAL] = np.empty([nz, ny, nx], dtype = NP_REAL) # (z, y, x)
            field[0::2,...] = field_lev
            field[1::2,...] = field_lay

            coord: NP_ARRAY[NP_REAL] = z / 1000. # (nz); [km]
        elif (field_lay is not None) and (field_lev is None):
            field: NP_ARRAY[NP_REAL] = field_lay

            coord: NP_ARRAY[NP_REAL] = z_lay / 1000. # (nlay); [km]
        else: #if (field_lay is None) and (field_lev is not None):
            field: NP_ARRAY[NP_REAL] = field_lev

            coord: NP_ARRAY[NP_REAL] = z_lev / 1000. # (nlev); [km]

        field_z: NP_ARRAY[NP_REAL] = np.nanmean(field, axis = (1, 2)) # (...)
        profiles: list[NP_ARRAY[NP_REAL]] = [field_z]

    ## Plot horizontally-averaged pressure profile
    file_path: str = os.path.join(out_dir_path, kwargs["file_name"])
    xlabel: str = kwargs["xlabel"]
    ylabel: str = r"z [$km$]"
    coord_axis: str = "y"
    xscale: str = kwargs["xscale"]

    plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
        xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis, xscale = xscale)

def plot_vert_integral(xr_input: XR_DATASET, out_dir_path: str, key: str, kwargs: dict) -> None:
    # Obtain horizontal and vertical grid information
    nx: NP_INT = NP_INT(xr_input.sizes["x"])
    ny: NP_INT = NP_INT(xr_input.sizes["y"])
    nlay: NP_INT = NP_INT(xr_input.sizes["z"])
    nlev: NP_INT = NP_INT(xr_input.sizes["zh"])
    nz: NP_INT = nlay + nlev

    x: NP_ARRAY[NP_REAL] = xr_input["x"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (n_col_x)
    y: NP_ARRAY[NP_REAL] = xr_input["y"].values.astype(NP_REAL) # Column-center - y-dimension [m]; (n_col_y)

    XX_sfc: NP_ARRAY[NP_REAL] # (x, y); [m]
    YY_sfc: NP_ARRAY[NP_REAL] # (x, y); [m]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertically-averaged profiles
    lay_key: str = key + "_lay"
    lev_key: str = key + "_lev"
    field_lay: Optional[NP_ARRAY[NP_REAL]] = None
    field_lev: Optional[NP_ARRAY[NP_REAL]] = None
    assert((key in xr_input.keys()) or
        (lay_key in xr_input.keys()) or
        (lev_key in xr_input.keys()))
    if key in xr_input.keys():
        field_lay = xr_input[key].values.astype(NP_REAL) # (lay, y, x)
    else:
        if lay_key in xr_input.keys():
            field_lay = xr_input[lay_key].values.astype(NP_REAL) # (lay, y, x)
        if lev_key in xr_input.keys():
            field_lev = xr_input[lev_key].values.astype(NP_REAL) # (lev, y, x)

    assert((field_lay is not None) or (field_lev is not None))

    if (field_lay is not None) and (field_lev is not None):
        field: NP_ARRAY[NP_REAL] = np.empty([nz, ny, nx], dtype = NP_REAL) # (z, y, x)
        field[0::2,...] = field_lev
        field[1::2,...] = field_lay
    elif (field_lay is not None) and (field_lev is None):
        field: NP_ARRAY[NP_REAL] = field_lay
    else: #if (field_lay is None) and (field_lev is not None):
        field: NP_ARRAY[NP_REAL] = field_lev

    field_xy: NP_ARRAY[NP_REAL] = np.sum(field, axis = (0)) # (y, x)
    field_xy = np.transpose(field_xy, axes = (1, 0)) # (x, y)
    profile: NP_ARRAY[NP_REAL] = field_xy

    ## Plot vertically-averaged pressure profile
    meshgrid: list[NP_ARRAY[NP_REAL]] = [XX_sfc / 1000., YY_sfc / 1000.] # [km], [km]
    file_path: str = os.path.join(out_dir_path, kwargs["file_name"])
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = kwargs["cbarlabel"]
    cmin: NP_REAL = NP_REAL(0.0)
    cmax: NP_REAL = field_xy.max()
    cmap: str = kwargs["cmap"]
    cscale: str = "normal"

    if (profile.max() > profile.min()):
        plot_profile_2d(meshgrid, profile, file_path, 
            xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
            cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale)

def plot_x_integral(xr_input: XR_DATASET, out_dir_path: str, key: str, kwargs: dict) -> None:
    # Obtain horizontal and vertical grid information
    nx: NP_INT = NP_INT(xr_input.sizes["x"])
    ny: NP_INT = NP_INT(xr_input.sizes["y"])
    nlay: NP_INT = NP_INT(xr_input.sizes["z"])
    nlev: NP_INT = NP_INT(xr_input.sizes["zh"])
    nz: NP_INT = nlay + nlev

    y: NP_ARRAY[NP_REAL] = xr_input["y"].values.astype(NP_REAL) # Column-center - y-dimension [m]; (n_col_y)
    z_lay: NP_ARRAY[NP_REAL] = xr_input["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
    z_lev: NP_ARRAY[NP_REAL] = xr_input["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

    z: NP_ARRAY[NP_REAL] = np.empty(nz, dtype = NP_REAL) # (nz); [m]
    z[0::2] = z_lev
    z[1::2] = z_lay

    # Obtain vertically-averaged profiles
    lay_key: str = key + "_lay"
    lev_key: str = key + "_lev"
    field_lay: Optional[NP_ARRAY[NP_REAL]] = None
    field_lev: Optional[NP_ARRAY[NP_REAL]] = None
    assert((key in xr_input.keys()) or
        (lay_key in xr_input.keys()) or
        (lev_key in xr_input.keys()))
    if key in xr_input.keys():
        field_lay = xr_input[key].values.astype(NP_REAL) # (lay, y, x)
    else:
        if lay_key in xr_input.keys():
            field_lay = xr_input[lay_key].values.astype(NP_REAL) # (lay, y, x)
        if lev_key in xr_input.keys():
            field_lev = xr_input[lev_key].values.astype(NP_REAL) # (lev, y, x)

    assert((field_lay is not None) or (field_lev is not None))

    YY: NP_ARRAY[NP_REAL] # (x, y); [m]
    ZZ: NP_ARRAY[NP_REAL] # (x, y); [m]
    if (field_lay is not None) and (field_lev is not None):
        field: NP_ARRAY[NP_REAL] = np.empty([nz, ny, nx], dtype = NP_REAL) # (z, y, x)
        field[0::2,...] = field_lev
        field[1::2,...] = field_lay

        YY, ZZ = np.meshgrid(y, z, indexing = "ij")
    elif (field_lay is not None) and (field_lev is None):
        field: NP_ARRAY[NP_REAL] = field_lay

        YY, ZZ = np.meshgrid(y, z_lay, indexing = "ij")
    else: #if (field_lay is None) and (field_lev is not None):
        field: NP_ARRAY[NP_REAL] = field_lev

        YY, ZZ = np.meshgrid(y, z_lev, indexing = "ij")

    field_yz: NP_ARRAY[NP_REAL] = np.sum(field, axis = (2)) # (z, y)
    field_yz = np.transpose(field_yz, axes = (1, 0)) # (y, z)
    profile: NP_ARRAY[NP_REAL] = field_yz

    ## Plot vertically-averaged pressure profile
    meshgrid: list[NP_ARRAY[NP_REAL]] = [YY / 1000., ZZ / 1000.] # [km], [km]
    file_path: str = os.path.join(out_dir_path, kwargs["file_name"])
    xlabel: str = r"y [$km$]"
    ylabel: str = r"z [$km$]"
    cbarlabel: str = kwargs["cbarlabel"]
    cmin: NP_REAL = NP_REAL(0.0)
    cmax: NP_REAL = field_yz.max()
    cmap: str = kwargs["cmap"]
    cscale: str = "normal"

    if (profile.max() > profile.min()):
        plot_profile_2d(meshgrid, profile, file_path, 
            xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
            cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale)

def plot_temperature(xr_input: XR_DATASET, out_dir_path: str) -> None:
    # Obtain horizontal and vertical grid information
    nx: NP_INT = NP_INT(xr_input.sizes["x"])
    ny: NP_INT = NP_INT(xr_input.sizes["y"])
    nlay: NP_INT = NP_INT(xr_input.sizes["z"])
    nlev: NP_INT = NP_INT(xr_input.sizes["zh"])
    nz: NP_INT = nlay + nlev

    x: NP_ARRAY[NP_REAL] = xr_input["x"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (n_col_x)
    y: NP_ARRAY[NP_REAL] = xr_input["y"].values.astype(NP_REAL) # Column-center - y-dimension [m]; (n_col_y)
    z_lay: NP_ARRAY[NP_REAL] = xr_input["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
    z_lev: NP_ARRAY[NP_REAL] = xr_input["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

    z: NP_ARRAY[NP_REAL] = np.empty(nz, dtype = NP_REAL) # (nz); [m]
    z[0::2] = z_lev
    z[1::2] = z_lay

    XX_sfc: NP_ARRAY[NP_REAL] # (x, y); [m]
    YY_sfc: NP_ARRAY[NP_REAL] # (x, y); [m]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain horizontally-averaged temperature profile
    t_lay: NP_ARRAY[NP_REAL] = xr_input["t_lay"].values.astype(NP_REAL) # (lay, y, x); [K]
    t_lev: NP_ARRAY[NP_REAL] = xr_input["t_lev"].values.astype(NP_REAL) # (lev, y, x); [K]

    t: NP_ARRAY[NP_REAL] = np.empty([nz, ny, nx], dtype = NP_REAL) # (nz); [K]
    t[0::2,...] = t_lev
    t[1::2,...] = t_lay

    t_z: NP_ARRAY[NP_REAL] = np.nanmean(t, axis = (1, 2)) # (nz); [K]

    ## Plot horizontally-averaged temperature profile
    file_name: str = "temperature_z.png"

    coord: NP_ARRAY[NP_REAL] = z / 1000. # [km]
    profiles: list[NP_ARRAY[NP_REAL]] = [t_z]
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Temperature $[K]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, 
        xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis)

    # Obtain surface temperature profile
    t_sfc: NP_ARRAY[NP_REAL] = xr_input["t_sfc"].values.astype(NP_REAL) # (y, x); [K]

    t_sfc: NP_ARRAY[NP_REAL] = np.transpose(t_sfc, axes = (1, 0)) # (y, x); [K]

    ## Plot surface temperature profile
    file_name: str = "temperature_sfc.png"

    meshgrid: list[NP_ARRAY[NP_REAL]] = [XX_sfc / 1000., YY_sfc / 1000.] # [km], [km] 
    profile: NP_ARRAY[NP_REAL] = t_sfc
    file_path: str = os.path.join(out_dir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Surface Temperature $[K]$"
    cmin: NP_REAL = t_sfc.min()
    cmax: NP_REAL = t_sfc.max()

    if (cmax > cmin):
        plot_profile_2d(meshgrid, profile, file_path, xlabel = xlabel,
            ylabel = ylabel, cbarlabel = cbarlabel, cmin = cmin, cmax = cmax)

def plot_rel(xr_input: XR_DATASET, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = xr_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = xr_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = xr_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged rel profiles
    rel: np.ma.MaskedArray = xr_input.variables["rel"][:] # (lay, y, x); [μm]

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

def plot_dei(xr_input: XR_DATASET, out_dir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = xr_input.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = xr_input.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = xr_input.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged dei profiles
    dei: np.ma.MaskedArray = xr_input.variables["dei"][:] # (lay, y, x); [μm]

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