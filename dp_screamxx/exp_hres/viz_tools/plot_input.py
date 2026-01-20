# Append the 'exp_hres' directory to the PYTHONPATH for future imports
import os, sys
src_dir: str = os.path.normpath( \
    os.path.join(os.path.dirname(__file__), os.pardir))
if src_dir not in sys.path:
    sys.path.append(src_dir)
    
# Standard Library Imports
import argparse
import os
import re
from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, NP_INF, MPI_COMM, MPI_ROOT
from plot_tools import plot_profiles_1d, plot_profile_2d, plot_distribution

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_input",
        description = ("Plots input to RTE-RRTMGP-CPP.")
    )
    
    parser.add_argument("--rte_rrtmgp_cpp_input_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP input directory."
    )

    parser.add_argument("--rte_rrtmgp_cpp_viz_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["comparison"],
        help = "Path to RTE-RRTMGP-CPP viz directory."
    )
    
    args: argparse.Namespace = parser.parse_args()

    rte_indir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_input_dir_path[0])
    plot_outdir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_viz_dir_path[0])

    comm: MPI_COMM = MPI.COMM_WORLD

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

    l_keys: list[str] = get_l_keys(list(horz_average_kwargs.keys()), comm)
    for key in l_keys:
        kwargs: dict = horz_average_kwargs[key]
        plot_horz_average(rte_indir_path, plot_outdir_path, key, kwargs)

    vert_integral_kwargs: dict = {
        "p" : {"file_name" : "p_vint.png",
            "cbarlabel" : r"Vertically-Integrated Pressure $[Pa]$",
            "cmap" : "Reds"},
        "t" : {"file_name" : "t_vint.png",
            "cbarlabel" : r"Vertically-Integrated Temperature $[K]$",
            "cmap" : "plasma"},
        "lwp" : {"file_name" : "lwp_vint.png",
            "cbarlabel" : r"Vertically-Integrated Liquid Water Path $[kg\,m^{-2}]$",
            "cmap" : "Blues_r"},
        "iwp" : {"file_name" : "iwp_vint.png",
            "cbarlabel" : r"Vertically-Integrated Ice Water Path $[kg\,m^{-2}]$",
            "cmap" : "Purples_r"},
        "rel" : {"file_name" : "rel_vint.png",
            "cbarlabel" : r"Vertically-Integrated Liquid Water Effective Radius [$\mu m$]",
            "cmap" : "Blues_r"},
        "dei" : {"file_name" : "dei_vint.png",
            "cbarlabel" : r"Vertically-Integrated Ice Water Effective Diameter [$\mu m$]",
            "cmap" : "Purples_r"}
    }

    l_keys: list[str] = get_l_keys(list(vert_integral_kwargs.keys()), comm)
    #for key in l_keys:
    #    kwargs: dict = vert_integral_kwargs[key]
    #    plot_vert_integral(rte_indir_path, plot_outdir_path, key, kwargs)

    x_integral_kwargs: dict = {
        "p" : {"file_name" : "p_xint.png",
            "cbarlabel" : r"$x$-Integrated Pressure $[Pa]$",
            "cmap" : "Reds"},
        "t" : {"file_name" : "t_xint.png",
            "cbarlabel" : r"$x$-Integrated Temperature $[K]$",
            "cmap" : "plasma"},
        "lwp" : {"file_name" : "lwp_xint.png",
            "cbarlabel" : r"$x$-Integrated Liquid Water Path $[kg\,m^{-2}]$",
            "cmap" : "Blues_r"},
        "iwp" : {"file_name" : "iwp_xint.png",
            "cbarlabel" : r"$x$-Integrated Ice Water Path $[kg\,m^{-2}]$",
            "cmap" : "Purples_r"},
        "rel" : {"file_name" : "rel_xint.png",
            "cbarlabel" : r"$x$-Integrated Liquid Water Effective Radius [$\mu m$]",
            "cmap" : "Blues_r"},
        "dei" : {"file_name" : "dei_xint.png",
            "cbarlabel" : r"$x$-Integrated Ice Water Effective Diameter [$\mu m$]",
            "cmap" : "Purples_r"}
    }

    l_keys: list[str] = get_l_keys(list(x_integral_kwargs.keys()), comm)
    #for key in l_keys:
    #    kwargs: dict = x_integral_kwargs[key]
    #    plot_x_integral(rte_indir_path, plot_outdir_path, key, kwargs)

    sfc_profile_kwargs: dict = {
        "t_sfc" : {"file_name" : "t_sfc.png",
            "cbarlabel" : r"Surface Temperature $[K]$",
            "cmap" : "plasma"},
        "tsi" : {"file_name" : "tsi.png",
            "cbarlabel" : r"Total Solar Irradiance $[W\,m^{-2}]$",
            "cmap" : "Reds_r"},
        "mu0" : {"file_name" : "mu_0.png",
            "cbarlabel" : r"Cosine Solar Zenith Angle",
            "cmap" : "bwr"}
    }

    l_keys: list[str] = get_l_keys(list(sfc_profile_kwargs.keys()), comm)
    for key in l_keys:
        kwargs: dict = sfc_profile_kwargs[key]
        plot_sfc_profile(rte_indir_path, plot_outdir_path, key, kwargs)

def plot_horz_average(rte_indir_path: str, plot_outdir_path: str,
    key: str, kwargs: dict) -> None:

    infile_names: list[str] = sorted(os.listdir(rte_indir_path))

    # Ignore files that only differ by SZA
    file_ext: re.Pattern = re.compile(".in.nc")
    sza_str: re.Pattern = re.compile(".sza_...")
    infile_names_trimmed: list[str] = [sza_str.sub("", file_ext.sub("", file_name)) for file_name in infile_names]
    unique_infile_name_idxs: NP_ARRAY[NP_INT] = np.unique(infile_names_trimmed, return_index = True)[1].astype(NP_INT)
    nuniqueinfiles: int = unique_infile_name_idxs.size

    # Set key-dependent and constant kwargs
    xlabel: str = kwargs["xlabel"]
    ylabel: str = r"z [$km$]"
    coord_axis: str = "y"
    xscale: str = kwargs["xscale"]

    plot_kwargs: dict = {}
    for ii in range(0, nuniqueinfiles):
        infile_idx: NP_INT = unique_infile_name_idxs[ii]
        infile_name: str = infile_names[infile_idx]
        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_ext.sub("", infile_name))
        os.makedirs(outfile_plot_outdir_path, exist_ok = True)

        xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
            engine = "netcdf4", decode_timedelta = False)

        # Obtain horizontal and vertical grid information
        nx: NP_INT = NP_INT(xr_rte_in.sizes["x"])
        ny: NP_INT = NP_INT(xr_rte_in.sizes["y"])
        nlay: NP_INT = NP_INT(xr_rte_in.sizes["z"])
        nlev: NP_INT = NP_INT(xr_rte_in.sizes["zh"])
        nz: NP_INT = nlay + nlev

        z_lay: NP_ARRAY[NP_REAL] = xr_rte_in["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
        z_lev: NP_ARRAY[NP_REAL] = xr_rte_in["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

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
                if gas_key in xr_rte_in.keys():
                    field: NP_ARRAY[NP_REAL] = xr_rte_in[gas_key].values.astype(NP_REAL) # (lay, y, x)
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

            assert((key in xr_rte_in.keys()) or
                (lay_key in xr_rte_in.keys()) or
                (lev_key in xr_rte_in.keys()))
            if key in xr_rte_in.keys():
                field_lay = xr_rte_in[key].values.astype(NP_REAL) # (lay, y, x)
            else:
                if lay_key in xr_rte_in.keys():
                    field_lay = xr_rte_in[lay_key].values.astype(NP_REAL) # (lay, y, x)
                if lev_key in xr_rte_in.keys():
                    field_lev = xr_rte_in[lev_key].values.astype(NP_REAL) # (lev, y, x)

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
        file_path: str = os.path.join(outfile_plot_outdir_path, kwargs["file_name"])

        plot_kwargs[file_ext.sub("", infile_name)] = {"coord" : coord,
            "profiles" : profiles,
            "file_path" : file_path,
            "profile_labels" : profile_labels
        }

    for _, val in plot_kwargs.items():
        coord: NP_ARRAY[NP_REAL] = val["coord"]
        profiles: list[NP_ARRAY[NP_REAL]] = val["profiles"]
        file_path: str = val["file_path"]
        profile_labels: list[str] = val["profile_labels"]

        plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
            xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis, xscale = xscale)

def plot_vert_integral(rte_indir_path: str, plot_outdir_path: str,
    key: str, kwargs: dict) -> None:

    infile_names: list[str] = sorted(os.listdir(rte_indir_path))

    # Ignore files that only differ by SZA
    file_ext: re.Pattern = re.compile(".in.nc")
    sza_str: re.Pattern = re.compile(".sza_...")
    infile_names_trimmed: list[str] = [sza_str.sub("", file_ext.sub("", file_name)) for file_name in infile_names]
    unique_infile_name_idxs: NP_ARRAY[NP_INT] = np.unique(infile_names_trimmed, return_index = True)[1].astype(NP_INT)
    nuniqueinfiles: int = unique_infile_name_idxs.size

    # Set key-dependent and constant kwargs
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = kwargs["cbarlabel"]
    cmap: str = kwargs["cmap"]
    cscale: str = "normal"
    plot_style: str = "colormesh"

    # Get profiles from each outfile to get uniform colorbar bounds
    cmax: NP_REAL = -NP_INF
    cmin: NP_REAL = NP_INF
    plot_kwargs: dict = {}
    for ii in range(0, nuniqueinfiles):
        infile_idx: NP_INT = unique_infile_name_idxs[ii]
        infile_name: str = infile_names[infile_idx]
        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_ext.sub("", infile_name))
        os.makedirs(outfile_plot_outdir_path, exist_ok = True)

        xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
            engine = "netcdf4", decode_timedelta = False)

        # Obtain horizontal and vertical grid information
        nx: NP_INT = NP_INT(xr_rte_in.sizes["x"])
        ny: NP_INT = NP_INT(xr_rte_in.sizes["y"])
        nlay: NP_INT = NP_INT(xr_rte_in.sizes["z"])
        nlev: NP_INT = NP_INT(xr_rte_in.sizes["zh"])
        nz: NP_INT = nlay + nlev

        xh: NP_ARRAY[NP_REAL] = xr_rte_in["x"].values.astype(NP_REAL) # Column-interfaces - x-dimension [m]; (nx + 1)
        yh: NP_ARRAY[NP_REAL] = xr_rte_in["y"].values.astype(NP_REAL) # Column-interfaces - y-dimension [m]; (ny + 1)

        XX_sfc: NP_ARRAY[NP_REAL] # (nx + 1, ny + 1); [m]
        YY_sfc: NP_ARRAY[NP_REAL] # (nx + 1, ny + 1); [m]
        XX_sfc, YY_sfc = np.meshgrid(xh, yh, indexing = "ij")

        # Obtain vertically-averaged profiles
        lay_key: str = key + "_lay"
        lev_key: str = key + "_lev"
        field_lay: Optional[NP_ARRAY[NP_REAL]] = None
        field_lev: Optional[NP_ARRAY[NP_REAL]] = None
        assert((key in xr_rte_in.keys()) or
            (lay_key in xr_rte_in.keys()) or
            (lev_key in xr_rte_in.keys()))
        if key in xr_rte_in.keys():
            field_lay = xr_rte_in[key].values.astype(NP_REAL) # (lay, y, x)
        else:
            if lay_key in xr_rte_in.keys():
                field_lay = xr_rte_in[lay_key].values.astype(NP_REAL) # (lay, y, x)
            if lev_key in xr_rte_in.keys():
                field_lev = xr_rte_in[lev_key].values.astype(NP_REAL) # (lev, y, x)

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
        file_path: str = os.path.join(outfile_plot_outdir_path, kwargs["file_name"])
        cmin: NP_REAL = min(cmin, field_xy.min())
        cmax: NP_REAL = max(cmax, field_xy.max())

        plot_kwargs[file_path] = {"meshgrid" : meshgrid,
            "profile" : profile,
            "file_path" : file_path,
        }

    for _, val in plot_kwargs.items():
        meshgrid: list[NP_ARRAY[NP_REAL]] = val["meshgrid"]
        profile: NP_ARRAY[NP_REAL] = val["profile"]
        file_path: str = val["file_path"]

        if (profile.max() >= profile.min()):
            plot_profile_2d(meshgrid, profile, file_path, 
                xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
                cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale,
                plot_style = plot_style)

def plot_x_integral(rte_indir_path: str, plot_outdir_path: str,
    key: str, kwargs: dict) -> None:

    infile_names: list[str] = sorted(os.listdir(rte_indir_path))

    # Ignore files that only differ by SZA
    file_ext: re.Pattern = re.compile(".in.nc")
    sza_str: re.Pattern = re.compile(".sza_...")
    infile_names_trimmed: list[str] = [sza_str.sub("", file_ext.sub("", file_name)) for file_name in infile_names]
    unique_infile_name_idxs: NP_ARRAY[NP_INT] = np.unique(infile_names_trimmed, return_index = True)[1].astype(NP_INT)
    nuniqueinfiles: int = unique_infile_name_idxs.size

    # Set key-dependent and constant kwargs
    xlabel: str = r"y [$km$]"
    ylabel: str = r"z [$km$]"
    cbarlabel: str = kwargs["cbarlabel"]
    cmap: str = kwargs["cmap"]
    cscale: str = "normal"
    plot_style: str = "colormesh"

    # Get profiles from each outfile to get uniform colorbar bounds
    cmax: NP_REAL = -NP_INF
    cmin: NP_REAL = NP_INF
    plot_kwargs: dict = {}
    for ii in range(0, nuniqueinfiles):
        infile_idx: NP_INT = unique_infile_name_idxs[ii]
        infile_name: str = infile_names[infile_idx]
        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_ext.sub("", infile_name))
        os.makedirs(outfile_plot_outdir_path, exist_ok = True)

        xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
            engine = "netcdf4", decode_timedelta = False)

        # Obtain horizontal and vertical grid information
        nx: NP_INT = NP_INT(xr_rte_in.sizes["x"])
        ny: NP_INT = NP_INT(xr_rte_in.sizes["y"])
        nlay: NP_INT = NP_INT(xr_rte_in.sizes["z"])
        nlev: NP_INT = NP_INT(xr_rte_in.sizes["zh"])
        nz: NP_INT = nlay + nlev

        y: NP_ARRAY[NP_REAL] = xr_rte_in["y"].values.astype(NP_REAL) # Column-center - y-dimension [m]; (n_col_y)
        yh: NP_ARRAY[NP_REAL] = xr_rte_in["yh"].values.astype(NP_REAL) # Column-interface - y-dimension [m]; (n_col_y)
        z_lay: NP_ARRAY[NP_REAL] = xr_rte_in["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
        z_lev: NP_ARRAY[NP_REAL] = xr_rte_in["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

        z: NP_ARRAY[NP_REAL] = np.empty(nz, dtype = NP_REAL) # (nz); [m]
        z[0::2] = z_lev
        z[1::2] = z_lay

        # Obtain vertically-averaged profiles
        lay_key: str = key + "_lay"
        lev_key: str = key + "_lev"
        field_lay: Optional[NP_ARRAY[NP_REAL]] = None
        field_lev: Optional[NP_ARRAY[NP_REAL]] = None
        assert((key in xr_rte_in.keys()) or
            (lay_key in xr_rte_in.keys()) or
            (lev_key in xr_rte_in.keys()))
        if key in xr_rte_in.keys():
            field_lay = xr_rte_in[key].values.astype(NP_REAL) # (lay, y, x)
        else:
            if lay_key in xr_rte_in.keys():
                field_lay = xr_rte_in[lay_key].values.astype(NP_REAL) # (lay, y, x)
            if lev_key in xr_rte_in.keys():
                field_lev = xr_rte_in[lev_key].values.astype(NP_REAL) # (lev, y, x)

        assert((field_lay is not None) or (field_lev is not None))

        YY: NP_ARRAY[NP_REAL] # (x, y); [m]
        ZZ: NP_ARRAY[NP_REAL] # (x, y); [m]
        if plot_style == "colormesh": # Ue pcolor mesh, only visualize layer values
            assert(field_lay is not None)
            field: NP_ARRAY[NP_REAL] = field_lay
            YY, ZZ = np.meshgrid(yh, z_lev, indexing = "ij")
        else: # ASSUME kwargs["plot_style"] = "contour"
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
        file_path: str = os.path.join(outfile_plot_outdir_path, kwargs["file_name"])
        cmin: NP_REAL = min(cmin, field_yz.min())
        cmax: NP_REAL = max(cmax, field_yz.max())

        plot_kwargs[file_path] = {"meshgrid" : meshgrid,
            "profile" : profile,
            "file_path" : file_path,
        }

    for _, val in plot_kwargs.items():
        meshgrid: list[NP_ARRAY[NP_REAL]] = val["meshgrid"]
        profile: NP_ARRAY[NP_REAL] = val["profile"]
        file_path: str = val["file_path"]

        if (profile.max() >= profile.min()):
            plot_profile_2d(meshgrid, profile, file_path, 
                xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
                cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale,
                plot_style = plot_style)

def plot_sfc_profile(rte_indir_path: str, plot_outdir_path: str,
    key: str, kwargs: dict) -> None:

    infile_names: list[str] = sorted(os.listdir(rte_indir_path))

    # Ignore files that only differ by SZA
    file_ext: re.Pattern = re.compile(".in.nc")
    sza_str: re.Pattern = re.compile(".sza_...")
    infile_names_trimmed: list[str] = [sza_str.sub("", file_ext.sub("", file_name)) for file_name in infile_names]
    unique_infile_name_idxs: NP_ARRAY[NP_INT] = np.unique(infile_names_trimmed, return_index = True)[1].astype(NP_INT)
    nuniqueinfiles: int = unique_infile_name_idxs.size

    # Set key-dependent and constant kwargs
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = kwargs["cbarlabel"]
    cmap: str = kwargs["cmap"]
    cscale: str = "normal"
    plot_style: str = "colormesh"

    # Get profiles from each outfile to get uniform colorbar bounds
    cmax: NP_REAL = -NP_INF
    cmin: NP_REAL = NP_INF
    plot_kwargs: dict = {}
    for ii in range(0, nuniqueinfiles):
        infile_idx: NP_INT = unique_infile_name_idxs[ii]
        infile_name: str = infile_names[infile_idx]
        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_ext.sub("", infile_name))
        os.makedirs(outfile_plot_outdir_path, exist_ok = True)

        xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
            engine = "netcdf4", decode_timedelta = False)

        # Obtain horizontal and vertical grid information
        xh: NP_ARRAY[NP_REAL] = xr_rte_in["x"].values.astype(NP_REAL) # Column-interfaces - x-dimension [m]; (nx + 1)
        yh: NP_ARRAY[NP_REAL] = xr_rte_in["y"].values.astype(NP_REAL) # Column-interfaces - y-dimension [m]; (ny + 1)

        XX_sfc: NP_ARRAY[NP_REAL] # (nx + 1, ny + 1); [m]
        YY_sfc: NP_ARRAY[NP_REAL] # (nx + 1, ny + 1); [m]
        XX_sfc, YY_sfc = np.meshgrid(xh, yh, indexing = "ij")

        field_xy: NP_ARRAY[NP_REAL] = xr_rte_in[key].values.astype(NP_REAL) # (y, x)
        field_xy = np.transpose(field_xy, axes = (1, 0)) # (x, y)
        profile: NP_ARRAY[NP_REAL] = field_xy

        ## Plot vertically-averaged pressure profile
        meshgrid: list[NP_ARRAY[NP_REAL]] = [XX_sfc / 1000., YY_sfc / 1000.] # [km], [km]
        file_path: str = os.path.join(outfile_plot_outdir_path, kwargs["file_name"])
        cmin: NP_REAL = min(cmin, field_xy.min())
        cmax: NP_REAL = max(cmax, field_xy.max())

        plot_kwargs[file_path] = {"meshgrid" : meshgrid,
            "profile" : profile,
            "file_path" : file_path,
        }

    for _, val in plot_kwargs.items():
        meshgrid: list[NP_ARRAY[NP_REAL]] = val["meshgrid"]
        profile: NP_ARRAY[NP_REAL] = val["profile"]
        file_path: str = val["file_path"]

        if (profile.max() >= profile.min()):
            plot_profile_2d(meshgrid, profile, file_path, 
                xlabel = xlabel, ylabel = ylabel, cbarlabel = cbarlabel,
                cmin = cmin, cmax = cmax, cmap = cmap, cscale = cscale,
                plot_style = plot_style)

def plot_temperature(xr_rte_in: XR_DATASET, outdir_path: str) -> None:
    # Obtain horizontal and vertical grid information
    nx: NP_INT = NP_INT(xr_rte_in.sizes["x"])
    ny: NP_INT = NP_INT(xr_rte_in.sizes["y"])
    nlay: NP_INT = NP_INT(xr_rte_in.sizes["z"])
    nlev: NP_INT = NP_INT(xr_rte_in.sizes["zh"])
    nz: NP_INT = nlay + nlev

    x: NP_ARRAY[NP_REAL] = xr_rte_in["x"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (n_col_x)
    y: NP_ARRAY[NP_REAL] = xr_rte_in["y"].values.astype(NP_REAL) # Column-center - y-dimension [m]; (n_col_y)
    z_lay: NP_ARRAY[NP_REAL] = xr_rte_in["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
    z_lev: NP_ARRAY[NP_REAL] = xr_rte_in["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

    z: NP_ARRAY[NP_REAL] = np.empty(nz, dtype = NP_REAL) # (nz); [m]
    z[0::2] = z_lev
    z[1::2] = z_lay

    XX_sfc: NP_ARRAY[NP_REAL] # (x, y); [m]
    YY_sfc: NP_ARRAY[NP_REAL] # (x, y); [m]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain horizontally-averaged temperature profile
    t_lay: NP_ARRAY[NP_REAL] = xr_rte_in["t_lay"].values.astype(NP_REAL) # (lay, y, x); [K]
    t_lev: NP_ARRAY[NP_REAL] = xr_rte_in["t_lev"].values.astype(NP_REAL) # (lev, y, x); [K]

    t: NP_ARRAY[NP_REAL] = np.empty([nz, ny, nx], dtype = NP_REAL) # (nz); [K]
    t[0::2,...] = t_lev
    t[1::2,...] = t_lay

    t_z: NP_ARRAY[NP_REAL] = np.nanmean(t, axis = (1, 2)) # (nz); [K]

    ## Plot horizontally-averaged temperature profile
    file_name: str = "temperature_z.png"

    coord: NP_ARRAY[NP_REAL] = z / 1000. # [km]
    profiles: list[NP_ARRAY[NP_REAL]] = [t_z]
    file_path: str = os.path.join(outdir_path, file_name)
    xlabel: str = r"Horizontally-Averaged Temperature $[K]$"
    ylabel: str = r"z $[km]$"
    coord_axis: str = "y"

    plot_profiles_1d(coord, profiles, file_path, 
        xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis)

    # Obtain surface temperature profile
    t_sfc: NP_ARRAY[NP_REAL] = xr_rte_in["t_sfc"].values.astype(NP_REAL) # (y, x); [K]

    t_sfc: NP_ARRAY[NP_REAL] = np.transpose(t_sfc, axes = (1, 0)) # (y, x); [K]

    ## Plot surface temperature profile
    file_name: str = "temperature_sfc.png"

    meshgrid: list[NP_ARRAY[NP_REAL]] = [XX_sfc / 1000., YY_sfc / 1000.] # [km], [km] 
    profile: NP_ARRAY[NP_REAL] = t_sfc
    file_path: str = os.path.join(outdir_path, file_name)
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    cbarlabel: str = r"Surface Temperature $[K]$"
    cmin: NP_REAL = t_sfc.min()
    cmax: NP_REAL = t_sfc.max()

    if (cmax > cmin):
        plot_profile_2d(meshgrid, profile, file_path, xlabel = xlabel,
            ylabel = ylabel, cbarlabel = cbarlabel, cmin = cmin, cmax = cmax)

def plot_rel(xr_rte_in: XR_DATASET, outdir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = xr_rte_in.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = xr_rte_in.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = xr_rte_in.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged rel profiles
    rel: np.ma.MaskedArray = xr_rte_in.variables["rel"][:] # (lay, y, x); [μm]

    rel_xy: np.ma.MaskedArray = np.sum(rel, axis = (0)) # (y, x); [μm]
    rel_xy: np.ma.MaskedArray = np.transpose(rel_xy, axes = (1, 0)) # (x, y); [μm]

    rel_lay: np.ma.MaskedArray = np.nanmean(rel, axis = (1, 2)) # (lay); [μm]

    ## Plot vertically-integrated rel profile
    file_name: str = "rel_xy.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = rel_xy
    file_path: str = os.path.join(outdir_path, file_name)
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
    file_path: str = os.path.join(outdir_path, file_name)
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
    file_path: str = os.path.join(outdir_path, file_name)
    nbins: int = 256
    xlabel: str = r"Liquid Water Effective Radius [$\mu m$]"
    ylabel: str = "Counts"
    title: str = r"Liquid Water Effective Radius Distribution"
    xscale: str = "linear"
    yscale: str = "linear"

    if (rel.max() > rel.min()):
        plot_distribution(profile, file_path, nbins = nbins, title = title,
            xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale)

def plot_dei(xr_rte_in: XR_DATASET, outdir_path: str) -> None:
    # Obtain the horizontal coordinates
    x: np.ma.MaskedArray = xr_rte_in.variables["x"][:] # (x); [m]
    y: np.ma.MaskedArray = xr_rte_in.variables["y"][:] # (y); [m]

    x: np.ndma.MaskedArray = x / 1000. # (x); [km]
    y: np.ndma.MaskedArray = y / 1000. # (y); [km]

    nx: int = np.size(x)
    ny: int = np.size(y)

    XX_sfc: np.ma.MaskedArray # (x, y); [km]
    YY_sfc: np.ma.MaskedArray # (x, y); [km]
    XX_sfc, YY_sfc = np.meshgrid(x, y, indexing = "ij")

    # Obtain vertical coordinate
    z_lay: np.ma.MaskedArray = xr_rte_in.variables["z_lay"][:] # (lay); [m]

    z_lay: np.ma.MaskedArray = z_lay / 1000. # (lay); [km]

    nlay: int = np.size(z_lay)

    # Obtain the vertically-integrated and horizontally-averaged dei profiles
    dei: np.ma.MaskedArray = xr_rte_in.variables["dei"][:] # (lay, y, x); [μm]

    dei_xy: np.ma.MaskedArray = np.sum(dei, axis = (0)) # (y, x); [μm]
    dei_xy: np.ma.MaskedArray = np.transpose(dei_xy, axes = (1, 0)) # (x, y); [μm]

    dei_lay: np.ma.MaskedArray = np.nanmean(dei, axis = (1, 2)) # (lay); [μm]

    ## Plot vertically-integrated dei profile
    file_name: str = "dei_xy.png"

    meshgrid: tuple = [XX_sfc, YY_sfc] 
    profile: np.ndarray = dei_xy
    file_path: str = os.path.join(outdir_path, file_name)
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
    file_path: str = os.path.join(outdir_path, file_name)
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
    file_path: str = os.path.join(outdir_path, file_name)
    nbins: int = 256
    xlabel: str = r"Ice Water Effective Diameter [$\mu m$]"
    ylabel: str = "Counts"
    title: str = r"Ice Water Effective Diameter Distribution"
    xscale: str = "linear"
    yscale: str = "linear"

    if (dei.max() > dei.min()):
        plot_distribution(profile, file_path, nbins = nbins, title = title,
            xlabel = xlabel, ylabel = ylabel, xscale = xscale, yscale = yscale)

def get_l_keys(g_keys: list[str], comm: MPI_COMM) -> list[str]:
    comm_size: NP_INT = NP_INT(comm.Get_size())
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    g_count: NP_INT = len(g_keys)
    l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
    l_counts[0] = (g_count // comm_size + int(0 < (g_count % comm_size)))

    l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
    ii: int
    for ii in range(1, comm_size):
        l_counts[ii] = g_count // comm_size + int(ii < (g_count % comm_size))
        l_displs[ii] = l_counts[ii - 1] + l_displs[ii - 1]

    l_keys: list[str] = g_keys[l_displs[l_rank]:l_displs[l_rank] + l_counts[l_rank]]

    return l_keys

if __name__ == "__main__":
    main()