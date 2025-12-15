# Append the 'exp_hres' directory to the PYTHONPATH for future imports
import os, sys
exp_hres_dir: str = os.path.normpath( \
    os.path.join(os.path.dirname(__file__), os.pardir))
if exp_hres_dir not in sys.path:
    sys.path.append(exp_hres_dir)
    
# Standard Library Imports
import argparse
import os
import re

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_INF, NP_ARRAY, XR_DATASET
from plot_tools import plot_profiles_1d, plot_profile_2d, plot_distribution

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_output",
        description = ("Plots output of the two-stream and ray-tracer "
            + "solvers of RTE-RRTMGP-CPP.")
    )
    
    parser.add_argument("--rte_indir",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP input directory."
    )

    parser.add_argument("--rte_outdir",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP output directory."
    )

    parser.add_argument("--plot_outdir",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = ["comparison"],
        help = "Path to plot output directory."
    )
    
    args: argparse.Namespace = parser.parse_args()

    rte_indir_path: str = os.path.normpath(args.rte_indir[0])
    rte_outdir_path: str = os.path.normpath(args.rte_outdir[0])
    plot_outdir_path: str = os.path.normpath(args.plot_outdir[0])

    horz_profile_kwargs: dict = {
        "sfc_up" : {"file_name" : "sfc_up.png",
            "cbarlabel" : r"Upwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "sfc_dn" : {"file_name" : "sfc_dn.png",
            "cbarlabel" : r"Downwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "tod_up" : {"file_name" : "tod_up.png",
            "cbarlabel" : r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]"},
        "sfc_up_rd" : {"file_name" : "sfc_up_rd.png",
            "title" : r"Upwelling Shortwave Surface Flux"},
        "sfc_dn_rd" : {"file_name" : "sfc_dn_rd.png",
            "title" : r"Downwelling Shortwave Surface Flux"},
        "tod_up_rd" : {"file_name" : "tod_up_rd.png",
            "title" : r"Upwelling Shortwave Top-of-Domain Flux"}
    }
    
    #for key, kwargs in horz_profile_kwargs.items():
    #    plot_horz_profile(rte_indir_path, rte_outdir_path, plot_outdir_path, key, kwargs)

    horz_avg_kwargs: dict = {
        "flux_abs" : {"file_name" : "flux_abs.png",
            "xlabel" : r"Horizontally-Averaged Absorbed Shortwave Flux [$W m^{-3}$]",
            "xscale" : "linear"},
        "flux_abs_rd" : {"file_name" : "flux_abs_rd.png",
            "xlabel" : r"Horizontally-Averaged Relative Error of Absorbed Shortwave Flux",
            "xscale" : "linear"}
    }
    
    #for key, kwargs in horz_avg_kwargs.items():
    #    plot_horz_average(rte_indir_path, rte_outdir_path, plot_outdir_path, key, kwargs)

    distribution_profile_kwargs: dict = {
        "sfc_up" : {"file_name" : "sfc_up_dist.png",
            "xlabel" : r"Upwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "sfc_dn" : {"file_name" : "sfc_dn_dist.png",
            "xlabel" : r"Downwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "tod_up" : {"file_name" : "tod_up_dist.png",
            "xlabel" : r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]"},
        "flux_abs" : {"file_name" : "flux_abs_dist.png",
            "xlabel" : r"Absorbed Shortwave Flux [$W m^{-3}$]"}
    }
    
    for key, kwargs in distribution_profile_kwargs.items():
        plot_distribution_profile(rte_indir_path, rte_outdir_path, plot_outdir_path, key, kwargs)

def plot_horz_profile(rte_indir_path: str, rte_outdir_path: str,
    plot_outdir_path: str, key: str, kwargs: dict) -> None:

    file_ext: re.Pattern = re.compile(".in.nc")
    file_names: list[str] = sorted([file_ext.sub("", file_name) for file_name in os.listdir(rte_indir_path)])
    nfiles: int = len(file_names)

    # Set key-dependent and constant kwargs
    xlabel: str = r"x [$km$]"
    ylabel: str = r"y [$km$]"
    plot_style: str = "colormesh"

    if "rd" in key:
        cbarlabel: str = "Relative Difference"
        cmap: str = "bwr"
    else:
        cbarlabel: str = kwargs["cbarlabel"]
        cmap: str = "afmhot"

    # Get profiles from each outfile to get uniform colorbar bounds
    cmax: NP_REAL = -NP_INF
    cmin: NP_REAL = NP_INF
    plot_kwargs: dict = {}
    for ii in range(0, nfiles):
        file_name: str = file_names[ii]
        infile_name: str = file_name + ".in.nc"
        outfile_name: str = file_name + ".out.nc"

        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_path: str = os.path.join(rte_outdir_path, outfile_name)

        if (os.path.isfile(infile_path) and os.path.isfile(outfile_path)):
            # Set up plot directories
            outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_name)
            tsdir_name: str = "ts"
            tsdir_path: str = os.path.join(outfile_plot_outdir_path, tsdir_name)
            rtdir_name: str = "rt"
            rtdir_path: str = os.path.join(outfile_plot_outdir_path, rtdir_name)

            for dir_path in [outfile_plot_outdir_path, tsdir_path, rtdir_path]:
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

            plot_outdir_paths: list[str] = [tsdir_path, rtdir_path, outfile_plot_outdir_path]

            xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
                engine = "netcdf4", decode_timedelta = False)
            xr_rte_out: XR_DATASET = xr.open_dataset(outfile_path,
                engine = "netcdf4", decode_timedelta = False)

            # Obtain horizontal and vertical grid information
            nx: NP_INT = NP_INT(xr_rte_in.sizes["x"])
            ny: NP_INT = NP_INT(xr_rte_in.sizes["y"])
            nlay: NP_INT = NP_INT(xr_rte_in.sizes["z"])
            nlev: NP_INT = NP_INT(xr_rte_in.sizes["zh"])
            nz: NP_INT = nlay + nlev

            xh: NP_ARRAY[NP_REAL] = xr_rte_in["xh"].values.astype(NP_REAL) # Column-interfaces - x-dimension [m]; (nx + 1)
            yh: NP_ARRAY[NP_REAL] = xr_rte_in["yh"].values.astype(NP_REAL) # Column-interfaces - y-dimension [m]; (ny + 1)

            XX_sfc: NP_ARRAY[NP_REAL] # (nx + 1, ny + 1); [m]
            YY_sfc: NP_ARRAY[NP_REAL] # (nx + 1, ny + 1); [m]
            XX_sfc, YY_sfc = np.meshgrid(xh, yh, indexing = "ij")

            # To keep colorbars consistent, we plot ray-tracer (rt) and two-stream (ts)
            # quanities at the same time
            if "rd" in key:
                key_root: str = key[:-3]
            else:
                key_root: str = key

            if key_root == "sfc_up":
                ts_field: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].isel(lev = 0).values.astype(NP_REAL) # (ny, nx)
                rt_field: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_up"].values.astype(NP_REAL) # (ny, nx)
            elif key_root == "sfc_dn":
                ts_field: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_dn"].isel(lev = 0).values.astype(NP_REAL) # (ny, nx)
                rt_flux_sfc_dir: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_dir"].values.astype(NP_REAL) # (ny, nx)
                rt_flux_sfc_dif: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_dif"].values.astype(NP_REAL) # (ny, nx)
                rt_field: NP_ARRAY[NP_REAL] = rt_flux_sfc_dir + rt_flux_sfc_dif
            elif key_root == "tod_up":
                ts_field: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].isel(lev = -1).values.astype(NP_REAL) # (ny, nx)
                rt_field: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_tod_up"].values.astype(NP_REAL) # (ny, nx)

            ts_field = np.transpose(ts_field, axes = (1, 0)) # (nx, ny)
            rt_field = np.transpose(rt_field, axes = (1, 0)) # (nx, ny)

            meshgrid: list[NP_ARRAY[NP_REAL]] = [XX_sfc / 1000., YY_sfc / 1000.] # [km], [km]

            if "rd" in key:
                profile: NP_ARRAY[NP_REAL] = (rt_field - ts_field) / np.mean(np.abs(rt_field))
                title: str = kwargs["title"]
                file_name: str = kwargs["file_name"]
                file_path: str = os.path.join(plot_outdir_paths[2], file_name)

                cmax: NP_REAL = max(cmax, np.max(np.abs(profile)))
                cmin: NP_REAL = -cmax

                plot_kwargs[file_path] = {"meshgrid" : meshgrid,
                    "profile" : profile,
                    "title" : title,
                    "file_path" : file_path,
                }
            else:
                profiles: list[NP_ARRAY[NP_REAL]] = [ts_field, rt_field]
                titles: list[str] = ["Two-Stream", "Ray-Tracer"]
                file_prefixes: list[str] = ["ts_", "rt_"]

                cmax: NP_REAL = max(cmax, ts_field.max(), rt_field.max())
                cmin: NP_REAL = min(cmin, ts_field.min(), rt_field.min())

                for ii in range(0, 2):
                    profile: NP_ARRAY[NP_REAL] = profiles[ii]
                    title: str = titles[ii]
                    file_name: str = file_prefixes[ii] + kwargs["file_name"]
                    file_path: str = os.path.join(plot_outdir_paths[ii], file_name)

                    plot_kwargs[file_path] = {"meshgrid" : meshgrid,
                        "profile" : profile,
                        "title" : title,
                        "file_path" : file_path,
                    }

    for _, val in plot_kwargs.items():
        meshgrid: list[NP_ARRAY[NP_REAL]] = val["meshgrid"]
        profile: NP_ARRAY[NP_REAL] = val["profile"]
        file_path: str = val["file_path"]
        title: str = val["title"]

        plot_profile_2d(meshgrid, profile, file_path, title = title, xlabel = xlabel,
            ylabel = ylabel, cbarlabel = cbarlabel, cmin = cmin, cmax = cmax,
            cmap = cmap, plot_style = plot_style)

def plot_horz_average(rte_indir_path: str, rte_outdir_path: str,
    plot_outdir_path: str, key: str, kwargs: dict) -> None:

    file_ext: re.Pattern = re.compile(".in.nc")
    file_names: list[str] = sorted([file_ext.sub("", file_name) for file_name in os.listdir(rte_indir_path)])
    nfiles: int = len(file_names)

    # Set key-dependent and constant kwargs
    xlabel: str = kwargs["xlabel"]
    ylabel: str = r"z [$km$]"
    coord_axis: str = "y"
    xscale: str = kwargs["xscale"]

    if "rd" in key:
        key_root: str = key[:-3]
    else:
        key_root: str = key

    # Get profiles from each outfile to get uniform colorbar bounds
    cmax: NP_REAL = -NP_INF
    cmin: NP_REAL = NP_INF
    plot_kwargs: dict = {}
    for ii in range(0, nfiles):
        file_name: str = file_names[ii]
        infile_name: str = file_name + ".in.nc"
        outfile_name: str = file_name + ".out.nc"

        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_path: str = os.path.join(rte_outdir_path, outfile_name)

        if (os.path.isfile(infile_path) and os.path.isfile(outfile_path)):
            # Set up plot directories
            outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_name)

            for dir_path in [outfile_plot_outdir_path]:
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

            xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
                engine = "netcdf4", decode_timedelta = False)
            xr_rte_out: XR_DATASET = xr.open_dataset(outfile_path,
                engine = "netcdf4", decode_timedelta = False)

            # Obtain horizontal and vertical grid information
            nlay: NP_INT = NP_INT(xr_rte_in.sizes["z"])
            nlev: NP_INT = NP_INT(xr_rte_in.sizes["zh"])
            nz: NP_INT = nlay + nlev

            z_lay: NP_ARRAY[NP_REAL] = xr_rte_in["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
            z_lev: NP_ARRAY[NP_REAL] = xr_rte_in["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

            z: NP_ARRAY[NP_REAL] = np.empty(nz, dtype = NP_REAL) # (nz); [m]
            z[0::2] = z_lev
            z[1::2] = z_lay

            # Obtain horizontally-averaged profiles
            profiles: list[NP_ARRAY[NP_REAL]] = []
            profile_labels: Optional[list[str]] = []
            if key_root == "flux_abs":
                # Two-Stream
                ts_flux_dn: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_dn"].values.astype(NP_REAL) # (z_lev, y, x); [W m^(-2)]
                ts_flux_up: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].values.astype(NP_REAL) # (z_lev, y, x); [W m^(-2)]
                ts_field: NP_ARRAY[NP_REAL] = ((ts_flux_dn[1:] + ts_flux_up[:-1]) - (ts_flux_dn[:-1] + ts_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (z_lay, y, x); [W m^(-3)]

                # Ray-Tracer
                rt_flux_abs_dif: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_abs_dif"].values.astype(NP_REAL) # (z_lay, y, x); [W m^(-3)]
                rt_flux_abs_dir: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_abs_dir"].values.astype(NP_REAL) # (z_lay, y, x); [W m^(-3)]
                rt_field: NP_ARRAY[NP_REAL] = rt_flux_abs_dif + rt_flux_abs_dir

                coord: NP_ARRAY[NP_REAL] = z_lay / 1000. # (z_lay); [km]

            if "rd" in key:
                field: NP_ARRAY[NP_REAL] = np.sqrt(np.pow(rt_field - ts_field, 2) / np.nanmean(np.pow(rt_field, 2))) # (z_?, y, x)
                field_z: NP_ARRAY[NP_REAL] = np.nanmean(field, axis = (1, 2)) # (z_?)

                ## Plot horizontally-averaged pressure profile
                profiles: list[NP_ARRAY[NP_REAL]] = [field_z]
                file_path: str = os.path.join(outfile_plot_outdir_path, kwargs["file_name"])

                plot_kwargs[file_path] = {"coord" : coord,
                    "profiles" : profiles,
                    "profile_labels" : None,
                    "file_path" : file_path,
                }

            else:
                ts_field_z: NP_ARRAY[NP_REAL] = np.nanmean(ts_field, axis = (1, 2)) # (z_?)
                rt_field_z: NP_ARRAY[NP_REAL] = np.nanmean(rt_field, axis = (1, 2)) # (z_?)

                ## Plot horizontally-averaged pressure profile
                profiles: list[NP_ARRAY[NP_REAL]] = [ts_field_z, rt_field_z]
                profile_labels: list[str] = ["Two-Stream", "Ray-Tracer"]
                file_path: str = os.path.join(outfile_plot_outdir_path, kwargs["file_name"])

                plot_kwargs[file_path] = {"coord" : coord,
                    "profiles" : profiles,
                    "profile_labels" : profile_labels,
                    "file_path" : file_path,
                }

    for _, val in plot_kwargs.items():
        coord: NP_ARRAY[NP_REAL] = val["coord"]
        profiles: list[NP_ARRAY[NP_REAL]] = val["profiles"]
        file_path: str = val["file_path"]
        profile_labels: list[str] = val["profile_labels"]

        plot_profiles_1d(coord, profiles, file_path, profile_labels = profile_labels,
            xlabel = xlabel, ylabel = ylabel, coord_axis = coord_axis, xscale = xscale)

def plot_distribution_profile(rte_indir_path: str, rte_outdir_path: str,
    plot_outdir_path: str, key: str, kwargs: dict) -> None:

    file_ext: re.Pattern = re.compile(".in.nc")
    file_names: list[str] = sorted([file_ext.sub("", file_name) for file_name in os.listdir(rte_indir_path)])
    nfiles: int = len(file_names)

    # Set key-dependent and constant kwargs
    nbins: int = 80
    xlabel: str = kwargs["xlabel"]
    ylabel: str = "Probability Density"
    density: bool = True
    xmin: NP_REAL = NP_INF
    xmax: NP_REAL = -NP_INF
    ymax: NP_REAL = -NP_INF

    # Get profiles from each outfile to get uniform colorbar bounds
    plot_kwargs: dict = {}
    for ii in range(0, nfiles):
        file_name: str = file_names[ii]
        infile_name: str = file_name + ".in.nc"
        outfile_name: str = file_name + ".out.nc"

        infile_path: str = os.path.join(rte_indir_path, infile_name)
        outfile_path: str = os.path.join(rte_outdir_path, outfile_name)

        if (os.path.isfile(infile_path) and os.path.isfile(outfile_path)):
            # Set up plot directories
            outfile_plot_outdir_path: str = os.path.join(plot_outdir_path, file_name)
            tsdir_name: str = "ts"
            tsdir_path: str = os.path.join(outfile_plot_outdir_path, tsdir_name)
            rtdir_name: str = "rt"
            rtdir_path: str = os.path.join(outfile_plot_outdir_path, rtdir_name)

            for dir_path in [outfile_plot_outdir_path, tsdir_path, rtdir_path]:
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

            plot_outdir_paths: list[str] = [tsdir_path, rtdir_path, outfile_plot_outdir_path]

            xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
                engine = "netcdf4", decode_timedelta = False)
            xr_rte_out: XR_DATASET = xr.open_dataset(outfile_path,
                engine = "netcdf4", decode_timedelta = False)

            # To keep colorbars consistent, we plot ray-tracer (rt) and two-stream (ts)
            # quanities at the same time
            ts_field: NP_ARRAY[NP_REAL]
            rt_field: NP_ARRAY[NP_REAL]
            if key == "sfc_up":
                ts_field = xr_rte_out["sw_flux_up"].isel(lev = 0).values.astype(NP_REAL) # (ny, nx)
                rt_field = xr_rte_out["rt_flux_sfc_up"].values.astype(NP_REAL) # (ny, nx)
            elif key == "sfc_dn":
                ts_field = xr_rte_out["sw_flux_dn"].isel(lev = 0).values.astype(NP_REAL) # (ny, nx)
                rt_flux_sfc_dir: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_dir"].values.astype(NP_REAL) # (ny, nx)
                rt_flux_sfc_dif: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_dif"].values.astype(NP_REAL) # (ny, nx)
                rt_field = rt_flux_sfc_dir + rt_flux_sfc_dif
            elif key == "tod_up":
                ts_field = xr_rte_out["sw_flux_up"].isel(lev = -1).values.astype(NP_REAL) # (ny, nx)
                rt_field = xr_rte_out["rt_flux_tod_up"].values.astype(NP_REAL) # (ny, nx)
            elif key == "flux_abs":
                # Two-Stream
                z_lev: NP_ARRAY[NP_REAL] = xr_rte_in["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)
                ts_flux_dn: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_dn"].values.astype(NP_REAL) # (z_lev, y, x); [W m^(-2)]
                ts_flux_up: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].values.astype(NP_REAL) # (z_lev, y, x); [W m^(-2)]
                ts_field: NP_ARRAY[NP_REAL] = ((ts_flux_dn[1:] + ts_flux_up[:-1]) - (ts_flux_dn[:-1] + ts_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (z_lay, y, x); [W m^(-3)]

                # Ray-Tracer
                rt_flux_abs_dif: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_abs_dif"].values.astype(NP_REAL) # (z_lay, y, x); [W m^(-3)]
                rt_flux_abs_dir: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_abs_dir"].values.astype(NP_REAL) # (z_lay, y, x); [W m^(-3)]
                rt_field: NP_ARRAY[NP_REAL] = rt_flux_abs_dif + rt_flux_abs_dir

            ts_field = ts_field.flatten() # (nz_lay(?) * ny * nx)
            rt_field = rt_field.flatten() # (nz_lay(?) * ny * nx)

            xmin = min(xmin, ts_field.min(), rt_field.min())
            xmax = max(xmax, ts_field.max(), rt_field.max())

            bins: NP_ARRAY[NP_REAL] = np.linspace(xmin, xmax, nbins,
                dtype = NP_REAL)
            ts_hist: NP_ARRAY[NP_REAL]
            ts_hist, _ = np.histogram(ts_field, bins, density = "density")
            rt_hist: NP_ARRAY[NP_REAL]
            rt_hist, _ = np.histogram(rt_field, bins, density = "density")
            ymax = max(ymax, ts_hist.max(), rt_hist.max())

            datas: list[NP_ARRAY[NP_REAL]] = [ts_field, rt_field]
            titles: list[str] = ["Two-Stream", "Ray-Tracer"]
            file_prefixes: list[str] = ["ts_", "rt_"]

            for ii in range(0, 2):
                data: NP_ARRAY[NP_REAL] = datas[ii]
                title: str = titles[ii]
                file_name: str = file_prefixes[ii] + kwargs["file_name"]
                file_path: str = os.path.join(plot_outdir_paths[ii], file_name)

                plot_kwargs[file_path] = {"data" : data,
                    "title" : title,
                    "file_path" : file_path,
                }

    for _, val in plot_kwargs.items():
        data: NP_ARRAY[NP_REAL] = val["data"]
        title: str = val["title"]
        file_path: str = val["file_path"]

        plot_distribution(data, file_path, nbins = nbins, xmin = xmin,
            xmax = xmax, ymax = ymax, title = title, xlabel = xlabel, ylabel = ylabel,
            density = density)

if __name__ == "__main__":
    main()