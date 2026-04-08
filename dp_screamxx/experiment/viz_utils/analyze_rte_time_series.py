import os, sys
exp_hres_dir: str = os.path.normpath( \
    os.path.join(os.path.dirname(__file__), os.pardir))
if exp_hres_dir not in sys.path:
    sys.path.append(exp_hres_dir)

# Standard Library Imports
import argparse
import ast
import os
import re
import sys
from contextlib import contextmanager
import time

from typing import Optional

# Third-Party Library Imports
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, XR_DATAARRAY, NP_EPS
from plot_utils import plot_profiles_1d


# Script variables
prog_name: str = "analyze_rte_timeseries"
prog_desc: str = "Analyze a time series of RTE-RRTMGP-CPP+RT output."

def main(argv):
    # MPI Communicator info
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = prog_name,
        description = prog_desc
    )
    
    parser.add_argument("--rte_rrtmgp_cpp_input_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP input directory."
    )

    parser.add_argument("--rte_rrtmgp_cpp_output_dir_path",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP output directory."
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

    rte_rrtmgp_cpp_input_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_input_dir_path[0])
    rte_rrtmgp_cpp_output_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_output_dir_path[0])
    rte_rrtmgp_cpp_viz_dir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_viz_dir_path[0])

    comm: MPI_COMM = MPI.COMM_WORLD

    rmsre_timeseries_kwargs: dict = {
        "sfc_up" : {"file_name" : "sfc_up_rmsre.png",
            "title" : r"Upwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "sfc_dn" : {"file_name" : "sfc_dn_rmsre.png",
            "title" : r"Downwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "tod_up" : {"file_name" : "tod_up_rmsre.png",
            "title" : r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]"},
        "flux_abs" : {"file_name" : "flux_abs_rmsre.png",
            "title" : r"Absorbed Shortwave Flux [$W m^{-3}$]",
            "z_max" : 8000} # [m]
    }

    l_keys: list[str] = get_l_keys(list(rmsre_timeseries_kwargs.keys()), comm)
    for key in l_keys:
        print("key: {}".format(key), flush = True)
        kwargs: dict = rmsre_timeseries_kwargs[key]
        plot_rmsre_timeseries(rte_rrtmgp_cpp_input_dir_path, rte_rrtmgp_cpp_output_dir_path, 
            rte_rrtmgp_cpp_viz_dir_path, key, **kwargs)

def plot_rmsre_timeseries(rte_rrtmgp_cpp_input_dir_path: str, rte_rrtmgp_cpp_output_dir_path: str,
    rte_rrtmgp_cpp_viz_dir_path: str, key: str, **kwargs):

    # Set up for plot_profiles_1d
    times_list: list[NP_ARRAY[NP_REAL]] = []
    szas_list: list[NP_ARRAY[NP_REAL]] = []
    rmsres_list: list[NP_ARRAY[NP_REAL]] = []
    profile_labels: list[str] = []

    # Group RTE-RRTMGP-CPP+RT file roots by resolution
    input_extension: re.Pattern = re.compile(".in.nc")
    output_extension: re.Pattern = re.compile(".out.nc")
    rte_rrtmgp_cpp_file_roots: list[str] = \
        sorted([input_extension.sub("", file_name) 
            for file_name in os.listdir(rte_rrtmgp_cpp_input_dir_path)],
            reverse = True) # Coarse resolutions first

    # Set up lists for plotting
    time_list: list[NP_ARRAY[NP_REAL]] = []
    sza_list: list[NP_ARRAY[NP_REAL]] = []
    rmsre_list: list[NP_ARRAY[NP_REAL]] = []

    plot_labels: list[str] = []

    # Loop resolutions
    for rte_rrtmgp_cpp_file_root in rte_rrtmgp_cpp_file_roots:
        print("Starting resolution file {}...".format(rte_rrtmgp_cpp_file_root), flush = True)

        rte_rrtmgp_cpp_input_file_path: str = os.path.join(
            rte_rrtmgp_cpp_input_dir_path, rte_rrtmgp_cpp_file_root + input_extension.pattern)
        rte_rrtmgp_cpp_output_file_path: str = os.path.join(
            rte_rrtmgp_cpp_output_dir_path, rte_rrtmgp_cpp_file_root + output_extension.pattern)

        if (os.path.isfile(rte_rrtmgp_cpp_input_file_path)
                and os.path.isfile(rte_rrtmgp_cpp_output_file_path)):

                with timer("{}: {}".format(key, rte_rrtmgp_cpp_file_root)):

                    xr_rte_rrtmgp_cpp_input: XR_DATASET = xr.open_dataset(rte_rrtmgp_cpp_input_file_path,
                        engine = "netcdf4", decode_timedelta = False)
                    xr_rte_rrtmgp_cpp_output: XR_DATASET = xr.open_dataset(rte_rrtmgp_cpp_output_file_path,
                        engine = "netcdf4", decode_timedelta = False)

                    # Assume spatially constant SZA
                    sza: XR_DATAARRAY = np.rad2deg(np.arccos(xr_rte_rrtmgp_cpp_input["mu0"].mean(dim = ["x", "y"])))
                    rmsre: XR_DATAARRAY = calculate_rte_rrtmgp_cpp_rmsre(xr_rte_rrtmgp_cpp_input, xr_rte_rrtmgp_cpp_output, key)

                    if rmsre.max() > NP_EPS:
                        mask: XR_DATAARRAY = abs(rmsre) > NP_EPS
                        sza = sza.where(mask)
                        rmsre = rmsre.where(mask)

                        time_list += [rmsre["time"].values.astype(NP_REAL)]
                        sza_list += [sza.values.astype(NP_REAL)]
                        rmsre_list += [rmsre.values.astype(NP_REAL)]

                        # Get profile label
                        dx: NP_REAL = np.diff(xr_rte_rrtmgp_cpp_input["xh"].isel(xh = [1,2]).values.astype(NP_REAL))[0] # Horizontal Resolution [m]

                        profile_label: str
                        if dx < NP_REAL(1000.0):
                            profile_label = r"{:0.1f} $m$".format(dx)
                        else:
                            profile_label = r"{:0.3f} $km$".format(dx / 1000.)
                        profile_labels += [profile_label]

    coord: NP_ARRAY[NP_REAL] = time_list[0]
    profiles: list[NP_ARRAY[NP_REAL]] = rmsre_list
    file_path: str = os.path.join(rte_rrtmgp_cpp_viz_dir_path, kwargs["file_name"])
    title: str = kwargs["title"]
    xlabel: str = "Time Since Simulation Start [h]"
    ylabel: str = "Root-Mean-Square Relative Error"
    yscale: str = "log"
    coord_axis: str = "x"

    plot_profiles_1d(coord, profiles, file_path, title = title,
        profile_labels = profile_labels, xlabel = xlabel, ylabel = ylabel,
        yscale = yscale, coord_axis = coord_axis)

def group_rte_rrtmgp_cpp_file_roots_by_resolution(rte_rrtmgp_cpp_dir_path: str) -> dict:
    io_extension: re.Pattern = re.compile("....nc")

    rte_rrtmgp_cpp_file_paths: list[str] = \
        sorted([io_extension.sub("", file_name) 
            for file_name in os.listdir(rte_rrtmgp_cpp_dir_path)])

    resolution_extension: re.Pattern = re.compile(".lr_..")
    rte_rrtmgp_cpp_file_path_resolution_groups: dict = {}
    for rte_rrtmgp_cpp_file_path in rte_rrtmgp_cpp_file_paths:
        resolution_match: Optional[re.Match] = resolution_extension.search(rte_rrtmgp_cpp_file_path)

        resolution_str: str
        if resolution_match is None:
            resolution_str = "base"
        else:
            resolution_str = resolution_match.group()

        if resolution_str in rte_rrtmgp_cpp_file_path_resolution_groups.keys():
            rte_rrtmgp_cpp_file_path_resolution_groups[resolution_str] += [rte_rrtmgp_cpp_file_path]
        else:
            rte_rrtmgp_cpp_file_path_resolution_groups[resolution_str] = [rte_rrtmgp_cpp_file_path]

    return rte_rrtmgp_cpp_file_path_resolution_groups

def calculate_rte_rrtmgp_cpp_rmsre(xr_rte_rrtmgp_cpp_input: XR_DATASET, xr_rte_rrtmgp_cpp_output: XR_DATASET,
    key: str, **kwargs) -> NP_REAL:

    default_kwargs: dict = {
        "z_max" : 100000.0,  # Maximum altitude
    }

    l_kwargs: dict = {**default_kwargs, **kwargs}

    assert(key in ["flux_abs", "sfc_up", "sfc_dn", "tod_up"])

    ts_field: XR_DATAARRAY # Two-Stream solver quantity
    rt_field: XR_DATAARRAY # Ray-Tracer solver quantity
    if key == "flux_abs": # Absorbed shortwave flux [W m^(-3)]
        z_lev: XR_DATAARRAY = xr_rte_rrtmgp_cpp_input["z_lev"].sel(z_lev = slice(None, l_kwargs["z_max"])) # Level altitude - z-dimension [m]; (n_lev_z)
        z_lay: XR_DATAARRAY = xr_rte_rrtmgp_cpp_input["z_lay"].sel(z_lay = slice(None, z_lev.max())) # Layer altitude - z-dimension [m]; (n_lay_z)

        # Two-Stream
        dz = z_lev.diff("z_lev").rename({"z_lev": "lay"}).rename("dz").assign_coords(lay = z_lay.values) # Layer thickness lives on layer midpoints
        ts_flux_dn: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_dn"].sel(lev = slice(None, l_kwargs["z_max"])).rename("ts_flux_dn") # (z_lev, y, x); [W m^(-2)]
        ts_flux_up: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_up"].sel(lev = slice(None, l_kwargs["z_max"])).rename("ts_flux_up") # (z_lev, y, x); [W m^(-2)]
        ts_flux_diff: XR_DATAARRAY = \
            ( ts_flux_up.isel(lev = slice(None, -1)) - ts_flux_dn.isel(lev = slice(None, -1)) ).rename({"lev": "lay"}).assign_coords(lay = z_lay.values) \
            + ( ts_flux_dn.isel(lev = slice(1, None)) - ts_flux_up.isel(lev = slice(1, None))).rename({"lev": "lay"}).assign_coords(lay = z_lay.values)
        ts_flux_diff.attrs.update({"long_name" : "Difference in incoming and outgoing shortwave flux at each layer (TwoStrem solver)"})
        ts_field: XR_DATAARRAY = ts_flux_diff / dz # (z_lay, y, x); [W m^(-3)]
        ts_field.attrs.pop("description")
        ts_field.attrs.update({"long_name" : "Absorbed shortwave fluxes (TwoStream solver)", "units" : "W m-3"})

        # Ray-Tracer
        rt_flux_abs_dif: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_abs_dif"].sel(z = slice(None, l_kwargs["z_max"])).rename({"z" : "lay"}) # (z_lay, y, x); [W m^(-3)]
        rt_flux_abs_dir: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_abs_dir"].sel(z = slice(None, l_kwargs["z_max"])).rename({"z" : "lay"}) # (z_lay, y, x); [W m^(-3)]
        rt_field: XR_DATAARRAY = rt_flux_abs_dif + rt_flux_abs_dir
        rt_field.attrs.update({"long_name" : "Absorbed shortwave fluxes (Monte Carlo ray tracer)", "units" : "W m-3"})

        rmsre: XR_DATARRAY = np.sqrt((((rt_field - ts_field)**2) / (rt_field**2).mean(dim = ["x", "y", "lay"])).fillna(0).mean(dim = ["x", "y", "lay"]))
        rmsre.attrs.pop("units")
        rmsre.attrs.update({"long_name" : "Root-Mean-Square Relative Error Absorbed Shortwave Fluxes"})
    elif key == "sfc_up":
        ts_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_up"].isel(lev = 0) # (ny, nx)
        ts_field.attrs.update({"long_name" : "Upwelling shortwave surface fluxes (TwoStream solver)"})

        rt_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_sfc_up"] # (ny, nx)

        rmsre: XR_DATARRAY = np.sqrt((((rt_field - ts_field)**2) / (rt_field**2).mean(dim = ["x", "y"])).fillna(0).mean(dim = ["x", "y"]))
        rmsre.attrs.pop("units")
        rmsre.attrs.update({"long_name" : "Root-Mean-Square Relative Error Absorbed Upwelling Shortwave Surface Fluxes"})
    elif key == "sfc_dn":
        ts_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_dn"].isel(lev = 0) # (ny, nx)
        ts_field.attrs.update({"long_name" : "Downwelling shortwave surface fluxes (TwoStream solver)"})

        rt_flux_sfc_dir: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_sfc_dir"] # (ny, nx)
        rt_flux_sfc_dif: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_sfc_dif"] # (ny, nx)
        rt_field: XR_DATAARRAY = rt_flux_sfc_dir + rt_flux_sfc_dif
        rt_field.attrs.update({"long_name" : "Downwelling shortwave surface fluxes (Monte Carlo ray tracer)"})

        rmsre: XR_DATARRAY = np.sqrt((((rt_field - ts_field)**2) / (rt_field**2).mean(dim = ["x", "y"])).fillna(0).mean(dim = ["x", "y"]))
        rmsre.attrs.pop("units")
        rmsre.attrs.update({"long_name" : "Root-Mean-Square Relative Error Absorbed Downwelling Shortwave Surface Fluxes"})
    elif key == "tod_up":
        ts_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["sw_flux_up"].isel(lev = -1) # (ny, nx)
        ts_field.attrs.update({"long_name" : "Upwelling shortwave top-of-domain fluxes (TwoStream solver)"})

        rt_field: XR_DATAARRAY = xr_rte_rrtmgp_cpp_output["rt_flux_tod_up"] # (ny, nx)

        rmsre: XR_DATARRAY = np.sqrt((((rt_field - ts_field)**2) / (rt_field**2).mean(dim = ["x", "y"])).fillna(0).mean(dim = ["x", "y"]))
        rmsre.attrs.pop("units")
        rmsre.attrs.update({"long_name" : "Root-Mean-Square Relative Error Absorbed Upwelling Shortwave Top-of-Domain Fluxes"})

    return rmsre.assign_coords(time = xr_rte_rrtmgp_cpp_input["time"].values)

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


@contextmanager
def timer(label = "Elapsed"):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"{label}: {dt:.6f} s")

if __name__ == "__main__":
    main(sys.argv)