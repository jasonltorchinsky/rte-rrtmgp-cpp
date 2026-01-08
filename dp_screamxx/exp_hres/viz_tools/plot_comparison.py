import os, sys
exp_hres_dir: str = os.path.normpath( \
    os.path.join(os.path.dirname(__file__), os.pardir))
if exp_hres_dir not in sys.path:
    sys.path.append(exp_hres_dir)
    
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
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, MPI_COMM, MPI_ROOT
from plot_tools import plot_profiles_1d

def main():
    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "plot_comparison",
        description = ("Plots comparisons of the two-stream and ray-tracer "
            + "solvers of RTE-RRTMGP-CPP.")
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

    rte_indir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_input_dir_path[0])
    rte_outdir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_output_dir_path[0])
    plot_outdir_path: str = os.path.normpath(args.rte_rrtmgp_cpp_viz_dir_path[0])

    comm: MPI_COMM = MPI.COMM_WORLD

    rmsre_convergence_kwargs: dict = {
        "sfc_up" : {"file_name" : "sfc_up_rmsre.png",
            "title" : r"Upwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "sfc_dn" : {"file_name" : "sfc_dn_rmsre.png",
            "title" : r"Downwelling Shortwave Surface Flux [$W m^{-2}$]"},
        "tod_up" : {"file_name" : "tod_up_rmsre.png",
            "title" : r"Upwelling Shortwave Top-of-Domain Flux [$W m^{-2}$]"},
        "flux_abs" : {"file_name" : "flux_abs_rmsre.png",
            "title" : r"Absorbed Shortwave Flux [$W m^{-3}$]",
            "zmax" : 16000} # [m]
    }
    
    l_keys: list[str] = get_l_keys(list(rmsre_convergence_kwargs.keys()), comm)
    for key in l_keys:
        kwargs: dict = rmsre_convergence_kwargs[key]
        plot_rmsre_convergence(rte_indir_path, rte_outdir_path, plot_outdir_path, key, kwargs)

def plot_rmsre_convergence(rte_indir_path: str, rte_outdir_path: str,
    plot_outdir_path: str, key: str, kwargs: dict):
    
    file_ext: re.Pattern = re.compile(".in.nc")
    file_names: list[str] = sorted([file_ext.sub("", file_name) for file_name in os.listdir(rte_indir_path)])

    # Group file names by resolution
    res_ext: re.Pattern = re.compile(".lr_..")
    file_groups: dict = {}
    for file_name in file_names:
        res_match: Optional[re.Match] = res_ext.search(file_name)

        res_str: str
        if res_match is None:
            res_str = "base"
        else:
            res_str = res_match.group()

        if res_str in file_groups.keys():
            file_groups[res_str] += [file_name]
        else:
            file_groups[res_str] = [file_name]

    szas_list: list[NP_ARRAY[NP_REAL]] = []
    rmsres_list: list[NP_ARRAY[NP_REAL]] = []

    profile_labels: list[str] = []

    for res_str, file_group in file_groups.items():
        group_size: int = len(file_group)
        szas: NP_ARRAY[NP_REAL] = np.zeros(group_size, dtype = NP_REAL) - 1.
        rmsres: NP_ARRAY[NP_REAL] = np.zeros(group_size, dtype = NP_REAL) - 1.

        # Get horizontal resolution for profile label
        infile_name: str = file_group[0] + ".in.nc"
        infile_path: str = os.path.join(rte_indir_path, infile_name)
        xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
                    engine = "netcdf4", decode_timedelta = False)

        xh: NP_ARRAY[NP_REAL] = xr_rte_in["xh"].values.astype(NP_REAL) # Column interfaces - x-dimension [m]; (nx + 1)
        dx: NP_REAL = xh[1] - xh[0] # Horizontal resolution [m]; ASSUME SAME IN x- AND y-
        profile_label: str
        if dx < NP_REAL(1000.0):
            profile_label = r"{:0.0f} $m$".format(dx)
        else:
            profile_label = r"{:0.2f} $km$".format(dx / 1000.)
        profile_labels += [profile_label]

        for ii in range(0, group_size):
            file_name: str = file_group[ii]
            infile_name: str = file_name + ".in.nc"
            outfile_name: str = file_name + ".out.nc"

            infile_path: str = os.path.join(rte_indir_path, infile_name)
            outfile_path: str = os.path.join(rte_outdir_path, outfile_name)

            if (os.path.isfile(infile_path) and os.path.isfile(outfile_path)):
                xr_rte_in: XR_DATASET = xr.open_dataset(infile_path,
                    engine = "netcdf4", decode_timedelta = False)
                xr_rte_out: XR_DATASET = xr.open_dataset(outfile_path,
                    engine = "netcdf4", decode_timedelta = False)

                if key == "flux_abs":
                    z_lev: NP_ARRAY[NP_REAL] = xr_rte_in["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)
                    zmax_idx: NP_INT = NP_INT(np.sum(z_lev <= kwargs["zmax"]))

                    # Two-Stream
                    ts_flux_dn: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_dn"].values.astype(NP_REAL) # (z_lev, y, x); [W m^(-2)]
                    ts_flux_up: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].values.astype(NP_REAL) # (z_lev, y, x); [W m^(-2)]
                    ts_field: NP_ARRAY[NP_REAL] = ((ts_flux_dn[1:] + ts_flux_up[:-1]) - (ts_flux_dn[:-1] + ts_flux_up[1:])) / np.expand_dims(z_lev[1:] - z_lev[:-1], [1, 2]) # (z_lay, y, x); [W m^(-3)]

                    # Ray-Tracer
                    rt_flux_abs_dif: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_abs_dif"].values.astype(NP_REAL) # (z_lay, y, x); [W m^(-3)]
                    rt_flux_abs_dir: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_abs_dir"].values.astype(NP_REAL) # (z_lay, y, x); [W m^(-3)]
                    rt_field: NP_ARRAY[NP_REAL] = rt_flux_abs_dif + rt_flux_abs_dir

                    ts_field = ts_field[:zmax_idx,...]
                    rt_field = rt_field[:zmax_idx,...]

                elif key == "sfc_up":
                    ts_field: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].isel(lev = 0).values.astype(NP_REAL) # (ny, nx)
                    rt_field: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_up"].values.astype(NP_REAL) # (ny, nx)
                elif key == "sfc_dn":
                    ts_field: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_dn"].isel(lev = 0).values.astype(NP_REAL) # (ny, nx)
                    rt_flux_sfc_dir: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_dir"].values.astype(NP_REAL) # (ny, nx)
                    rt_flux_sfc_dif: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_sfc_dif"].values.astype(NP_REAL) # (ny, nx)
                    rt_field: NP_ARRAY[NP_REAL] = rt_flux_sfc_dir + rt_flux_sfc_dif
                elif key == "tod_up":
                    ts_field: NP_ARRAY[NP_REAL] = xr_rte_out["sw_flux_up"].isel(lev = -1).values.astype(NP_REAL) # (ny, nx)
                    rt_field: NP_ARRAY[NP_REAL] = xr_rte_out["rt_flux_tod_up"].values.astype(NP_REAL) # (ny, nx)

                szas[ii] = np.rad2deg(np.nanmean(np.arccos(xr_rte_in["mu0"].values.astype(NP_REAL)))) ## ASSUME: Uniform SZA
                rmsres[ii] = np.sqrt(np.nanmean(np.pow(rt_field - ts_field, 2) / np.nanmean(np.pow(rt_field, 2))))

        szas_list += [szas[szas >= 0.]]
        rmsres_list += [rmsres[szas >= 0.]]

    coord: NP_ARRAY[NP_REAL] = szas_list[0]
    profiles: list[NP_ARRAY[NP_REAL]] = rmsres_list
    file_path: str = os.path.join(plot_outdir_path, kwargs["file_name"])
    title: str = kwargs["title"]
    xlabel: str = "Solar Zenith Angle"
    ylabel: str = "Root-Mean-Square Relative Error"
    yscale: str = "log"
    coord_axis: str = "x"

    plot_profiles_1d(coord, profiles, file_path, title = title,
        profile_labels = profile_labels, xlabel = xlabel, ylabel = ylabel,
        yscale = yscale, coord_axis = coord_axis)
    
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