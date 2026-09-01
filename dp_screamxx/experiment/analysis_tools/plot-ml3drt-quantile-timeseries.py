#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
import re
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, \
    MPL_FIGURE, MPL_AXES
from consts.numeric import NP_INF
from consts.visual import plot_colors
from ml3drt import calc_sw_reflectance as ml3drt_calc_sw_reflectance, \
    calc_sw_heating as ml3drt_calc_sw_heating, \
    calc_sw_flux_sfc_dn as ml3drt_calc_sw_flux_sfc_dn
from rte_rrtmgp_cpp import find_inout_pairs as rte_rrtmgp_cpp_find_inout_pairs, \
    find_daytime_indices as rte_rrtmgp_cpp_find_daytime_indices, \
    find_szas as rte_rrtmgp_cpp_find_szas, \
    find_times as rte_rrtmgp_cpp_find_times, \
    calc_sw_reflectance as rte_rrtmgp_cpp_calc_sw_reflectance, \
    calc_sw_heating as rte_rrtmgp_cpp_calc_sw_heating, \
    calc_sw_flux_sfc_dn as rte_rrtmgp_cpp_calc_sw_flux_sfc_dn, \
    find_grid as rte_rrtmgp_cpp_find_grid, \
    calc_z_max_info as rte_rrtmgp_cpp_calc_z_max_info, \
    print_msg as rte_rrtmgp_cpp_print_msg

# Script variables
prog_name: str = "plot-ml3drt-quantile-timeseries"
prog_desc: str = "Visualize distributions of ML3DRT emulator and two-stream solver differences for RTE-RRTMGP-CPP."

dist_stat_names: list[str] = [
    "min",
    "p10",
    "p20",
    "p40",
    "median",
    "p60",
    "p80",
    "p90",
    "max"
]

solver_names: list[str] = [
    "emulator",
    "two_stream"
]

solver_label_names: list[str] = [
    "Emulator",
    "Two-Stream"
]

def calc_distribution_info(diff: XR_DATAARRAY) -> dict:
    values: NP_ARRAY[NP_REAL] = np.asarray(diff, dtype = NP_REAL)

    percentiles: NP_ARRAY[NP_REAL] = NP_REAL(
        np.nanpercentile(
            values,
            [10., 20., 40., 50., 60., 80., 90.],
            axis = 1
        )
    )

    distribution_info: dict = {
        "min": NP_REAL(np.nanmin(values, axis = 1)),
        "p10": percentiles[0,...],
        "p20": percentiles[1,...],
        "p40": percentiles[2,...],
        "median": percentiles[3,...],
        "p60": percentiles[4,...],
        "p80": percentiles[5,...],
        "p90": percentiles[6,...],
        "max": NP_REAL(np.nanmax(values, axis = 1)),
    }

    return distribution_info

def add_distribution_info_to_array(
    distribution_array: NP_ARRAY[NP_REAL],
    solver_index: NP_INT,
    day_index: NP_INT,
    distribution_info: dict
):
    ss: int
    for ss in range(0, len(dist_stat_names)):
        stat_name: str = dist_stat_names[ss]
        distribution_array[solver_index,day_index,:,ss] = distribution_info[stat_name]

def calc_distribution_max(
    diff: XR_DATAARRAY
) -> NP_REAL:
    diff_values: NP_ARRAY[NP_REAL] = np.asarray(diff, dtype = NP_REAL)

    return NP_REAL(np.nanmax(np.abs(diff_values)))

def calc_distribution_range_max_from_dataset(
    dataset: xr.Dataset,
    var_name: str
) -> NP_REAL:
    min_values: NP_ARRAY[NP_REAL] = np.asarray(
        dataset[var_name].sel(stat = "min"),
        dtype = NP_REAL
    )
    max_values: NP_ARRAY[NP_REAL] = np.asarray(
        dataset[var_name].sel(stat = "max"),
        dtype = NP_REAL
    )

    return NP_REAL(np.nanmax(max_values - min_values))

def get_distribution_info_from_dataset(
    dataset: xr.Dataset,
    var_name: str,
    solver_name: str,
    day_index: NP_INT
) -> dict:
    distribution_info: dict = {}

    ss: int
    for ss in range(0, len(dist_stat_names)):
        stat_name: str = dist_stat_names[ss]
        distribution_info[stat_name] = np.asarray(
            dataset[var_name].sel(
                solver = solver_name,
                day = day_index,
                stat = stat_name
            ),
            dtype = NP_REAL
        )

    return distribution_info

def calc_ml3drt_distribution_dataset(
    rad_tran_infile: str,
    rad_tran_outfile: str,
    ml3drt_outfile: str,
    coarse_factor_str: str,
    daytime_indices: NP_ARRAY[NP_INT],
    daytime_times: NP_ARRAY[NP_REAL],
    daytime_szas: NP_ARRAY[NP_REAL],
    z_max_info: dict,
    z_max: Optional[NP_REAL]
) -> xr.Dataset:
    nsolvers: NP_INT = NP_INT(len(solver_names))
    ndays: NP_INT = NP_INT(daytime_indices.shape[0])
    ntime: NP_INT = NP_INT(daytime_indices.shape[1])
    nstats: NP_INT = NP_INT(len(dist_stat_names))

    distribution_shape: tuple[int, int, int, int] = (
        int(nsolvers),
        int(ndays),
        int(ntime),
        int(nstats)
    )

    reflectance_diff_array: NP_ARRAY[NP_REAL] = np.full(
        distribution_shape,
        np.nan,
        dtype = NP_REAL
    )
    heating_diff_array: NP_ARRAY[NP_REAL] = np.full(
        distribution_shape,
        np.nan,
        dtype = NP_REAL
    )
    flux_sfc_dn_diff_array: NP_ARRAY[NP_REAL] = np.full(
        distribution_shape,
        np.nan,
        dtype = NP_REAL
    )

    reflectance_diff_max: NP_ARRAY[NP_REAL] = np.full(
        nsolvers,
        -NP_INF,
        dtype = NP_REAL
    )
    heating_diff_max: NP_ARRAY[NP_REAL] = np.full(
        nsolvers,
        -NP_INF,
        dtype = NP_REAL
    )
    flux_sfc_dn_diff_max: NP_ARRAY[NP_REAL] = np.full(
        nsolvers,
        -NP_INF,
        dtype = NP_REAL
    )

    jj: int
    for jj in range(0, ndays):
        msg: str = "- Day {}...".format(jj)
        rte_rrtmgp_cpp_print_msg(msg)

        #-----------------------------------------------------------------------
        # Calculate reflectance
        #-----------------------------------------------------------------------
        msg: str = "-- Reflectance..."
        rte_rrtmgp_cpp_print_msg(msg)

        reflectance_rt: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_reflectance(
            rad_tran_infile,
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "rt") # Shortwave reflectance, ray-tracer; [N/A]; [time, y, x]

        reflectance_emulator: XR_DATAARRAY = ml3drt_calc_sw_reflectance(
            rad_tran_infile,
            ml3drt_outfile,
            time_indices = daytime_indices[jj,...]) # Shortwave reflectance, emulator; [N/A]; [time, y, x]

        reflectance_ts: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_reflectance(
            rad_tran_infile,
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "ts") # Shortwave reflectance, two-stream; [N/A]; [time, y, x]

        reflectance_emulator_diff: XR_DATAARRAY = (
            (reflectance_emulator - reflectance_rt)
            .stack(spatial = ("y", "x"))
            .reset_index("spatial")
        )

        reflectance_ts_diff: XR_DATAARRAY = (
            (reflectance_ts - reflectance_rt)
            .stack(spatial = ("y", "x"))
            .reset_index("spatial")
        )

        reflectance_diff_max[0] = max(
            reflectance_diff_max[0],
            calc_distribution_max(reflectance_emulator_diff)
        )
        reflectance_diff_max[1] = max(
            reflectance_diff_max[1],
            calc_distribution_max(reflectance_ts_diff)
        )

        add_distribution_info_to_array(
            reflectance_diff_array,
            NP_INT(0),
            NP_INT(jj),
            calc_distribution_info(reflectance_emulator_diff)
        )
        add_distribution_info_to_array(
            reflectance_diff_array,
            NP_INT(1),
            NP_INT(jj),
            calc_distribution_info(reflectance_ts_diff)
        )

        #-----------------------------------------------------------------------
        # Calculate heating rates
        #-----------------------------------------------------------------------
        msg: str = "-- Heating..."
        rte_rrtmgp_cpp_print_msg(msg)

        heating_rt: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_heating(
            rad_tran_infile,
            rad_tran_outfile,
            z_max_info = z_max_info,
            time_indices = daytime_indices[jj,...],
            solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

        heating_emulator: XR_DATAARRAY = ml3drt_calc_sw_heating(
            rad_tran_infile,
            ml3drt_outfile,
            z_max_info = z_max_info,
            time_indices = daytime_indices[jj,...]) # Shortwave heating rate, emulator; [K d^{-1}]; [ntime, lay, y, x]

        heating_ts: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_heating(
            rad_tran_infile,
            rad_tran_outfile,
            z_max_info = z_max_info,
            time_indices = daytime_indices[jj,...],
            solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

        heating_emulator_diff: XR_DATAARRAY = (
            (heating_emulator - heating_rt)
            .stack(spatial = ("lay", "y", "x"))
            .reset_index("spatial")
        )

        heating_ts_diff: XR_DATAARRAY = (
            (heating_ts - heating_rt)
            .stack(spatial = ("lay", "y", "x"))
            .reset_index("spatial")
        )

        heating_diff_max[0] = max(
            heating_diff_max[0],
            calc_distribution_max(heating_emulator_diff)
        )
        heating_diff_max[1] = max(
            heating_diff_max[1],
            calc_distribution_max(heating_ts_diff)
        )

        add_distribution_info_to_array(
            heating_diff_array,
            NP_INT(0),
            NP_INT(jj),
            calc_distribution_info(heating_emulator_diff)
        )
        add_distribution_info_to_array(
            heating_diff_array,
            NP_INT(1),
            NP_INT(jj),
            calc_distribution_info(heating_ts_diff)
        )

        #-----------------------------------------------------------------------
        # Calculate downwelling surface flux
        #-----------------------------------------------------------------------
        msg: str = "-- Downwelling Surface Flux..."
        rte_rrtmgp_cpp_print_msg(msg)

        flux_sfc_dn_rt: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_flux_sfc_dn(
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_emulator: XR_DATAARRAY = ml3drt_calc_sw_flux_sfc_dn(
            ml3drt_outfile,
            time_indices = daytime_indices[jj,...]) # Shortwave downwelling surface flux, emulator; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_ts: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_flux_sfc_dn(
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_emulator_diff: XR_DATAARRAY = (
            (flux_sfc_dn_emulator - flux_sfc_dn_rt)
            .stack(spatial = ("y", "x"))
            .reset_index("spatial")
        )

        flux_sfc_dn_ts_diff: XR_DATAARRAY = (
            (flux_sfc_dn_ts - flux_sfc_dn_rt)
            .stack(spatial = ("y", "x"))
            .reset_index("spatial")
        )

        flux_sfc_dn_diff_max[0] = max(
            flux_sfc_dn_diff_max[0],
            calc_distribution_max(flux_sfc_dn_emulator_diff)
        )
        flux_sfc_dn_diff_max[1] = max(
            flux_sfc_dn_diff_max[1],
            calc_distribution_max(flux_sfc_dn_ts_diff)
        )

        add_distribution_info_to_array(
            flux_sfc_dn_diff_array,
            NP_INT(0),
            NP_INT(jj),
            calc_distribution_info(flux_sfc_dn_emulator_diff)
        )
        add_distribution_info_to_array(
            flux_sfc_dn_diff_array,
            NP_INT(1),
            NP_INT(jj),
            calc_distribution_info(flux_sfc_dn_ts_diff)
        )

    #---------------------------------------------------------------------------
    # Obtain grid information
    #---------------------------------------------------------------------------
    msg: str = "Obtaining grid information..."
    rte_rrtmgp_cpp_print_msg(msg)
    grid: dict = rte_rrtmgp_cpp_find_grid(rad_tran_infile)

    dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]

    dataset: xr.Dataset = xr.Dataset(
        data_vars = {
            "daytime_time": (
                ("day", "time_index"),
                np.asarray(daytime_times, dtype = NP_REAL),
                {
                    "long_name": "time since simulation start",
                    "units": "h"
                }
            ),
            "daytime_sza": (
                ("day", "time_index"),
                np.asarray(daytime_szas, dtype = NP_REAL),
                {
                    "long_name": "solar zenith angle",
                    "units": "degree"
                }
            ),
            "reflectance_diff_dist": (
                ("solver", "day", "time_index", "stat"),
                reflectance_diff_array,
                {
                    "long_name": "emulator or two-stream minus ray-tracer shortwave reflectance distribution",
                    "units": "1"
                }
            ),
            "heating_diff_dist": (
                ("solver", "day", "time_index", "stat"),
                heating_diff_array,
                {
                    "long_name": "emulator or two-stream minus ray-tracer shortwave heating rate distribution",
                    "units": "K d-1"
                }
            ),
            "flux_sfc_dn_diff_dist": (
                ("solver", "day", "time_index", "stat"),
                flux_sfc_dn_diff_array,
                {
                    "long_name": "emulator or two-stream minus ray-tracer shortwave downwelling surface flux distribution",
                    "units": "W m-2"
                }
            )
        },
        coords = {
            "solver": solver_names,
            "day": np.arange(0, ndays, dtype = NP_INT),
            "time_index": np.arange(0, ntime, dtype = NP_INT),
            "stat": dist_stat_names
        },
        attrs = {
            "coarse_factor": coarse_factor_str,
            "dx_m": float(dx),
            "z_max_km": float(z_max) if z_max is not None else np.nan,
            "reflectance_emulator_diff_max": float(reflectance_diff_max[0]),
            "reflectance_twostream_diff_max": float(reflectance_diff_max[1]),
            "heating_emulator_diff_max": float(heating_diff_max[0]),
            "heating_twostream_diff_max": float(heating_diff_max[1]),
            "flux_sfc_dn_emulator_diff_max": float(flux_sfc_dn_diff_max[0]),
            "flux_sfc_dn_twostream_diff_max": float(flux_sfc_dn_diff_max[1])
        }
    )

    return dataset

def plot_distribution_range_curves(
    ax: MPL_AXES,
    time: NP_ARRAY[NP_REAL],
    distribution_info: dict,
    line_color: str
):
    interquantile_range: NP_ARRAY[NP_REAL] = NP_REAL(
        distribution_info["p90"] - distribution_info["p10"]
    )
    minmax_range: NP_ARRAY[NP_REAL] = NP_REAL(
        distribution_info["max"] - distribution_info["min"]
    )

    # 10-90 inter-quantile range
    ax.plot(
        time,
        interquantile_range,
        color = line_color,
        linestyle = "solid",
        linewidth = 2.0
    )

    # Min-max range
    ax.plot(
        time,
        minmax_range,
        color = line_color,
        linestyle = "dashed",
        linewidth = 2.0
    )

def reorder_legend_entries(handles: list, labels: list[str], ncols: NP_INT):
    nentries: NP_INT = NP_INT(len(labels))

    if nentries <= 0 or ncols <= 1:
        return handles, labels

    col_indices: list[NP_ARRAY[NP_INT]] = np.array_split(
        np.arange(0, nentries, dtype = NP_INT),
        ncols)

    reorder_indices: list[NP_INT] = []
    col: NP_INT
    for col in range(0, ncols):
        nrows_col: NP_INT = NP_INT(col_indices[col].size)

        row: NP_INT
        for row in range(0, nrows_col):
            reorder_index: NP_INT = NP_INT(row * ncols + col)

            if reorder_index < nentries:
                reorder_indices += [reorder_index]

    handles = [handles[ii] for ii in reorder_indices]
    labels = [labels[ii] for ii in reorder_indices]

    return handles, labels

def linlog_forward(
    values: NP_ARRAY[NP_REAL],
    linthresh: NP_REAL
) -> NP_ARRAY[NP_REAL]:
    values: NP_ARRAY[NP_REAL] = np.asarray(values, dtype = NP_REAL)
    abs_values: NP_ARRAY[NP_REAL] = np.abs(values)
    sign_values: NP_ARRAY[NP_REAL] = np.sign(values)

    transformed_values: NP_ARRAY[NP_REAL] = np.where(
        abs_values <= linthresh,
        abs_values,
        linthresh * (1.0 + np.log10(abs_values / linthresh))
    )

    return sign_values * transformed_values

def linlog_inverse(
    values: NP_ARRAY[NP_REAL],
    linthresh: NP_REAL
) -> NP_ARRAY[NP_REAL]:
    values: NP_ARRAY[NP_REAL] = np.asarray(values, dtype = NP_REAL)
    abs_values: NP_ARRAY[NP_REAL] = np.abs(values)
    sign_values: NP_ARRAY[NP_REAL] = np.sign(values)

    inverse_values: NP_ARRAY[NP_REAL] = np.where(
        abs_values <= linthresh,
        abs_values,
        linthresh * np.power(10.0, abs_values / linthresh - 1.0)
    )

    return sign_values * inverse_values

def calc_linlog_ticks(
    ymax: NP_REAL,
    linthresh: NP_REAL
) -> NP_ARRAY[NP_REAL]:
    ticks: list[NP_REAL] = [
        NP_REAL(0.0)
    ]

    if ymax <= 0:
        return np.array(ticks, dtype = NP_REAL)

    exponent_start: int = int(np.ceil(np.log10(linthresh)))
    exponent_end: int = int(np.floor(np.log10(ymax)))

    exponent: int
    for exponent in range(exponent_start, exponent_end + 1):
        tick: NP_REAL = NP_REAL(np.power(10.0, exponent))

        if tick >= linthresh and tick <= ymax:
            ticks += [tick]

    ticks = sorted(list(set(ticks)))

    return np.array(ticks, dtype = NP_REAL)

def calc_linlog_ticklabels(
    ticks: NP_ARRAY[NP_REAL]
) -> list[str]:
    ticklabels: list[str] = []

    tick: NP_REAL
    for tick in ticks:
        if tick == 0:
            ticklabels += [r"$0$"]
        else:
            exponent: int = int(np.round(np.log10(tick)))
            ticklabels += [r"$10^{{{:d}}}$".format(exponent)]

    return ticklabels

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    msg: str = "Parsing command-line input..."
    rte_rrtmgp_cpp_print_msg(msg)

    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory.")
    parser.add_argument("--rad-tran-outdir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT combined output directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--ml3drt-outfile", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for ML3DRT output file.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate distribution quantities and save them to NetCDF files.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    ml3drt_outfile: str = os.path.normpath(args.ml3drt_outfile)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

    lr_re: re.Pattern = re.compile(r"lr_[0-9]+")
    lr_match: Optional[re.Match] = lr_re.search(os.path.basename(ml3drt_outfile))
    if lr_match is None:
        lr_match = lr_re.search(ml3drt_outfile)

    if lr_match is None:
        raise ValueError("Could not extract lr coarsening factor from ML3DRT output file path.")

    coarse_factor_str: str = lr_match.group()
    coarse_factor: NP_INT = NP_INT(int(coarse_factor_str.replace("lr_", "")))
    coarse_factors: NP_ARRAY[NP_INT] = np.array([coarse_factor], dtype = NP_INT)

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Find file pair at requested resolution
    #---------------------------------------------------------------------------
    rad_tran_infiles: list[str]
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = rte_rrtmgp_cpp_find_inout_pairs(
        rad_tran_indir,
        rad_tran_outdir,
        coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    if nfiles <= 0:
        raise ValueError("No RTE-RRTMGP-CPP input/output file pairs were found for {}.".format(coarse_factor_str))

    if nfiles > 1:
        raise ValueError("Multiple RTE-RRTMGP-CPP input/output file pairs were found for {}; expected exactly one.".format(coarse_factor_str))

    rad_tran_infile: str = rad_tran_infiles[0]
    rad_tran_outfile: str = rad_tran_outfiles[0]

    #---------------------------------------------------------------------------
    # Determine whether distribution information needs to be calculated
    #---------------------------------------------------------------------------
    nc_filename: str = "ml3drt_quantile_timeseries.{}.nc".format(coarse_factor_str)
    nc_filepath: str = os.path.join(working_dir, nc_filename)

    recalculate_distribution: bool = args.recalculate or not os.path.exists(nc_filepath)

    #---------------------------------------------------------------------------
    # Load or calculate distribution information
    #---------------------------------------------------------------------------
    msg: str = "Loading or calculating distributions..."
    rte_rrtmgp_cpp_print_msg(msg)

    if recalculate_distribution:
        if args.recalculate:
            msg: str = "Recalculating {}...".format(coarse_factor_str)
        else:
            msg: str = "NetCDF file not found for {}; calculating...".format(coarse_factor_str)
        rte_rrtmgp_cpp_print_msg(msg)

        #-----------------------------------------------------------------------
        # Calculate quantities common to the resolution
        #-----------------------------------------------------------------------
        msg: str = "Calculating quantities common to the resolution..."
        rte_rrtmgp_cpp_print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = rte_rrtmgp_cpp_find_daytime_indices(
            rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
        daytime_times: NP_ARRAY[NP_REAL] = rte_rrtmgp_cpp_find_times(
            rad_tran_infile,
            daytime_indices) # Time since simulation start; [h]; [ndays, time_per_day]
        daytime_szas: NP_ARRAY[NP_REAL] = rte_rrtmgp_cpp_find_szas(
            rad_tran_infile,
            daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, time_per_day]
        z_max_info: dict = rte_rrtmgp_cpp_calc_z_max_info(
            rad_tran_infile,
            z_max = z_max) #

        distribution_dataset: xr.Dataset = calc_ml3drt_distribution_dataset(
            rad_tran_infile,
            rad_tran_outfile,
            ml3drt_outfile,
            coarse_factor_str,
            daytime_indices,
            daytime_times,
            daytime_szas,
            z_max_info,
            z_max
        )

        msg: str = "Writing {}...".format(nc_filepath)
        rte_rrtmgp_cpp_print_msg(msg)
        distribution_dataset.to_netcdf(nc_filepath)
    else:
        msg: str = "Reading {}...".format(nc_filepath)
        rte_rrtmgp_cpp_print_msg(msg)

    distribution_dataset: xr.Dataset = xr.load_dataset(nc_filepath)

    #---------------------------------------------------------------------------
    # Calculate global y-axis limits from loaded distribution data
    #---------------------------------------------------------------------------
    reflectance_range_max: NP_REAL = calc_distribution_range_max_from_dataset(
        distribution_dataset,
        "reflectance_diff_dist"
    )
    heating_range_max: NP_REAL = calc_distribution_range_max_from_dataset(
        distribution_dataset,
        "heating_diff_dist"
    )
    flux_sfc_dn_range_max: NP_REAL = calc_distribution_range_max_from_dataset(
        distribution_dataset,
        "flux_sfc_dn_diff_dist"
    )

    #---------------------------------------------------------------------------
    # Set up figure for plotting
    #---------------------------------------------------------------------------
    msg: str = "Setting up figure..."
    rte_rrtmgp_cpp_print_msg(msg)

    ndays: NP_INT = NP_INT(distribution_dataset.sizes["day"])

    nrows: NP_INT = NP_INT(3)
    ncols: NP_INT = NP_INT(ndays)
    fig_height: NP_REAL =  NP_REAL(8.0)
    fig_width: NP_REAL = (NP_REAL(ncols) / NP_REAL(nrows)) * fig_height
    fig_size: list[NP_REAL] = [fig_width, fig_height]
    fig: MPL_FIGURE
    axs: MPL_AXES
    fig, axs = plt.subplots(
        nrows = nrows, ncols = ncols,
        sharex = "col", sharey = "row",
        constrained_layout = True,
        figsize = fig_size,
        squeeze = False)

    #---------------------------------------------------------------------------
    # Create solver labels and legend handles
    #---------------------------------------------------------------------------
    legend_handles: list = []
    legend_labels: list[str] = []

    nplot_colors: NP_INT = NP_INT(len(plot_colors))
    nsolvers: NP_INT = NP_INT(len(solver_names))

    ii: int
    for ii in range(0, nsolvers):
        legend_handles += [
            Patch(
                facecolor = plot_colors[ii % nplot_colors],
                edgecolor = plot_colors[ii % nplot_colors]
            )
        ]
        legend_labels += [solver_label_names[ii]]

    #---------------------------------------------------------------------------
    # Plot distributions across each day
    #---------------------------------------------------------------------------
    msg: str = "Plotting distribution range curves..."
    rte_rrtmgp_cpp_print_msg(msg)

    jj: int
    for jj in range(0, ndays):
        msg: str = "- Day {}...".format(jj)
        rte_rrtmgp_cpp_print_msg(msg)

        reference_time: NP_ARRAY[NP_REAL] = np.asarray(
            distribution_dataset["daytime_time"].sel(day = jj),
            dtype = NP_REAL
        )
        reference_sza: NP_ARRAY[NP_REAL] = np.asarray(
            distribution_dataset["daytime_sza"].sel(day = jj),
            dtype = NP_REAL
        )

        #-----------------------------------------------------------------------
        # Plot each solver
        #-----------------------------------------------------------------------
        ii: int
        for ii in range(0, nsolvers):
            solver_name: str = solver_names[ii]
            line_color: str = plot_colors[ii % nplot_colors]

            time: NP_ARRAY[NP_REAL] = np.asarray(
                distribution_dataset["daytime_time"].sel(day = jj),
                dtype = NP_REAL
            )

            # Row 0 - Reflectance
            row: NP_INT = NP_INT(0)
            plot_distribution_range_curves(
                axs[row,jj],
                time,
                get_distribution_info_from_dataset(
                    distribution_dataset,
                    "reflectance_diff_dist",
                    solver_name,
                    NP_INT(jj)
                ),
                line_color
            )

            # Row 1 - Heating
            row: NP_INT = NP_INT(1)
            plot_distribution_range_curves(
                axs[row,jj],
                time,
                get_distribution_info_from_dataset(
                    distribution_dataset,
                    "heating_diff_dist",
                    solver_name,
                    NP_INT(jj)
                ),
                line_color
            )

            # Row 2 - Downwelling Surface Flux
            row: NP_INT = NP_INT(2)
            plot_distribution_range_curves(
                axs[row,jj],
                time,
                get_distribution_info_from_dataset(
                    distribution_dataset,
                    "flux_sfc_dn_diff_dist",
                    solver_name,
                    NP_INT(jj)
                ),
                line_color
            )

        #-----------------------------------------------------------------------
        # Add common column-wise plot elements
        #-----------------------------------------------------------------------
        # x-ticks
        xlim: NP_ARRAY[NP_REAL] = np.array(
            [reference_time[0], reference_time[-1]],
            dtype = NP_REAL
        )
        time_xticks: NP_ARRAY[NP_REAL] = np.array(
            [
                reference_time[0],
                reference_time[NP_INT(reference_time.size/2)],
                reference_time[-1]
            ],
            dtype = NP_REAL
        )
        sza_xticks: NP_ARRAY[NP_REAL] = np.array(
            [
                reference_sza[0],
                reference_sza[NP_INT(reference_time.size/2)],
                reference_sza[-1]
            ],
            dtype = NP_REAL
        )
        sza_xtick_labels: list[str] = [
            r"{:.1f}$^{{\circ}}$".format(solar_zenith_angle)
            for solar_zenith_angle in sza_xticks
        ]

        ll: int
        for ll in range(0, nrows):
            axs[ll,jj].set_xlim(xlim)
            axs[ll,jj].set_xticks(time_xticks)
            axs[ll,jj].axvline(time_xticks[1],
                color = "gray",
                linestyle = "solid",
                linewidth = 0.5)

            ax_2: MPL_AXES = axs[ll,jj].secondary_xaxis("top")
            if ll == 0:
                ax_2.set_xticks(time_xticks, labels = sza_xtick_labels)
            else:
                ax_2.set_xticks(time_xticks, labels = [None, None, None])

        axs[0,jj].set_title(
            "Day {}".format(jj),
            pad = 24.0
        )

    #---------------------------------------------------------------------------
    # Add plot elements
    #---------------------------------------------------------------------------
    title_str: str = r"Inter-Quantile Range"
    fig.suptitle(title_str,
        y = 1.10)
    fig.supxlabel(r"Time $\left[ h \right]$")

    axs[0,0].set_ylabel(r"Reflectance")
    axs[1,0].set_ylabel(r"Heating Rate $\left[ K\,d^{-1} \right]$")
    axs[2,0].set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")

    nlegend_cols: NP_INT = NP_INT(min(len(legend_labels), 3))
    legend_handles, legend_labels = reorder_legend_entries(
        legend_handles,
        legend_labels,
        nlegend_cols)

    fig.legend(
        legend_handles,
        legend_labels,
        loc = "upper center",
        bbox_to_anchor = (0.5, 1.085),
        ncol = nlegend_cols,
        handlelength = 2.0,
        columnspacing = 1.2,
        handletextpad = 0.4
    )

    # Axes Scaling, Limits
    eps_array: NP_ARRAY[NP_REAL] = np.array(
        [0.2, 1.0, 100.0],
        dtype = NP_REAL
    )

    ylim_array: NP_ARRAY[NP_REAL] = np.array(
        [
            reflectance_range_max,
            heating_range_max,
            flux_sfc_dn_range_max
        ],
        dtype = NP_REAL
    )

    kk: int
    for kk in range(0, nrows):
        ymax: NP_REAL = NP_REAL(ylim_array[kk])

        if not np.isfinite(ymax) or ymax <= 0:
            ymax = NP_REAL(eps_array[kk])

        ymax = NP_REAL(max(ymax, eps_array[kk]))
        ymax = NP_REAL(1.05 * ymax)

        jj: int
        for jj in range(0, ndays):
            if kk > 0:
                linthresh: NP_REAL = NP_REAL(eps_array[kk])

                axs[kk,jj].set_yscale(
                    "function",
                    functions = (
                        lambda values, linthresh = linthresh: linlog_forward(
                            values,
                            linthresh
                        ),
                        lambda values, linthresh = linthresh: linlog_inverse(
                            values,
                            linthresh
                        )
                    )
                )

                yticks: NP_ARRAY[NP_REAL] = calc_linlog_ticks(
                    ymax,
                    linthresh
                )

                axs[kk,jj].set_yticks(yticks)
                axs[kk,jj].set_yticklabels(
                    calc_linlog_ticklabels(yticks)
                )
            else:
                axs[kk,jj].set_yscale("linear")

            axs[kk,jj].set_ylim([0, ymax])

            axs[kk,jj].axhline(
                eps_array[kk],
                color = "gray",
                linestyle = "solid",
                linewidth = 0.5
            )

    # Line-style legend
    style_handles: list = [
        Line2D(
            [0],
            [0],
            color = "black",
            linestyle = "solid",
            linewidth = 2.0
        ),
        Line2D(
            [0],
            [0],
            color = "black",
            linestyle = "dashed",
            linewidth = 2.0
        )
    ]
    style_labels: list[str] = [
        r"10%-90% Inter-Quantile Range",
        "Min.-Max. Range"
    ]

    fig.legend(
        style_handles,
        style_labels,
        loc = "upper center",
        bbox_to_anchor = (0.5, 1.04),
        ncol = len(style_labels),
        handlelength = 2.0,
        columnspacing = 1.2,
        handletextpad = 0.4
    )

    #---------------------------------------------------------------------------
    # Save the plot to file
    #---------------------------------------------------------------------------
    plt_filename = "ml3drt_quantile_timeseries.png"
    plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
    fig.savefig(plt_filepath, dpi = 512, bbox_inches = "tight")
    plt.close(fig)

if __name__ == "__main__":
    main()