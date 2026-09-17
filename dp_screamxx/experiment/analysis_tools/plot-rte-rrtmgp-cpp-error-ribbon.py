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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, \
    MPL_FIGURE, MPL_AXES
from consts.numeric import NP_INF
from consts.visual import plot_colors
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_sw_reflectance, calc_sw_heating, calc_sw_flux_sfc_dn, \
    find_grid, calc_z_max_info, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-error-distribution"
prog_desc: str = "Visualize distributions of two-stream and ray-tracer solver differences for RTE-RRTMGP-CPP."

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
    day_index: NP_INT,
    distribution_info: dict
):
    ss: int
    for ss in range(0, len(dist_stat_names)):
        stat_name: str = dist_stat_names[ss]
        distribution_array[day_index,:,ss] = distribution_info[stat_name]

def calc_distribution_max(
    diff: XR_DATAARRAY
) -> NP_REAL:
    diff_values: NP_ARRAY[NP_REAL] = np.asarray(diff, dtype = NP_REAL)

    return NP_REAL(np.nanmax(np.abs(diff_values)))

def calc_distribution_max_from_dataset(
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

    return NP_REAL(np.nanmax(np.abs(np.concatenate(
        [
            min_values.ravel(),
            max_values.ravel()
        ]
    ))))

def get_distribution_info_from_dataset(
    dataset: xr.Dataset,
    var_name: str,
    day_index: NP_INT
) -> dict:
    distribution_info: dict = {}

    ss: int
    for ss in range(0, len(dist_stat_names)):
        stat_name: str = dist_stat_names[ss]
        distribution_info[stat_name] = np.asarray(
            dataset[var_name].sel(day = day_index, stat = stat_name),
            dtype = NP_REAL
        )

    return distribution_info

def calc_resolution_distribution_dataset(
    rad_tran_infile: str,
    rad_tran_outfile: str,
    coarse_factor_str: str,
    daytime_indices: NP_ARRAY[NP_INT],
    daytime_times: NP_ARRAY[NP_REAL],
    daytime_szas: NP_ARRAY[NP_REAL],
    z_max_info: dict,
    z_max: Optional[NP_REAL]
) -> xr.Dataset:
    ndays: NP_INT = NP_INT(daytime_indices.shape[0])
    ntime: NP_INT = NP_INT(daytime_indices.shape[1])
    nstats: NP_INT = NP_INT(len(dist_stat_names))

    distribution_shape: tuple[int, int, int] = (
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

    reflectance_diff_max: NP_REAL = NP_REAL(-NP_INF)
    heating_diff_max: NP_REAL = NP_REAL(-NP_INF)
    flux_sfc_dn_diff_max: NP_REAL = NP_REAL(-NP_INF)

    jj: int
    for jj in range(0, ndays):
        msg: str = "- Day {}...".format(jj)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Calculate reflectance
        #-----------------------------------------------------------------------
        msg: str = "-- Reflectance..."
        print_msg(msg)

        reflectance_rt: XR_DATAARRAY = calc_sw_reflectance(
            rad_tran_infile,
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "rt") # Shortwave reflectance, ray-tracer; [N/A]; [time, y, x]

        reflectance_ts: XR_DATAARRAY = calc_sw_reflectance(
            rad_tran_infile,
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "ts") # Shortwave reflectance, two-stream; [N/A]; [time, y, x]

        reflectance_diff: XR_DATAARRAY = (
            (reflectance_ts - reflectance_rt)
            .stack(spatial = ("y", "x"))
            .reset_index("spatial")
        )

        reflectance_diff_max = max(
            reflectance_diff_max,
            calc_distribution_max(reflectance_diff)
        )

        add_distribution_info_to_array(
            reflectance_diff_array,
            NP_INT(jj),
            calc_distribution_info(reflectance_diff)
        )

        #-----------------------------------------------------------------------
        # Calculate heating rates
        #-----------------------------------------------------------------------
        msg: str = "-- Heating..."
        print_msg(msg)

        heating_rt: XR_DATAARRAY = calc_sw_heating(
            rad_tran_infile,
            rad_tran_outfile,
            z_max_info = z_max_info,
            time_indices = daytime_indices[jj,...],
            solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [ntime, lay, y, x]

        heating_ts: XR_DATAARRAY = calc_sw_heating(
            rad_tran_infile,
            rad_tran_outfile,
            z_max_info = z_max_info,
            time_indices = daytime_indices[jj,...],
            solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [ntime, lay, y, x]

        heating_diff: XR_DATAARRAY = (
            (heating_ts - heating_rt)
            .stack(spatial = ("lay", "y", "x"))
            .reset_index("spatial")
        )

        heating_diff_max = max(
            heating_diff_max,
            calc_distribution_max(heating_diff)
        )

        add_distribution_info_to_array(
            heating_diff_array,
            NP_INT(jj),
            calc_distribution_info(heating_diff)
        )

        #-----------------------------------------------------------------------
        # Calculate downwelling surface flux
        #-----------------------------------------------------------------------
        msg: str = "-- Downwelling Surface Flux..."
        print_msg(msg)

        flux_sfc_dn_rt: XR_DATAARRAY = calc_sw_flux_sfc_dn(
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_ts: XR_DATAARRAY = calc_sw_flux_sfc_dn(
            rad_tran_outfile,
            time_indices = daytime_indices[jj,...],
            solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

        flux_sfc_dn_diff: XR_DATAARRAY = (
            (flux_sfc_dn_ts - flux_sfc_dn_rt)
            .stack(spatial = ("y", "x"))
            .reset_index("spatial")
        )

        flux_sfc_dn_diff_max = max(
            flux_sfc_dn_diff_max,
            calc_distribution_max(flux_sfc_dn_diff)
        )

        add_distribution_info_to_array(
            flux_sfc_dn_diff_array,
            NP_INT(jj),
            calc_distribution_info(flux_sfc_dn_diff)
        )

    #---------------------------------------------------------------------------
    # Obtain grid information
    #---------------------------------------------------------------------------
    msg: str = "Obtaining grid information..."
    print_msg(msg)
    grid: dict = find_grid(rad_tran_infile)

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
                ("day", "time_index", "stat"),
                reflectance_diff_array,
                {
                    "long_name": "two-stream minus ray-tracer shortwave reflectance distribution",
                    "units": "1"
                }
            ),
            "heating_diff_dist": (
                ("day", "time_index", "stat"),
                heating_diff_array,
                {
                    "long_name": "two-stream minus ray-tracer shortwave heating rate distribution",
                    "units": "K d-1"
                }
            ),
            "flux_sfc_dn_diff_dist": (
                ("day", "time_index", "stat"),
                flux_sfc_dn_diff_array,
                {
                    "long_name": "two-stream minus ray-tracer shortwave downwelling surface flux distribution",
                    "units": "W m-2"
                }
            )
        },
        coords = {
            "day": np.arange(0, ndays, dtype = NP_INT),
            "time_index": np.arange(0, ntime, dtype = NP_INT),
            "stat": dist_stat_names
        },
        attrs = {
            "coarse_factor": coarse_factor_str,
            "dx_m": float(dx),
            "z_max_km": float(z_max) if z_max is not None else np.nan,
            "reflectance_diff_max": float(reflectance_diff_max),
            "heating_diff_max": float(heating_diff_max),
            "flux_sfc_dn_diff_max": float(flux_sfc_dn_diff_max)
        }
    )

    return dataset

def plot_distribution_ribbon(
    ax: MPL_AXES,
    time: NP_ARRAY[NP_REAL],
    distribution_info: dict,
    ribbon_color: str,
    add_legend_labels: bool = False
):
    min_max_label: str = "Min-Max" if add_legend_labels else "_nolegend_"
    p10_p90_label: str = "10-90" if add_legend_labels else "_nolegend_"
    p20_p80_label: str = "20-80" if add_legend_labels else "_nolegend_"
    p40_p60_label: str = "40-60" if add_legend_labels else "_nolegend_"
    median_label: str = "Median" if add_legend_labels else "_nolegend_"

    # Min-max
    ax.plot(
        time,
        distribution_info["min"],
        color = ribbon_color,
        linestyle = "dashed",
        linewidth = 1.0,
        label = min_max_label
    )
    ax.plot(
        time,
        distribution_info["max"],
        color = ribbon_color,
        linestyle = "dashed",
        linewidth = 1.0,
        label = "_nolegend_"
    )

    # 10-90 percentile range
    ax.fill_between(
        time,
        distribution_info["p10"],
        distribution_info["p90"],
        color = ribbon_color,
        alpha = 0.15,
        linewidth = 0.0,
        label = p10_p90_label
    )

    # 20-80 percentile range
    ax.fill_between(
        time,
        distribution_info["p20"],
        distribution_info["p80"],
        color = ribbon_color,
        alpha = 0.25,
        linewidth = 0.0,
        label = p20_p80_label
    )

    # 40-60 percentile range
    ax.fill_between(
        time,
        distribution_info["p40"],
        distribution_info["p60"],
        color = ribbon_color,
        alpha = 0.40,
        linewidth = 0.0,
        label = p40_p60_label
    )

    # Median
    ax.plot(
        time,
        distribution_info["median"],
        color = ribbon_color,
        linestyle = "solid",
        linewidth = 2.0,
        label = median_label
    )

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    msg: str = "Parsing command-line input..."
    print_msg(msg)

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
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate distribution quantities and save them to NetCDF files.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

    coarse_factors: Optional[NP_ARRAY[NP_INT]] = None
    if args.coarse_factors is not None:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Find file pairs at requested resolutions
    #---------------------------------------------------------------------------
    rad_tran_infiles: list[str]
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(rad_tran_indir,
        rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    #---------------------------------------------------------------------------
    # Determine which resolutions need to be calculated
    #---------------------------------------------------------------------------
    coarse_factor_strs: list[str] = []
    nc_filepaths: list[str] = []
    recalculate_resolution: list[bool] = []

    ii: int
    for ii in range(0, nfiles):
        coarse_factor_str: str = lr_re.search(rad_tran_infiles[ii]).group()
        nc_filename: str = "rte_rrtmgp_cpp_error_ribbon.{}.nc".format(coarse_factor_str)
        nc_filepath: str = os.path.join(working_dir, nc_filename)

        coarse_factor_strs.append(coarse_factor_str)
        nc_filepaths.append(nc_filepath)
        recalculate_resolution.append(args.recalculate or not os.path.exists(nc_filepath))

    need_calculation: bool = any(recalculate_resolution)

    #---------------------------------------------------------------------------
    # Calculate quantities that should be common across all resolutions
    #---------------------------------------------------------------------------
    daytime_indices: Optional[NP_ARRAY[NP_INT]] = None
    daytime_times: Optional[NP_ARRAY[NP_REAL]] = None
    daytime_szas: Optional[NP_ARRAY[NP_REAL]] = None
    z_max_info: Optional[dict] = None

    if need_calculation:
        msg: str = "Calculating quantities common across all resolutions..."
        print_msg(msg)

        daytime_indices = find_daytime_indices(
            rad_tran_infiles[0]) # Time indices for each day; [ndays; time_per_day]
        daytime_times = find_times(
            rad_tran_infiles[0],
            daytime_indices) # Time since simulation start; [h]; [ndays, time_per_day]
        daytime_szas = find_szas(
            rad_tran_infiles[0],
            daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, time_per_day]
        z_max_info = calc_z_max_info(
            rad_tran_infiles[0],
            z_max = z_max) #

    #---------------------------------------------------------------------------
    # Load or calculate distribution information for each requested resolution
    #---------------------------------------------------------------------------
    msg: str = "Loading or calculating distributions..."
    print_msg(msg)

    distribution_datasets: dict = {}

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]
        coarse_factor_str: str = coarse_factor_strs[ii]
        nc_filepath: str = nc_filepaths[ii]

        if recalculate_resolution[ii]:
            if args.recalculate:
                msg: str = "Recalculating {}...".format(coarse_factor_str)
            else:
                msg: str = "NetCDF file not found for {}; calculating...".format(coarse_factor_str)
            print_msg(msg)

            distribution_dataset: xr.Dataset = calc_resolution_distribution_dataset(
                rad_tran_infile,
                rad_tran_outfile,
                coarse_factor_str,
                daytime_indices,
                daytime_times,
                daytime_szas,
                z_max_info,
                z_max
            )

            msg: str = "Writing {}...".format(nc_filepath)
            print_msg(msg)
            distribution_dataset.to_netcdf(nc_filepath)
        else:
            msg: str = "Reading {}...".format(nc_filepath)
            print_msg(msg)

        distribution_datasets[coarse_factor_str] = xr.load_dataset(nc_filepath)

    #---------------------------------------------------------------------------
    # Calculate global y-axis limits from loaded distribution data
    #---------------------------------------------------------------------------
    reflectance_diff_max: NP_REAL = NP_REAL(-NP_INF)
    heating_diff_max: NP_REAL = NP_REAL(-NP_INF)
    flux_sfc_dn_diff_max: NP_REAL = NP_REAL(-NP_INF)

    ii: int
    for ii in range(0, nfiles):
        coarse_factor_str: str = coarse_factor_strs[ii]
        distribution_dataset: xr.Dataset = distribution_datasets[coarse_factor_str]

        reflectance_diff_max = max(
            reflectance_diff_max,
            calc_distribution_max_from_dataset(
                distribution_dataset,
                "reflectance_diff_dist"
            )
        )
        heating_diff_max = max(
            heating_diff_max,
            calc_distribution_max_from_dataset(
                distribution_dataset,
                "heating_diff_dist"
            )
        )
        flux_sfc_dn_diff_max = max(
            flux_sfc_dn_diff_max,
            calc_distribution_max_from_dataset(
                distribution_dataset,
                "flux_sfc_dn_diff_dist"
            )
        )

    #---------------------------------------------------------------------------
    # Set up figures for plotting
    #---------------------------------------------------------------------------
    ii: int
    for ii in range(0, nfiles):
        coarse_factor_str: str = coarse_factor_strs[ii]
        distribution_dataset: xr.Dataset = distribution_datasets[coarse_factor_str]

        msg: str = "Setting up figure for {}...".format(coarse_factor_str)
        print_msg(msg)

        ndays: NP_INT = NP_INT(distribution_dataset.sizes["day"])

        nrows: NP_INT = NP_INT(3)
        ncols: NP_INT = NP_INT(ndays)
        fig_width: NP_REAL = NP_REAL(6.5)
        fig_height: NP_REAL = (NP_REAL(nrows) / NP_REAL(ncols)) * fig_width
        fig_size: list[NP_REAL] = [fig_width, fig_height]
        fig: MPL_FIGURE
        axs: MPL_AXES
        fig, axs = plt.subplots(
            nrows = nrows, ncols = ncols,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = fig_size)

        if ncols == 1:
            axs = axs[...,None]
        elif nrows == 1:
            axs = axs[None,...]

        ribbon_color: str = plot_colors[2]

        #-----------------------------------------------------------------------
        # Plot distributions across each day
        #-----------------------------------------------------------------------
        msg: str = "Plotting distribution ribbons..."
        print_msg(msg)

        jj: int
        for jj in range(0, ndays):
            msg: str = "- Day {}...".format(jj)
            print_msg(msg)

            time: NP_ARRAY[NP_REAL] = np.asarray(
                distribution_dataset["daytime_time"].sel(day = jj),
                dtype = NP_REAL
            )
            sza: NP_ARRAY[NP_REAL] = np.asarray(
                distribution_dataset["daytime_sza"].sel(day = jj),
                dtype = NP_REAL
            )
            xlim: NP_ARRAY[NP_REAL] = np.array([time[0], time[-1]], dtype = NP_REAL)

            # Row 0 - Reflectance
            row: NP_INT = NP_INT(0)
            plot_distribution_ribbon(
                axs[row,jj],
                time,
                get_distribution_info_from_dataset(
                    distribution_dataset,
                    "reflectance_diff_dist",
                    NP_INT(jj)
                ),
                ribbon_color,
                add_legend_labels = jj == 0
            )

            # Row 1 - Heating
            row: NP_INT = NP_INT(1)
            plot_distribution_ribbon(
                axs[row,jj],
                time,
                get_distribution_info_from_dataset(
                    distribution_dataset,
                    "heating_diff_dist",
                    NP_INT(jj)
                ),
                ribbon_color
            )

            # Row 2 - Downwelling Surface Flux
            row: NP_INT = NP_INT(2)
            plot_distribution_ribbon(
                axs[row,jj],
                time,
                get_distribution_info_from_dataset(
                    distribution_dataset,
                    "flux_sfc_dn_diff_dist",
                    NP_INT(jj)
                ),
                ribbon_color
            )

            #-------------------------------------------------------------------
            # Add common column-wise plot elements
            #-------------------------------------------------------------------
            # x-ticks
            time_xticks: NP_ARRAY[NP_REAL] = np.array([time[0], time[NP_INT(time.size/2)], time[-1]], dtype = NP_REAL)
            sza_xticks: NP_ARRAY[NP_REAL] = np.array([sza[0], sza[NP_INT(time.size/2)], sza[-1]], dtype = NP_REAL)
            sza_xtick_labels: list[str] = [r"{:.1f}$^{{\circ}}$".format(solar_zenith_angle) for solar_zenith_angle in sza_xticks]
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

                axs[ll,jj].axhline(0,
                    color = "black",
                    linestyle = "solid",
                    linewidth = 0.5)

            axs[0,jj].set_title(
                "Day {}".format(jj),
                pad = 24.0
            )

        #-----------------------------------------------------------------------
        # Add plot elements
        #-----------------------------------------------------------------------
        dx: NP_REAL = NP_REAL(distribution_dataset.attrs["dx_m"]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

        fig.suptitle(r"Difference Distribution" + " - {}".format(dx_str),
            y = 1.065)
        fig.supxlabel(r"Time $\left[ h \right]$")
        fig.supylabel(r"Two-Stream - Ray-Tracer")

        axs[0,0].set_ylabel(r"Reflectance")
        axs[1,0].set_ylabel(r"Heating Rate $\left[ K\,d^{-1} \right]$")
        axs[2,0].set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")

        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc = "upper center",
            bbox_to_anchor = (0.5, 1.04),
            ncol = len(labels),
            handlelength = 2.0,
            columnspacing = 1.2,
            handletextpad = 0.4
        )

        # Symmetric y-axes with epsilon reference lines
        eps_array: NP_ARRAY[NP_REAL] = np.array(
            [0.2, 1.0, 100.0],
            dtype = NP_REAL
        )

        ylim_array: NP_ARRAY[NP_REAL] = np.array(
            [reflectance_diff_max,
            heating_diff_max,
            flux_sfc_dn_diff_max],
            dtype = NP_REAL
        )

        kk: int
        for kk in range(0, nrows):
            jj: int
            for jj in range(0, ndays):
                if kk > 0:
                    axs[kk,jj].set_yscale("symlog", linthresh = eps_array[kk])

                axs[kk,jj].set_ylim(-ylim_array[kk], ylim_array[kk])
                axs[kk,jj].axhline(-eps_array[kk],
                    color = "gray",
                    linestyle = "solid",
                    linewidth = 0.5)
                axs[kk,jj].axhline(eps_array[kk],
                    color = "gray",
                    linestyle = "solid",
                    linewidth = 0.5)

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_error_ribbon.{}.png".format(coarse_factor_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 512, bbox_inches = "tight")
        plt.close(fig)

if __name__ == "__main__":
    main()