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
from datetime import datetime, timezone
from typing import Optional

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_FIGURE, MPL_AXES
from consts.visual import plot_colors
from rte_rrtmgp_cpp import find_inout_pairs as rte_rrtmgp_cpp_find_inout_pairs, \
    find_daytime_indices as rte_rrtmgp_cpp_find_daytime_indices, \
    find_szas as rte_rrtmgp_cpp_find_szas, \
    find_times as rte_rrtmgp_cpp_find_times, \
    calc_sw_reflectance as rte_rrtmgp_cpp_calc_sw_reflectance, \
    calc_sw_heating as rte_rrtmgp_cpp_calc_sw_heating, \
    calc_sw_flux_sfc_dn as rte_rrtmgp_cpp_calc_sw_flux_sfc_dn, \
    calc_z_max_info as rte_rrtmgp_cpp_calc_z_max_info, \
    print_msg as rte_rrtmgp_cpp_print_msg
from ml3drt import calc_sw_reflectance as ml3drt_calc_sw_reflectance, \
    calc_sw_heating as ml3drt_calc_sw_heating, \
    calc_sw_flux_sfc_dn as ml3drt_calc_sw_flux_sfc_dn

# Script variables
prog_name: str = "plot-ml3drt-error-timeseries"
prog_desc: str = "Visualize error for ML3DRT and two-stream radiative transfer."

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
    parser.add_argument("--recalculate", action = "store_true", default = False,
        help = "Re-calculate all necessary quantities for plotting.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--error-types", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Error types to calculate, e.g., mae,mbe,rmse,corr.")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    ml3drt_outfile: str = os.path.normpath(args.ml3drt_outfile)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

    lr_re: re.Pattern = re.compile("lr_..")
    lr_match: Optional[re.Match] = lr_re.search(os.path.basename(ml3drt_outfile))
    if lr_match is None:
        raise ValueError("Could not determine horizontal coarsening factor from ML3DRT output file name: {}".format(
            ml3drt_outfile))

    coarse_factor_str: str = lr_match.group()
    coarse_factor: NP_INT = NP_INT(coarse_factor_str.split("_")[1])
    coarse_factors: NP_ARRAY[NP_INT] = np.array([coarse_factor], dtype = NP_INT)

    error_types: list[str]
    if args.error_types is not None:
        error_types = args.error_types.split(",")
    else:
        error_types = ["mae", "mbe", "rmse", "corr"]

    error_type: str
    for error_type in error_types:
        assert(error_type in ["mae", "mbe", "rmse", "corr"])

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
    if nfiles != 1:
        raise ValueError("Expected exactly one RTE-RRTMGP-CPP file pair for {}, found {}.".format(
            coarse_factor_str, nfiles))

    rad_tran_infile: str = rad_tran_infiles[0]
    rad_tran_outfile: str = rad_tran_outfiles[0]

    error_filepaths: dict = {}
    error_datasets: dict = {}
    calculate_errors: bool = recalculate

    error_type: str
    for error_type in error_types:
        plt_filename: str = "ml3drt_{}_timeseries.png".format(error_type)
        nc_filename: str = os.path.splitext(plt_filename)[0] + ".nc"
        nc_filepath: str = os.path.join(working_dir, nc_filename)

        error_filepaths[error_type] = nc_filepath

        if not recalculate and os.path.exists(nc_filepath):
            msg: str = "Reading {} errors from {}...".format(error_type, nc_filepath)
            rte_rrtmgp_cpp_print_msg(msg)

            with xr.open_dataset(nc_filepath) as error_dataset:
                error_datasets[error_type] = error_dataset.load()
        else:
            if not recalculate:
                msg: str = "No existing {} error file found at {}; it will be calculated.".format(
                    error_type, nc_filepath)
                rte_rrtmgp_cpp_print_msg(msg)

            calculate_errors = True

    #---------------------------------------------------------------------------
    # Calculate relevant quantities for each day
    #---------------------------------------------------------------------------
    if calculate_errors:
        reflectance_rt: dict = {}
        reflectance_ml3drt: dict = {}
        reflectance_ts: dict = {}
        flux_sfc_dn_rt: dict = {}
        flux_sfc_dn_ml3drt: dict = {}
        flux_sfc_dn_ts: dict = {}
        heating_rt: dict = {}
        heating_ml3drt: dict = {}
        heating_ts: dict = {}

        msg: str = "Processing {}...".format(coarse_factor_str)
        rte_rrtmgp_cpp_print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain daytime indices, times, SZAs
        #-----------------------------------------------------------------------
        msg: str = "Obtaining daytime information..."
        rte_rrtmgp_cpp_print_msg(msg)

        daytime_indices: NP_ARRAY[NP_INT] = rte_rrtmgp_cpp_find_daytime_indices(
            rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
        daytime_times: NP_ARRAY[NP_REAL] = rte_rrtmgp_cpp_find_times(
            rad_tran_infile, 
            daytime_indices) # Time since simulation start; [h]; [ndays, 3]
        daytime_szas: NP_ARRAY[NP_REAL] = rte_rrtmgp_cpp_find_szas(
            rad_tran_infile, 
            daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(daytime_indices.shape[0])
        z_max_info: dict = rte_rrtmgp_cpp_calc_z_max_info(
            rad_tran_infile, 
            z_max = z_max)

        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate reflectance
            #-------------------------------------------------------------------
            msg: str = "Calculating reflectance for day {} of {}...".format(jj, ndays - 1)
            rte_rrtmgp_cpp_print_msg(msg)

            reflectance_rt[jj] = rte_rrtmgp_cpp_calc_sw_reflectance(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "rt") # Shortwave reflectance, ray-tracer; [N/A]; [time, y, x]

            reflectance_ml3drt[jj] = ml3drt_calc_sw_reflectance(
                rad_tran_infile,
                ml3drt_outfile,
                time_indices = daytime_indices[jj,...]) # Shortwave reflectance, ML3DRT; [N/A]; [time, y, x]

            reflectance_ts[jj] = rte_rrtmgp_cpp_calc_sw_reflectance(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "ts") # Shortwave reflectance, two-stream; [N/A]; [time, y, x]

            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
            msg: str = "Calculating heating rates for day {} of {}...".format(jj, ndays - 1)
            rte_rrtmgp_cpp_print_msg(msg)

            heating_rt[jj] = rte_rrtmgp_cpp_calc_sw_heating(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max_info = z_max_info,
                solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [time, lay, y, x]

            heating_ml3drt[jj] = ml3drt_calc_sw_heating(
                rad_tran_infile,
                ml3drt_outfile,
                time_indices = daytime_indices[jj,...],
                z_max_info = z_max_info) # Shortwave heating rate, ML3DRT; [K d^{-1}]; [time, lay, y, x]

            heating_ts[jj] = rte_rrtmgp_cpp_calc_sw_heating(
                rad_tran_infile,
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                z_max_info = z_max_info,
                solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [time, lay, y, x]
            
            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
            msg: str = "Calculating downwelling surface fluxes for day {} of {}...".format(jj, ndays - 1)
            rte_rrtmgp_cpp_print_msg(msg)

            flux_sfc_dn_rt[jj] = rte_rrtmgp_cpp_calc_sw_flux_sfc_dn(
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "rt") # Shortwave downwelling surface flux, ray-tracer; [W m^{-2}]; [time, y, x]

            flux_sfc_dn_ml3drt[jj] = ml3drt_calc_sw_flux_sfc_dn(
                ml3drt_outfile,
                time_indices = daytime_indices[jj,...]) # Shortwave downwelling surface flux, ML3DRT; [W m^{-2}]; [time, y, x]

            flux_sfc_dn_ts[jj] = rte_rrtmgp_cpp_calc_sw_flux_sfc_dn(
                rad_tran_outfile,
                time_indices = daytime_indices[jj,...],
                solver = "ts") # Shortwave downwelling surface flux, two-stream; [W m^{-2}]; [time, y, x]

    #---------------------------------------------------------------------------
    # Loop through error types, save missing NetCDF files, and create plots
    #---------------------------------------------------------------------------
    error_type: str
    for error_type in error_types:
        error_dataset: Optional[xr.Dataset] = error_datasets.get(error_type)

        if error_dataset is None:
            solver_names: list[str] = ["ml3drt", "two_stream"]
            solver_labels: list[str] = ["Emulator", "Two-Stream"]

            ntime: NP_INT = NP_INT(daytime_times.shape[1])
            nsolvers: NP_INT = NP_INT(len(solver_names))

            reflectance_error_data: NP_ARRAY[NP_REAL] = np.full(
                [ndays, ntime, nsolvers], np.nan, dtype = NP_REAL)
            heating_error_data: NP_ARRAY[NP_REAL] = np.full(
                [ndays, ntime, nsolvers], np.nan, dtype = NP_REAL)
            flux_sfc_dn_error_data: NP_ARRAY[NP_REAL] = np.full(
                [ndays, ntime, nsolvers], np.nan, dtype = NP_REAL)

            #-------------------------------------------------------------------
            # Calculate errors before creating plots
            #-------------------------------------------------------------------
            jj: int
            for jj in range(0, ndays):
                msg: str = "Calculating {} for day {} of {}...".format(error_type, jj, ndays - 1)
                rte_rrtmgp_cpp_print_msg(msg)

                reflectance_error_ml3drt: xr.DataArray = calc_error(
                    reflectance_rt[jj],
                    reflectance_ml3drt[jj],
                    error_type)
                heating_error_ml3drt: xr.DataArray = calc_error(
                    heating_rt[jj],
                    heating_ml3drt[jj],
                    error_type)
                flux_sfc_dn_error_ml3drt: xr.DataArray = calc_error(
                    flux_sfc_dn_rt[jj],
                    flux_sfc_dn_ml3drt[jj],
                    error_type)

                reflectance_error_ts: xr.DataArray = calc_error(
                    reflectance_rt[jj],
                    reflectance_ts[jj],
                    error_type)
                heating_error_ts: xr.DataArray = calc_error(
                    heating_rt[jj],
                    heating_ts[jj],
                    error_type)
                flux_sfc_dn_error_ts: xr.DataArray = calc_error(
                    flux_sfc_dn_rt[jj],
                    flux_sfc_dn_ts[jj],
                    error_type)

                reflectance_error_data[jj,:,0] = np.array(
                    reflectance_error_ml3drt.values, dtype = NP_REAL)
                heating_error_data[jj,:,0] = np.array(
                    heating_error_ml3drt.values, dtype = NP_REAL)
                flux_sfc_dn_error_data[jj,:,0] = np.array(
                    flux_sfc_dn_error_ml3drt.values, dtype = NP_REAL)

                reflectance_error_data[jj,:,1] = np.array(
                    reflectance_error_ts.values, dtype = NP_REAL)
                heating_error_data[jj,:,1] = np.array(
                    heating_error_ts.values, dtype = NP_REAL)
                flux_sfc_dn_error_data[jj,:,1] = np.array(
                    flux_sfc_dn_error_ts.values, dtype = NP_REAL)

            #-------------------------------------------------------------------
            # Save errors to netCDF before attempting to create plots
            #-------------------------------------------------------------------
            msg: str = "Saving {} errors to {}...".format(
                error_type, error_filepaths[error_type])
            rte_rrtmgp_cpp_print_msg(msg)

            if error_type == "mae":
                error_str = "Mean Absolute Error"
                error_formula = "mean(abs(test_solver - ray_tracer), space)"
            elif error_type == "mbe":
                error_str = "Mean Bias Error"
                error_formula = "mean(test_solver, space) - mean(ray_tracer, space)"
            elif error_type == "rmse":
                error_str = "Root-Mean-Square Error"
                error_formula = "sqrt(mean((test_solver - ray_tracer)^2, space))"
            elif error_type == "corr":
                error_str = "Correlation"
                error_formula = "corr(test_solver, ray_tracer, space)"

            if error_type == "corr":
                reflectance_units: str = "1"
                heating_units: str = "1"
                flux_sfc_dn_units: str = "1"

                reflectance_long_name: str = "Correlation of shortwave reflectance"
                heating_long_name: str = "Correlation of shortwave heating rate"
                flux_sfc_dn_long_name: str = "Correlation of downwelling shortwave surface flux"

                reflectance_description: str = "Shortwave reflectance spatial correlation between each test solver and the ray-tracer solver."
                heating_description: str = "Shortwave heating-rate spatial correlation between each test solver and the ray-tracer solver."
                flux_sfc_dn_description: str = "Downwelling shortwave surface-flux spatial correlation between each test solver and the ray-tracer solver."
            else:
                reflectance_units: str = "1"
                heating_units: str = "K d-1"
                flux_sfc_dn_units: str = "W m-2"

                reflectance_long_name: str = "{} in shortwave reflectance".format(error_str)
                heating_long_name: str = "{} in shortwave heating rate".format(error_str)
                flux_sfc_dn_long_name: str = "{} in downwelling shortwave surface flux".format(error_str)

                reflectance_description: str = "Shortwave reflectance error between each test solver and the ray-tracer solver."
                heating_description: str = "Shortwave heating-rate error between each test solver and the ray-tracer solver."
                flux_sfc_dn_description: str = "Downwelling shortwave surface-flux error between each test solver and the ray-tracer solver."

            error_dataset = xr.Dataset(
                data_vars = {
                    "reflectance_error": (
                        ["day", "time_index", "solver"],
                        reflectance_error_data,
                        {
                            "long_name": reflectance_long_name,
                            "units": reflectance_units,
                            "description": reflectance_description
                        }
                    ),
                    "heating_error": (
                        ["day", "time_index", "solver"],
                        heating_error_data,
                        {
                            "long_name": heating_long_name,
                            "units": heating_units,
                            "description": heating_description
                        }
                    ),
                    "flux_sfc_dn_error": (
                        ["day", "time_index", "solver"],
                        flux_sfc_dn_error_data,
                        {
                            "long_name": flux_sfc_dn_long_name,
                            "units": flux_sfc_dn_units,
                            "description": flux_sfc_dn_description
                        }
                    )
                },
                coords = {
                    "day": (
                        ["day"],
                        np.arange(0, ndays, dtype = NP_INT),
                        {
                            "long_name": "Day index"
                        }
                    ),
                    "time_index": (
                        ["time_index"],
                        np.arange(0, ntime, dtype = NP_INT),
                        {
                            "long_name": "Time index within day"
                        }
                    ),
                    "solver": (
                        ["solver"],
                        np.array(solver_names, dtype = str),
                        {
                            "long_name": "Test solver"
                        }
                    ),
                    "solver_label": (
                        ["solver"],
                        np.array(solver_labels, dtype = str),
                        {
                            "long_name": "Test-solver label used in plots"
                        }
                    ),
                    "time": (
                        ["day", "time_index"],
                        daytime_times,
                        {
                            "long_name": "Time since simulation start",
                            "units": "h"
                        }
                    ),
                    "solar_zenith_angle": (
                        ["day", "time_index"],
                        daytime_szas,
                        {
                            "long_name": "Solar zenith angle",
                            "units": "degrees"
                        }
                    )
                },
                attrs = {
                    "title": "{} timeseries for ML3DRT and two-stream radiative transfer".format(error_str),
                    "description": "Timeseries data used to create the {} error plot.".format(error_type),
                    "error_type": error_type,
                    "error_description": error_str,
                    "error_formula": error_formula,
                    "reference_solver": "ray_tracer",
                    "test_solvers": ",".join(solver_names),
                    "rad_tran_indir": rad_tran_indir,
                    "rad_tran_outdir": rad_tran_outdir,
                    "rad_tran_infile": rad_tran_infile,
                    "rad_tran_outfile": rad_tran_outfile,
                    "ml3drt_outfile": ml3drt_outfile,
                    "coarse_factor": int(coarse_factor),
                    "coarse_factor_label": coarse_factor_str,
                    "z_max": "None" if z_max is None else float(z_max),
                    "z_max_units": "km",
                    "created_by": prog_name,
                    "created": datetime.now(timezone.utc).isoformat()
                }
            )

            error_dataset.to_netcdf(error_filepaths[error_type])
            error_datasets[error_type] = error_dataset

        ndays: NP_INT = NP_INT(error_dataset.sizes["day"])
        plot_solver_names: list[str] = [
            str(solver_name)
            for solver_name in error_dataset["solver"].values
        ]
        plot_solver_labels: list[str] = [
            str(solver_label)
            for solver_label in error_dataset["solver_label"].values
        ]

        #-----------------------------------------------------------------------
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        rte_rrtmgp_cpp_print_msg(msg)

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

        #-----------------------------------------------------------------------
        # Loop through days
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            time: xr.DataArray = error_dataset["time"].isel(day = jj) # [time]; [h]

            #-------------------------------------------------------------------
            # Plot errors for each test solver
            #-------------------------------------------------------------------
            msg: str = "Plotting {} for day {} of {}...".format(error_type, jj, ndays - 1)
            rte_rrtmgp_cpp_print_msg(msg)

            nplot_colors: NP_INT = NP_INT(len(plot_colors))

            # Row 0 - Reflectance
            row: NP_INT = NP_INT(0)
            ii: int
            for ii in range(0, len(plot_solver_names)):
                reflectance_plot: xr.DataArray = error_dataset["reflectance_error"].isel(
                    day = jj, solver = ii)

                axs[row,jj].plot(
                    time, 
                    reflectance_plot,
                    color = plot_colors[ii % nplot_colors], 
                    label = plot_solver_labels[ii])

            # Row 1 - Heating Rate
            row: NP_INT = NP_INT(1)
            ii: int
            for ii in range(0, len(plot_solver_names)):
                heating_plot: xr.DataArray = error_dataset["heating_error"].isel(
                    day = jj, solver = ii)

                axs[row,jj].plot(
                    time, 
                    heating_plot,
                    color = plot_colors[ii % nplot_colors], 
                    label = plot_solver_labels[ii])
                
            # Row 2 - Downwelling Surface Flux
            row: NP_INT = NP_INT(2)
            ii: int
            for ii in range(0, len(plot_solver_names)):
                flux_sfc_dn_plot: xr.DataArray = error_dataset["flux_sfc_dn_error"].isel(
                    day = jj, solver = ii)

                axs[row,jj].plot(
                    time, 
                    flux_sfc_dn_plot,
                    color = plot_colors[ii % nplot_colors], 
                    label = plot_solver_labels[ii])

            #-------------------------------------------------------------------
            # Add common column-wise plot elements
            #-------------------------------------------------------------------
            # x-ticks
            time: NP_ARRAY[NP_REAL] = np.array(
                error_dataset["time"].isel(day = jj).values, dtype = NP_REAL)
            sza: NP_ARRAY[NP_REAL] = np.array(
                error_dataset["solar_zenith_angle"].isel(day = jj).values, dtype = NP_REAL)

            xlim: list[NP_REAL] = np.array([time[0], time[-1]], dtype = NP_REAL)
            time_xticks: NP_ARRAY[NP_REAL] = np.array([time[0], time[NP_INT(time.size/2)], time[-1]], dtype = NP_REAL)
            sza_xticks: NP_ARRAY[NP_REAL] = np.array([sza[0], sza[NP_INT(time.size/2)], sza[-1]], dtype = NP_REAL)
            sza_xtick_labels: list[str] = [r"{:.1f}$^{{\circ}}$".format(solar_zenith_angle) for solar_zenith_angle in sza_xticks]
            ll: int
            for ll in range(0, nrows):
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

        #-----------------------------------------------------------------------
        # Add plot elements
        #-----------------------------------------------------------------------
        # Labels
        error_str: str
        if error_type == "mae":
            error_str = "Mean Absolute Error"
        elif error_type == "mbe":
            error_str = "Mean Bias Error"
        elif error_type == "rmse":
            error_str = "Root-Mean-Square Error"
        elif error_type == "corr":
            error_str = "Correlation"
        title_str: str = error_str

        fig.suptitle(title_str,
            y = 1.06)
        fig.supxlabel(r"Time $\left[ h \right]$")

        for ii in range(0, ndays):
            col_title: str = "Day {}".format(ii)
            axs[0,ii].set_title(col_title,
                pad = 24.0)

        if error_type == "corr":
            axs[0,0].set_ylabel(r"Reflectance")
            axs[1,0].set_ylabel(r"Heating Rate")
            axs[2,0].set_ylabel(r"Downwelling Surface Flux")
        else:
            axs[0,0].set_ylabel(r"Reflectance")
            axs[1,0].set_ylabel(r"Heating Rate $\left[ K\,d^{-1} \right]$")
            axs[2,0].set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")

        handles, labels = axs[0,0].get_legend_handles_labels()
        nlegend_cols: NP_INT = NP_INT(min(len(labels), 2))
        handles, labels = reorder_legend_entries(
            handles, 
            labels, 
            nlegend_cols)

        fig.legend(
            handles,
            labels,
            loc = "upper center",
            bbox_to_anchor = (0.5, 1.045),
            ncol = nlegend_cols,
            handlelength = 2.0,
            columnspacing = 1.2,
            handletextpad = 0.4
        )

        # Axes Scaling, Limits
        if error_type in ["mae", "rmse"]:
            for ax in axs.flatten():
                ax.set_yscale("linear")
        elif error_type in ["mbe"]:
            for ax in axs.flatten():
                ylim: tuple[NP_REAL] = ax.get_ylim()
                ymax: NP_REAL = np.abs(ylim).max()

                ax.set_ylim([-ymax, ymax])
                ax.set_yscale("linear")
                ax.axhline(
                    0, 
                    color = "gray", 
                    linewidth = 0.5, 
                    linestyle = "solid")
        elif error_type in ["corr"]:
            for ax in axs.flatten():
                ax.set_yscale("linear")

                has_negative: bool = False
                for line in ax.get_lines():
                    ydata: NP_ARRAY[NP_REAL] = np.array(
                        line.get_ydata(), dtype = NP_REAL)

                    if np.any(np.isfinite(ydata) & (ydata < 0)):
                        has_negative = True
                        break

                if has_negative:
                    ax.axhline(
                        0, 
                        color = "gray", 
                        linewidth = 0.5, 
                        linestyle = "solid")

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        plt_filename = "ml3drt_{}_timeseries.png".format(error_type)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 512, bbox_inches = "tight")
        plt.close(fig)

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

def calc_error(rt_field: xr.DataArray, test_field: xr.DataArray, error_type: str):
    assert(error_type in ["mae", "mbe", "rmse", "corr"])

    space_dims: list[str] = [dim for dim in rt_field.dims if dim != "time"]

    error: xr.DataArray
    if error_type == "mae":
        error = (np.abs(test_field - rt_field)).mean(dim = space_dims)
    elif error_type == "mbe":
        error = test_field.mean(dim = space_dims) - rt_field.mean(dim = space_dims)
    elif error_type == "rmse":
        error = np.sqrt((np.pow(test_field - rt_field, 2).mean(dim = space_dims)))
    elif error_type == "corr":
        error = xr.corr(test_field, rt_field, dim = space_dims)

    return error

if __name__ == "__main__":
    main()