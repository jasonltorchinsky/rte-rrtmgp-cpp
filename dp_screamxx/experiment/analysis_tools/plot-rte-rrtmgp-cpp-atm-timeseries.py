#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
from datetime import datetime, timezone
import re
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_cloud_wc, calc_z_max_info, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-atm-timeseries"
prog_desc: str = "Visualize atmsopheric state throughout the simulation."

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
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true", default = False,
        help = "Re-calculate atmospheric timeseries quantities.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
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
    [rad_tran_infiles, _] = find_inout_pairs(rad_tran_indir, None, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()
        coarse_factor: NP_INT = NP_INT(coarse_factor_str.split("_")[1])

        plt_filename: str = "rte_rrtmgp_cpp_atm_timeseries.{}.png".format(coarse_factor_str)
        nc_filename: str = os.path.splitext(plt_filename)[0] + ".nc"

        plt_filepath: str = os.path.join(rad_tran_vizdir, plt_filename)
        nc_filepath: str = os.path.join(working_dir, nc_filename)

        msg: str = "Processing {}...".format(coarse_factor_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Read or calculate quantities for this resolution
        #-----------------------------------------------------------------------
        atm_dataset: xr.Dataset
        if not recalculate and os.path.exists(nc_filepath):
            msg: str = "Reading atmospheric timeseries from {}...".format(nc_filepath)
            print_msg(msg)

            with xr.open_dataset(nc_filepath) as dataset:
                atm_dataset = dataset.load()
        else:
            if not recalculate:
                msg: str = "No existing atmospheric timeseries file found at {}; it will be calculated.".format(
                    nc_filepath)
                print_msg(msg)

            #-------------------------------------------------------------------
            # Obtain daytime indices, times, SZAs
            #-------------------------------------------------------------------
            msg: str = "Obtaining daytime information..."
            print_msg(msg)

            daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(
                rad_tran_infile) # Time indices for each day; [ndays; time_per_day]
            daytime_times: NP_ARRAY[NP_REAL] = find_times(
                rad_tran_infile,
                daytime_indices) # Time since simulation start; [h]; [ndays, time_per_day]
            daytime_szas: NP_ARRAY[NP_REAL] = find_szas(
                rad_tran_infile,
                daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, time_per_day]
            ndays: NP_INT = NP_INT(daytime_indices.shape[0])
            ntime: NP_INT = NP_INT(daytime_indices.shape[1])
            z_max_info: dict = calc_z_max_info(
                rad_tran_infile,
                z_max = z_max)

            total_mass_cloud_data: NP_ARRAY[NP_REAL] = np.full(
                [ndays, ntime], np.nan, dtype = NP_REAL)

            dx: NP_REAL = NP_REAL(np.nan)
            dy: NP_REAL = NP_REAL(np.nan)
            dz: NP_REAL = NP_REAL(np.nan)

            #-------------------------------------------------------------------
            # Calculate total cloud water mass for each day
            #-------------------------------------------------------------------
            jj: int
            for jj in range(0, ndays):
                msg: str = "Calculating total cloud water mass for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile,
                    time_indices = daytime_indices[jj],
                    z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [time, lay, y, x]

                # Obtain grid information
                x: XR_DATAARRAY = cloud_wc["x"] # x-coordinate of column-midpoints; [n_x]; [m]
                y: XR_DATAARRAY = cloud_wc["y"] # y-coordinate of column-midpoints; [n_y]; [m]
                lay: XR_DATAARRAY = cloud_wc["lay"] # z-coordinate of layer-midpoints; [n_lay]; [m]

                dx = NP_REAL(x[1] - x[0]) # x-dimension spacing; [m]; # ASSUME UNIFORM
                dy = NP_REAL(y[1] - y[0]) # y-dimension spacing; [m]; # ASSUME UNIFORM
                dz = NP_REAL(lay[1] - lay[0]) # z-dimension spacing; [m]; # ASSUME UNIFORM

                # Total cloud water mass
                total_mass_cloud: XR_DATAARRAY = (dx * dy * dz) * cloud_wc.sum(
                    dim = ["lay", "y", "x"]) * 1.e-3 # [time]; [g] => [kg]

                total_mass_cloud_data[jj,:] = np.array(
                    total_mass_cloud.values, dtype = NP_REAL)

            horizontal_resolution_label: str = format_horizontal_resolution(dx)

            #-------------------------------------------------------------------
            # Save quantities to netCDF
            #-------------------------------------------------------------------
            msg: str = "Saving atmospheric timeseries to {}...".format(nc_filepath)
            print_msg(msg)

            atm_dataset = xr.Dataset(
                data_vars = {
                    "total_cloud_water_mass": (
                        ["day", "time_index"],
                        total_mass_cloud_data,
                        {
                            "long_name": "Total cloud water mass",
                            "units": "kg",
                            "description": "Total cloud water mass integrated over the full horizontal domain and selected vertical range."
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
                    "title": "Atmospheric timeseries for RTE-RRTMGP-CPP",
                    "description": "Timeseries data used to create the total cloud water mass plot.",
                    "coarse_factor": int(coarse_factor),
                    "coarse_factor_label": coarse_factor_str,
                    "horizontal_resolution": float(dx),
                    "horizontal_resolution_units": "m",
                    "horizontal_resolution_label": horizontal_resolution_label,
                    "dx": float(dx),
                    "dy": float(dy),
                    "dz": float(dz),
                    "grid_spacing_units": "m",
                    "rad_tran_indir": rad_tran_indir,
                    "z_max": "None" if z_max is None else float(z_max),
                    "z_max_units": "km",
                    "created_by": prog_name,
                    "created": datetime.now(timezone.utc).isoformat()
                }
            )

            atm_dataset.to_netcdf(nc_filepath)

        #-----------------------------------------------------------------------
        # Set up figure for plotting
        #-----------------------------------------------------------------------
        msg: str = "Setting up figure..."
        print_msg(msg)

        ndays: NP_INT = NP_INT(atm_dataset.sizes["day"])

        nrows: NP_INT = NP_INT(1)
        ncols: NP_INT = NP_INT(ndays)
        fig_width: NP_REAL = NP_REAL(6.5)
        fig_height: NP_REAL = (NP_REAL(nrows) / NP_REAL(ncols)) * fig_width
        fig_size: list[NP_REAL] = [fig_width, fig_height]
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            sharex = "col", sharey = "row",
            constrained_layout = True,
            figsize = fig_size,
            squeeze = False)

        horizontal_resolution_label: str = str(
            atm_dataset.attrs["horizontal_resolution_label"])

        #-----------------------------------------------------------------------
        # Plot each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            msg: str = "Plotting data for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            row: NP_INT = NP_INT(0)

            total_mass_cloud: XR_DATAARRAY = atm_dataset["total_cloud_water_mass"].isel(
                day = jj)
            time: XR_DATAARRAY = atm_dataset["time"].isel(day = jj)

            axs[row,jj].plot(
                time,
                total_mass_cloud,
                color = "black")
            axs[row,jj].set_yscale("log")

            #-------------------------------------------------------------------
            # Add common column-wise plot elements
            #-------------------------------------------------------------------
            time_values: NP_ARRAY[NP_REAL] = np.array(
                atm_dataset["time"].isel(day = jj).values, dtype = NP_REAL)
            sza_values: NP_ARRAY[NP_REAL] = np.array(
                atm_dataset["solar_zenith_angle"].isel(day = jj).values, dtype = NP_REAL)

            xlim: NP_ARRAY[NP_REAL] = np.array(
                [time_values[0], time_values[-1]], dtype = NP_REAL)
            time_xticks: NP_ARRAY[NP_REAL] = np.array(
                [time_values[0], time_values[NP_INT(time_values.size/2)], time_values[-1]],
                dtype = NP_REAL)
            sza_xticks: NP_ARRAY[NP_REAL] = np.array(
                [sza_values[0], sza_values[NP_INT(time_values.size/2)], sza_values[-1]],
                dtype = NP_REAL)
            sza_xtick_labels: list[str] = [
                r"{:.1f}$^{{\circ}}$".format(solar_zenith_angle)
                for solar_zenith_angle in sza_xticks
            ]

            axs[row,jj].set_xlim(xlim)
            axs[row,jj].set_xticks(time_xticks)
            axs[row,jj].axvline(time_xticks[1],
                color = "gray",
                linestyle = "solid",
                linewidth = 0.5)

            ax_2 = axs[row,jj].secondary_xaxis("top")
            ax_2.set_xticks(time_xticks, labels = sza_xtick_labels)

        #-----------------------------------------------------------------------
        # Labels
        #-----------------------------------------------------------------------
        fig.suptitle("Cloud Water Mass [kg] - {}".format(
            horizontal_resolution_label),
            y = 1.08)
        fig.supxlabel(r"Time $\left[ h \right]$")

        for jj in range(0, ndays):
            col_title: str = "Day {}".format(jj)
            axs[0,jj].set_title(col_title,
                pad = 24.0)

        #-----------------------------------------------------------------------
        # Save the plot to file
        #-----------------------------------------------------------------------
        msg: str = "Saving plot to {}...".format(plt_filepath)
        print_msg(msg)

        fig.savefig(plt_filepath, dpi = 200, bbox_inches = "tight")
        plt.close(fig)

def format_horizontal_resolution(dx: NP_REAL):
    dx_str: str
    if dx <= 1.e3:
        dx_str = r"{:.0f} $m$".format(dx)
    else:
        dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

    return dx_str

if __name__ == "__main__":
    main()