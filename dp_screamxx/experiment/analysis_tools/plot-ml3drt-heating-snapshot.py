#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
from datetime import datetime
import glob
import re
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, MPL_FIGURE, MPL_AXES, MPL_PCOLORMESH
from consts.numeric import NP_SMALL
from consts.visual import diff_cmap, flux_cmap, heating_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating as rte_rrtmgp_cpp_calc_sw_heating, \
    calc_z_max_info, find_grid, find_y_islice, print_msg
from ml3drt import calc_sw_heating as ml3drt_calc_sw_heating

# Script variables
prog_name: str = "plot-ml3drt-heating-snapshot"
prog_desc: str = "Visualize atmospheric heating rates for ML3DRT."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time: str = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line input...".format(current_time)
    print(msg, flush = True)

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
        default = False,
        help = "Re-calculate all necessary quantities for plotting.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Ignored. Coarsening factor is extracted from --ml3drt-outfile.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    ml3drt_outfile: str = os.path.normpath(args.ml3drt_outfile)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None

    #---------------------------------------------------------------------------
    # Extract coarse factor from ML3DRT output file name
    #---------------------------------------------------------------------------
    lr_re: re.Pattern = re.compile("lr_([0-9][0-9])")
    lr_match: Optional[re.Match] = lr_re.search(os.path.basename(ml3drt_outfile))
    if lr_match is None:
        msg: str = "Could not extract coarse factor from ML3DRT output file name: {}".format(
            ml3drt_outfile)
        raise ValueError(msg)

    ml3drt_lr_str: str = lr_match.group()
    coarse_factors: NP_ARRAY[NP_INT] = np.array(
        [NP_INT(lr_match.group(1))], dtype = NP_INT)

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Find file pairs at requested resolution
    #---------------------------------------------------------------------------
    rad_tran_infiles: list[str]
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(rad_tran_indir,
        rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        if lr_str != ml3drt_lr_str:
            continue

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        cached_working_filepaths: list[str] = sorted(
            glob.glob(os.path.join(
                working_dir,
                "ml3drt_heating_snapshot.{}.day_*.nc".format(lr_str))),
            key = lambda fp: int(re.search(
                r"\.day_([0-9]+)\.nc$",
                os.path.basename(fp)).group(1))
        )

        use_cached_files: bool = False
        if not recalculate and len(cached_working_filepaths) > 0:
            with xr.open_dataset(cached_working_filepaths[0]) as cache_check_ds:
                ndays_expected: NP_INT = NP_INT(cache_check_ds.attrs.get(
                    "ndays", len(cached_working_filepaths)))
                cached_ml3drt_outfile: str = str(cache_check_ds.attrs.get(
                    "ml3drt_outfile", ml3drt_outfile))
                cached_rad_tran_infile: str = str(cache_check_ds.attrs.get(
                    "rad_tran_infile", rad_tran_infile))
                cached_rad_tran_outfile: str = str(cache_check_ds.attrs.get(
                    "rad_tran_outfile", rad_tran_outfile))

                required_cache_vars: list[str] = [
                    "cloud_wc",
                    "heating_rt",
                    "heating_ml3drt",
                    "heating_ts",
                    "x_indices",
                    "y_view_slice_start",
                    "y_view_slice_stop",
                    "y_data_slice_start",
                    "y_data_slice_stop",
                    "y_view_min",
                    "y_view_max",
                    "source_time_index",
                    "time",
                    "sza",
                    "dx"
                ]
                required_cache_attrs: list[str] = [
                    "ndays",
                    "y_view_slice_width",
                    "y_view_slice_width_units",
                    "y_extra_fraction"
                ]

                cache_has_required_fields: bool = (
                    all([var_name in cache_check_ds for var_name in required_cache_vars])
                    and all([attr_name in cache_check_ds.attrs for attr_name in required_cache_attrs])
                )

            cached_day_indices: NP_ARRAY[NP_INT] = np.array([
                NP_INT(re.search(
                    r"\.day_([0-9]+)\.nc$",
                    os.path.basename(working_filepath)).group(1))
                for working_filepath in cached_working_filepaths], dtype = NP_INT)

            use_cached_files = (
                cache_has_required_fields
                and len(cached_working_filepaths) == ndays_expected
                and np.array_equal(
                    cached_day_indices,
                    np.arange(0, ndays_expected, dtype = NP_INT))
                and os.path.normpath(cached_ml3drt_outfile) == ml3drt_outfile
                and os.path.normpath(cached_rad_tran_infile) == rad_tran_infile
                and os.path.normpath(cached_rad_tran_outfile) == rad_tran_outfile
            )

        if use_cached_files:
            msg: str = "Using cached plotting data from {}...".format(working_dir)
            print_msg(msg)
            working_filepaths: list[str] = cached_working_filepaths
        else:
            if recalculate:
                msg: str = "Recalculating plotting data..."
            else:
                msg: str = "Cached plotting data incomplete or unavailable. Calculating plotting data..."
            print_msg(msg)

            #-------------------------------------------------------------------
            # Obtain grid information
            #-------------------------------------------------------------------
            msg: str = "Obtaining grid information..."
            print_msg(msg)
            grid: dict = find_grid(rad_tran_infile)

            #-------------------------------------------------------------------
            # Obtain Morning-Noon-Night time indices, times, SZAs, z_max index
            #-------------------------------------------------------------------
            msg: str = "Obtaining morning-noon-night information..."
            print_msg(msg)

            mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(
                rad_tran_infile
                ) # [ndays, 3]
            mnn_times: NP_ARRAY[NP_REAL] = find_times(
                rad_tran_infile, 
                mnn_indices
                ) # Time since simulation start; [h]; [ndays, 3]
            mnn_szas: NP_ARRAY[NP_REAL] = find_szas(
                rad_tran_infile, 
                mnn_indices
                ) # Solar zenith angle (SZA); [degrees]; [ndays, 3]

            ndays: NP_INT = NP_INT(mnn_indices.shape[0])
            nszas: NP_INT = NP_INT(2)

            z_max_info: dict = calc_z_max_info(
                rad_tran_infile, 
                z_max = z_max
                )

            working_filepaths: list[str] = []

            #-------------------------------------------------------------------
            # Calculate fields for each requested SZA of each day
            #-------------------------------------------------------------------
            jj: int
            for jj in range(0, ndays):
                day_str: str = "day_{}".format(jj)
                working_filename: str = "ml3drt_heating_snapshot.{}.{}.nc".format(
                    lr_str, day_str)
                working_filepath: str = os.path.join(working_dir, working_filename)

                msg: str = "Calculating plotting data for day {} of {}...".format(
                    jj, ndays - 1)
                print_msg(msg)

                #---------------------------------------------------------------
                # Calculate spatial extent of plots based on maximal cloud water content
                #---------------------------------------------------------------
                msg: str = "Calculating plot spatial extent for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile,
                    time_indices = mnn_indices[jj, 0:nszas],
                    z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
            
                x_indices: NP_ARRAY[NP_INT] = np.array([
                    np.unravel_index(
                        np.argmax(cloud_wc.isel(time = ll).to_numpy()), 
                        cloud_wc.shape[1:])[2] 
                    for ll in range(0, nszas)])

                # Match the rad-tran-snapshot visual style:
                # the y-view width is three times the plotted z-extent.
                # An expanded y-range is saved so that changing the view later
                # does not necessarily require recalculation.
                y_view_slice_width: NP_REAL = 3. * z_max_info["zh_max"] # [m]
                y_extra_fraction: NP_REAL = NP_REAL(0.25)
                y_data_slice_width: NP_REAL = (1. + y_extra_fraction) * y_view_slice_width # [m]

                y_view_islices: list[slice] = [[] for _ in range(0, nszas)]
                y_data_islices: list[slice] = [[] for _ in range(0, nszas)]
                for ll in range(0, nszas):
                    y_view_islices[ll] = find_y_islice(
                        grid["y"],
                        cloud_wc.isel(time = ll, x = x_indices[ll]),
                        slice_width = y_view_slice_width
                    )

                    y_data_islices[ll] = find_y_islice(
                        grid["y"],
                        cloud_wc.isel(time = ll, x = x_indices[ll]),
                        slice_width = y_data_slice_width
                    )

                y_data_slice_start: NP_INT = NP_INT(min([
                    y_data_islices[ll].start for ll in range(0, nszas)]))
                y_data_slice_stop: NP_INT = NP_INT(max([
                    y_data_islices[ll].stop for ll in range(0, nszas)]))
                y_data_islice: slice = slice(y_data_slice_start, y_data_slice_stop)

                y_view_slice_start: NP_ARRAY[NP_INT] = np.array([
                    y_view_islices[ll].start for ll in range(0, nszas)], dtype = NP_INT)
                y_view_slice_stop: NP_ARRAY[NP_INT] = np.array([
                    y_view_islices[ll].stop for ll in range(0, nszas)], dtype = NP_INT)

                y_view_min: NP_ARRAY[NP_REAL] = np.array([
                    NP_REAL((grid["yh"].isel(yh = y_view_islices[ll].start) * 1.e-3).to_numpy())
                    for ll in range(0, nszas)], dtype = NP_REAL)
                y_view_max: NP_ARRAY[NP_REAL] = np.array([
                    NP_REAL((grid["yh"].isel(yh = y_view_islices[ll].stop) * 1.e-3).to_numpy())
                    for ll in range(0, nszas)], dtype = NP_REAL)

                #---------------------------------------------------------------
                # Calculate desired atmospheric and radiative quantities
                #---------------------------------------------------------------
                msg: str = "Calculating desired atmospheric quantities for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile, 
                    time_indices = mnn_indices[jj, 0:nszas], 
                    x_indices = x_indices, 
                    z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [slice, lay, y]

                heating_rt: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = mnn_indices[jj, 0:nszas], 
                    x_indices = x_indices, 
                    z_max_info = z_max_info,
                    solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [slice, lay, y]

                heating_ts: XR_DATAARRAY = rte_rrtmgp_cpp_calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = mnn_indices[jj, 0:nszas], 
                    x_indices = x_indices, 
                    z_max_info = z_max_info,
                    solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [slice, lay, y]

                heating_ml3drt: XR_DATAARRAY = ml3drt_calc_sw_heating(
                    rad_tran_infile,
                    ml3drt_outfile,
                    time_indices = mnn_indices[jj, 0:nszas],
                    x_indices = x_indices,
                    z_max_info = z_max_info) # Shortwave heating rate, ML3DRT; [K d^{-1}]; [slice, lay, y]

                if heating_ml3drt.dims != heating_rt.dims:
                    heating_ml3drt = xr.DataArray(
                        heating_ml3drt.to_numpy(),
                        dims = heating_rt.dims,
                        coords = heating_rt.coords
                    )

                cloud_wc = cloud_wc.isel(y = y_data_islice).load()
                heating_rt = heating_rt.isel(y = y_data_islice).load()
                heating_ts = heating_ts.isel(y = y_data_islice).load()
                heating_ml3drt = heating_ml3drt.isel(y = y_data_islice).load()

                dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]

                y: NP_ARRAY[NP_REAL] = (grid["y"]
                    .isel(y = y_data_islice)
                    .load() * 1.e-3).to_numpy() # [m] => [km]
                yh: NP_ARRAY[NP_REAL] = (grid["yh"]
                    .isel(yh = slice(y_data_slice_start, y_data_slice_stop + 1))
                    .load() * 1.e-3).to_numpy() # [m] => [km]
                z: NP_ARRAY[NP_REAL] = (grid["z"]
                    .isel(z = z_max_info["isel_indexers"]["z"])
                    .load() * 1.e-3).to_numpy() # [m] => [km]
                zh: NP_ARRAY[NP_REAL] = (grid["zh"]
                    .isel(zh = z_max_info["isel_indexers"]["zh"])
                    .load() * 1.e-3).to_numpy() # [m] => [km]

                plot_ds: xr.Dataset = xr.Dataset(
                    data_vars = {
                        "cloud_wc": (
                            cloud_wc.dims,
                            cloud_wc.to_numpy()),
                        "heating_rt": (
                            heating_rt.dims,
                            heating_rt.to_numpy()),
                        "heating_ml3drt": (
                            heating_ml3drt.dims,
                            heating_ml3drt.to_numpy()),
                        "heating_ts": (
                            heating_ts.dims,
                            heating_ts.to_numpy()),
                        "x_indices": (
                            ["slice"],
                            x_indices),
                        "y_view_slice_start": (
                            ["slice"],
                            y_view_slice_start),
                        "y_view_slice_stop": (
                            ["slice"],
                            y_view_slice_stop),
                        "y_data_slice_start": (
                            [],
                            y_data_slice_start),
                        "y_data_slice_stop": (
                            [],
                            y_data_slice_stop),
                        "y_view_min": (
                            ["slice"],
                            y_view_min),
                        "y_view_max": (
                            ["slice"],
                            y_view_max),
                        "source_time_index": (
                            ["slice"],
                            mnn_indices[jj, 0:nszas]),
                        "time": (
                            ["slice"],
                            mnn_times[jj, 0:nszas]),
                        "sza": (
                            ["slice"],
                            mnn_szas[jj, 0:nszas]),
                        "dx": (
                            [],
                            dx)
                    },
                    coords = {
                        "slice": np.arange(0, nszas, dtype = NP_INT),
                        "y": (
                            ["y"],
                            y),
                        "yh": (
                            ["yh"],
                            yh),
                        "z": (
                            ["z"],
                            z),
                        "zh": (
                            ["zh"],
                            zh)
                    },
                    attrs = {
                        "rad_tran_infile": rad_tran_infile,
                        "rad_tran_outfile": rad_tran_outfile,
                        "ml3drt_outfile": ml3drt_outfile,
                        "lr_str": lr_str,
                        "day_str": day_str,
                        "ndays": ndays,
                        "y_view_slice_width": y_view_slice_width,
                        "y_data_slice_width": y_data_slice_width,
                        "y_extra_fraction": y_extra_fraction,
                        "y_view_slice_width_units": "m",
                        "y_data_slice_width_units": "m",
                        "y_units": "km",
                        "yh_units": "km",
                        "z_units": "km",
                        "zh_units": "km",
                        "dx_units": "m",
                        "cloud_wc_units": "g m^{-3}",
                        "heating_units": "K d^{-1}"
                    }
                )

                plot_ds["cloud_wc"].attrs["long_name"] = "Cloud water content"
                plot_ds["heating_rt"].attrs["long_name"] = "Shortwave heating rate, ray-tracer"
                plot_ds["heating_ml3drt"].attrs["long_name"] = "Shortwave heating rate, ML3DRT emulator"
                plot_ds["heating_ts"].attrs["long_name"] = "Shortwave heating rate, two-stream"
                plot_ds["sza"].attrs["long_name"] = "Solar zenith angle"
                plot_ds["sza"].attrs["units"] = "degrees"
                plot_ds["time"].attrs["long_name"] = "time_since_simulation_start"
                plot_ds["time"].attrs["units"] = "h"

                msg: str = "Saving plotting data to {}...".format(working_filepath)
                print_msg(msg)
                plot_ds.to_netcdf(working_filepath)

                working_filepaths.append(working_filepath)

        #-----------------------------------------------------------------------
        # Plot cached/calculated files
        #-----------------------------------------------------------------------
        working_filepath: str
        for working_filepath in working_filepaths:
            msg: str = "Reading plotting data from {}...".format(working_filepath)
            print_msg(msg)
            with xr.open_dataset(working_filepath) as cache_ds:
                plot_ds: xr.Dataset = cache_ds.load()

            lr_str: str = str(plot_ds.attrs["lr_str"])
            day_str: str = str(plot_ds.attrs["day_str"])
            nszas: NP_INT = NP_INT(plot_ds.sizes["slice"])

            #-------------------------------------------------------------------
            # Read fields and plotting metadata
            #-------------------------------------------------------------------
            x_indices: NP_ARRAY[NP_INT] = plot_ds["x_indices"].to_numpy().astype(NP_INT)
            mnn_szas_plot: NP_ARRAY[NP_REAL] = plot_ds["sza"].to_numpy().astype(NP_REAL)

            y_data_slice_start: NP_INT = NP_INT(plot_ds["y_data_slice_start"].to_numpy())

            y_view_islices: list[slice] = [slice(
                    NP_INT(plot_ds["y_view_slice_start"].isel(slice = ll)) - y_data_slice_start,
                    NP_INT(plot_ds["y_view_slice_stop"].isel(slice = ll)) - y_data_slice_start
                ) for ll in range(0, nszas)]

            #-------------------------------------------------------------------
            # Read fields
            #-------------------------------------------------------------------
            cloud_wc: list[XR_DATAARRAY] = [(plot_ds["cloud_wc"]
                .isel(slice = ll)
                .load()) for ll in range(0, nszas)]   

            heating_rt: list[XR_DATAARRAY] = [(plot_ds["heating_rt"]
                .isel(slice = ll)
                .load()) for ll in range(0, nszas)]

            heating_ts: list[XR_DATAARRAY] = [(plot_ds["heating_ts"]
                .isel(slice = ll)
                .load()) for ll in range(0, nszas)]

            heating_ml3drt: list[XR_DATAARRAY] = [(plot_ds["heating_ml3drt"]
                .isel(slice = ll)
                .load()) for ll in range(0, nszas)]

            #-------------------------------------------------------------------
            # Calculate differences
            #-------------------------------------------------------------------
            heating_ml3drt_diff: list[XR_DATAARRAY] = [
                (heating_ml3drt[ll] - heating_rt[ll])
                for ll in range(0, nszas)]

            heating_ts_diff: list[XR_DATAARRAY] = [
                (heating_ts[ll] - heating_rt[ll])
                for ll in range(0, nszas)]

            #-------------------------------------------------------------------
            # Read grids
            #-------------------------------------------------------------------
            yh: list[XR_DATAARRAY] = [(plot_ds["yh"]
                .load()) for ll in range(0, nszas)] # [km]

            zh: list[XR_DATAARRAY] = [(plot_ds["zh"]
                .load()) for ll in range(0, nszas)] # [km]

            y: list[XR_DATAARRAY] = [(plot_ds["y"]
                .load()) for ll in range(0, nszas)] # [km]

            z: list[XR_DATAARRAY] = [(plot_ds["z"]
                .load()) for ll in range(0, nszas)] # [km]
            
            #-------------------------------------------------------------------
            # Obtain data bounds across both SZA view windows
            #-------------------------------------------------------------------
            cloud_wc_max: list[NP_REAL] = [
                NP_REAL(cloud_wc[ll].isel(y = y_view_islices[ll]).max()) 
                for ll in range(0, nszas)]
            cloud_wc_min: list[NP_REAL] = [
                NP_REAL(cloud_wc[ll].isel(y = y_view_islices[ll]).min()) 
                for ll in range(0, nszas)]

            heating_max: list[NP_REAL] = [max(
                NP_REAL(heating_rt[ll].isel(y = y_view_islices[ll]).max()), 
                NP_REAL(heating_ml3drt[ll].isel(y = y_view_islices[ll]).max()),
                NP_REAL(heating_ts[ll].isel(y = y_view_islices[ll]).max())) 
                for ll in range(0, nszas)]

            heating_min: list[NP_REAL] = [max(1.e-4,
                min(
                NP_REAL(heating_rt[ll].isel(y = y_view_islices[ll]).min()), 
                NP_REAL(heating_ml3drt[ll].isel(y = y_view_islices[ll]).min()),
                NP_REAL(heating_ts[ll].isel(y = y_view_islices[ll]).min()))
                )
                for ll in range(0, nszas)]

            heating_diff_max: list[NP_REAL] = [max(
                NP_REAL(np.abs(heating_ml3drt_diff[ll].isel(y = y_view_islices[ll])).max()),
                NP_REAL(np.abs(heating_ts_diff[ll].isel(y = y_view_islices[ll])).max())) 
                for ll in range(0, nszas)]

            linthresh: NP_REAL = 1.0 # Linear threshold for symlog scale

            heating_norm: colors.LogNorm = colors.LogNorm(
                vmin = max(NP_SMALL, min(heating_min)), 
                vmax = max(heating_max))

            cloud_wc_norm: colors.LogNorm = colors.LogNorm(
                vmin = max(1.e-2, min(cloud_wc_min)), 
                vmax = max(cloud_wc_max))

            heating_diff_bound: NP_REAL = max(NP_SMALL, max(heating_diff_max))
            heating_diff_norm: colors.SymLogNorm = colors.SymLogNorm(
                linthresh = linthresh,
                vmin = -heating_diff_bound,
                vmax = heating_diff_bound)

            #-------------------------------------------------------------------
            # Plot one file for each SZA
            #-------------------------------------------------------------------
            ll: int
            for ll in range(0, nszas):
                msg: str = "Plotting data for SZA {:.1f}...".format(mnn_szas_plot[ll])
                print_msg(msg)

                nrows: NP_INT = NP_INT(3)
                ncols: NP_INT = NP_INT(2)
                fig_height: NP_REAL = NP_REAL(5.0)

                y_lim_min: NP_REAL = NP_REAL(plot_ds["y_view_min"].isel(slice = ll).to_numpy())
                y_lim_max: NP_REAL = NP_REAL(plot_ds["y_view_max"].isel(slice = ll).to_numpy())

                y_view_width_km: NP_REAL = NP_REAL(np.abs(y_lim_max - y_lim_min))
                z_plot_width_km: NP_REAL = NP_REAL(np.abs(
                    NP_REAL(zh[ll].max()) - NP_REAL(zh[ll].min())))

                fig_width: NP_REAL = (
                    y_view_width_km / max(NP_SMALL, z_plot_width_km)
                    ) * (NP_REAL(ncols) / NP_REAL(nrows)) * fig_height

                fig_width = 8.5

                fig_size: list[NP_REAL] = [fig_width, fig_height]

                # Keep the figure size fixed, but allocate more of it to the data panels.
                # Narrower colorbar columns and reduced layout padding make the subplot panels larger.
                cbar_width_ratio: NP_REAL = NP_REAL(0.035)

                try:
                    # Matplotlib >= 3.6.  The "compressed" layout is especially helpful for
                    # fixed-aspect axes such as these height-distance panels.
                    fig: MPL_FIGURE = plt.figure(
                        figsize = fig_size,
                        layout = "compressed")

                    fig.set_layout_engine(
                        "compressed",
                        w_pad = 0.01,
                        h_pad = 0.01,
                        wspace = 0.02,
                        hspace = 0.02)

                except TypeError:
                    # Fallback for older Matplotlib versions.
                    fig: MPL_FIGURE = plt.figure(
                        constrained_layout = True,
                        figsize = fig_size)

                    fig.set_constrained_layout_pads(
                        w_pad = 0.01,
                        h_pad = 0.01,
                        wspace = 0.02,
                        hspace = 0.02)

                gs = fig.add_gridspec(
                    nrows = nrows,
                    ncols = 4,
                    width_ratios = [
                        cbar_width_ratio,
                        NP_REAL(1.0),
                        NP_REAL(1.0),
                        cbar_width_ratio
                    ],
                    wspace = 0.02,
                    hspace = 0.02)

                axs: MPL_AXES = np.empty((nrows, ncols), dtype = object)

                cax_heating: MPL_AXES = fig.add_subplot(gs[:,0])
                cax_cloud_wc: MPL_AXES = fig.add_subplot(gs[0,3])
                cax_heating_diff: MPL_AXES = fig.add_subplot(gs[1:3,3])

                row: int
                col: int
                for row in range(0, nrows):
                    for col in range(0, ncols):
                        sharex_ax: Optional[MPL_AXES] = axs[0,col] if row > 0 else None
                        sharey_ax: Optional[MPL_AXES] = axs[0,0] if (row > 0 or col > 0) else None

                        axs[row,col] = fig.add_subplot(
                            gs[row,col + 1],
                            sharex = sharex_ax,
                            sharey = sharey_ax)

                #---------------------------------------------------------------
                # Column 0: Ray-Tracer, Two-Stream, Emulator
                #---------------------------------------------------------------
                heating_rt_pcm: MPL_PCOLORMESH = axs[0,0].pcolormesh(
                    yh[ll],
                    zh[ll],
                    heating_rt[ll],
                    norm = heating_norm,
                    cmap = heating_cmap, shading = "flat")

                heating_ts_pcm: MPL_PCOLORMESH = axs[1,0].pcolormesh(
                    yh[ll],
                    zh[ll],
                    heating_ts[ll],
                    norm = heating_norm,
                    cmap = heating_cmap, shading = "flat")

                heating_ml3drt_pcm: MPL_PCOLORMESH = axs[2,0].pcolormesh(
                    yh[ll],
                    zh[ll],
                    heating_ml3drt[ll],
                    norm = heating_norm,
                    cmap = heating_cmap, shading = "flat")

                #---------------------------------------------------------------
                # Column 1: Cloud water content and differences
                #---------------------------------------------------------------
                cloud_wc_pcm: MPL_PCOLORMESH = axs[0,1].pcolormesh(
                    yh[ll],
                    zh[ll],
                    cloud_wc[ll],
                    norm = cloud_wc_norm,
                    cmap = cw_cmap, shading = "flat")

                heating_ts_diff_pcm: MPL_PCOLORMESH = axs[1,1].pcolormesh(
                    yh[ll],
                    zh[ll],
                    heating_ts_diff[ll],
                    norm = heating_diff_norm,
                    cmap = diff_cmap, shading = "flat")

                heating_ml3drt_diff_pcm: MPL_PCOLORMESH = axs[2,1].pcolormesh(
                    yh[ll],
                    zh[ll],
                    heating_ml3drt_diff[ll],
                    norm = heating_diff_norm,
                    cmap = diff_cmap, shading = "flat")

                axs[1,1].contour(
                    y[ll],
                    z[ll],
                    heating_ts_diff[ll],
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

                axs[2,1].contour(
                    y[ll],
                    z[ll],
                    heating_ml3drt_diff[ll],
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

                heating_cbar = fig.colorbar(
                    heating_rt_pcm,
                    cax = cax_heating)
                cloud_wc_cbar = fig.colorbar(
                    cloud_wc_pcm,
                    cax = cax_cloud_wc,
                    extend = "min")
                heating_diff_cbar = fig.colorbar(
                    heating_ts_diff_pcm,
                    cax = cax_heating_diff)

                heating_cbar.ax.yaxis.set_ticks_position("left")
                heating_cbar.ax.yaxis.set_label_position("left")

                #---------------------------------------------------------------
                # Axis limits and endpoint ticks
                #---------------------------------------------------------------
                y_min: NP_REAL = NP_REAL(np.abs(np.nanmin(np.asarray(yh))))
                y_max: NP_REAL = NP_REAL(np.abs(np.nanmax(np.asarray(yh))))
                z_min: NP_REAL = NP_REAL(np.abs(np.nanmin(np.asarray(zh))))
                z_max: NP_REAL = NP_REAL(np.abs(np.nanmax(np.asarray(zh))))

                for mm in range(0, nrows):
                    for kk in range(0, ncols):
                        axs[mm,kk].set_xlim(y_min, y_max)
                        axs[mm,kk].set_ylim(z_min, z_max)

                x_ticks: NP_ARRAY[NP_REAL] = np.asarray(axs[0,0].get_xticks(),
                    dtype = NP_REAL)
                y_ticks: NP_ARRAY[NP_REAL] = np.asarray(axs[0,0].get_yticks(),
                    dtype = NP_REAL)

                x_ticks = x_ticks[(x_ticks >= y_min) & (x_ticks <= y_max)]
                y_ticks = y_ticks[(y_ticks >= z_min) & (y_ticks <= z_max)]

                x_ticks = np.unique(np.concatenate((
                    np.array([y_min], dtype = NP_REAL),
                    x_ticks)))
                y_ticks = np.unique(np.concatenate((
                    np.array([z_min], dtype = NP_REAL),
                    y_ticks)))

                for mm in range(0, nrows):
                    for kk in range(0, ncols):
                        axs[mm,kk].set_xticks(x_ticks)
                        axs[mm,kk].set_yticks(y_ticks)
                        axs[mm,kk].tick_params(axis = "x",
                            labelbottom = mm == nrows - 1)
                        axs[mm,kk].tick_params(axis = "y",
                            labelleft = kk == 0)

                #---------------------------------------------------------------
                # Plot contours at powers of 10
                #---------------------------------------------------------------
                cloud_wc_levels: NP_ARRAY[NP_REAL] = NP_REAL(cloud_wc_cbar.ax.get_yticks())
                axs[0,1].contour(
                    y[ll],
                    z[ll],
                    cloud_wc[ll],
                    levels = cloud_wc_levels,
                    colors = "k",
                    linewidths = 1.0
                )
            
                level: NP_REAL
                for level in cloud_wc_levels:
                    cloud_wc_cbar.ax.axhline(
                        level,
                        color = "k",
                        linewidth = 1.0,
                        linestyle = "solid"
                    )

                heating_levels: NP_ARRAY[NP_REAL] = NP_REAL(heating_cbar.ax.get_yticks())

                axs[0,0].contour(
                    y[ll],
                    z[ll],
                    heating_rt[ll],
                    levels = heating_levels,
                    colors = "k",
                    linewidths = 1.0
                )

                axs[1,0].contour(
                    y[ll],
                    z[ll],
                    heating_ts[ll],
                    levels = heating_levels,
                    colors = "k",
                    linewidths = 1.0
                )

                axs[2,0].contour(
                    y[ll],
                    z[ll],
                    heating_ml3drt[ll],
                    levels = heating_levels,
                    colors = "k",
                    linewidths = 1.0
                )
            
                level: NP_REAL
                for level in heating_levels:
                    heating_cbar.ax.axhline(
                        level,
                        color = "k",
                        linewidth = 1.0,
                        linestyle = "solid"
                    )

                #---------------------------------------------------------------
                # Labels
                #---------------------------------------------------------------
                dx: NP_REAL = NP_REAL(plot_ds["dx"].to_numpy()) # [m]
                dx_str: str
                if dx < 1.e3:
                    dx_str = r"{:.0f} $m$".format(dx)
                else:
                    dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

                fig.suptitle(
                    r"Heating Rate $\left[ K\,d^{-1} \right]$"
                    + r" - SZA {:.1f}$^{{\circ}}$".format(mnn_szas_plot[ll]))
                fig.supxlabel(r"y $\left[ km \right]$")
                fig.supylabel(r"z $\left[ km \right]$")

                axs[0,0].set_title(r"Ray-Tracer")
                axs[1,0].set_title(r"Two-Stream")
                axs[2,0].set_title(r"Emulator")

                axs[1,1].set_title(r"Two-Stream - Ray-Tracer")
                axs[2,1].set_title(r"Emulator - Ray-Tracer")

                cloud_wc_cbar.ax.set_ylabel(r"CWC $\left[ g\,m^{-3} \right]$")
                heating_diff_cbar.ax.set_ylabel(r"Difference")

                #---------------------------------------------------------------
                # Additional Colorbar Elements
                #---------------------------------------------------------------
                heating_diff_cbar.ax.axhline(
                    linthresh,
                    color = "k",
                    linestyle = "solid",
                    linewidth = 1.0
                )
                heating_diff_cbar.ax.axhline(
                    -linthresh,
                    color = "k",
                    linestyle = "dashed",
                    linewidth = 1.0
                )

                #---------------------------------------------------------------
                # Additional figure styling
                #---------------------------------------------------------------
                for ax in axs.ravel():
                    ax.set_xlim([y_lim_min, y_lim_max])
                    ax.set_ylim([zh[ll].min(), zh[ll].max()])

                for row in range(0, nrows):
                    for col in range(0, ncols):
                        axs[row,col].set_aspect("equal")

                #---------------------------------------------------------------
                # Save the plot to file
                #---------------------------------------------------------------
                sza_str: str = "sza_{:02d}".format(NP_INT(np.round(mnn_szas_plot[ll])))
                plt_filename: str = "ml3drt_heating_snapshot.{}.{}.{}.png".format(
                    lr_str, day_str, sza_str)
                plt_filepath: str = os.path.join(rad_tran_vizdir, plt_filename)
                fig.savefig(plt_filepath, dpi = 200)
                plt.close(fig)

if __name__ == "__main__":
    main()