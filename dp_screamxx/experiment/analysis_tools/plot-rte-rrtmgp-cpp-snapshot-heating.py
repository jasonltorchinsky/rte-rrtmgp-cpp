#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys
experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
from datetime import datetime
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
    calc_cloud_wc, calc_sw_heating, calc_z_max_info, find_grid, find_y_islice, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-snapshot-heating"
prog_desc: str = "Visualize atmospheric heating rates for RTE-RRTMGP-CPP."

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
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate plotted quantities and save them to file.")
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
    rad_tran_outfiles: list[str]
    [rad_tran_infiles, rad_tran_outfiles] = find_inout_pairs(rad_tran_indir,
        rad_tran_outdir, coarse_factors)

    nfiles: NP_INT = NP_INT(len(rad_tran_infiles))

    lr_re: re.Pattern = re.compile("lr_..")

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        plt_data_filename: str = "rte_rrtmgp_cpp_rad_tran_snapshot_heating.{}.nc".format(lr_str)
        plt_data_filepath: str = os.path.join(working_dir, plt_data_filename)

        necessary_data_vars: list[str] = [
            "cloud_wc",
            "heating_ts",
            "heating_rt",
            "heating_diff",
            "y",
            "yh",
            "x",
            "y_count",
            "yh_count",
            "x_indices",
            "y_slice_start",
            "y_slice_stop",
            "mnn_indices",
            "mnn_times",
            "mnn_szas"
        ]
        necessary_coords: list[str] = [
            "day",
            "plot_slice",
            "z",
            "zh",
            "y_plot_index",
            "yh_plot_index"
        ]
        necessary_attrs: list[str] = [
            "lr_str",
            "dx_m",
            "zh_max_km",
            "y_slice_width_km",
            "cloud_wc_units",
            "heating_units",
            "y_units",
            "z_units",
            "x_units",
            "sza_units",
            "time_units"
        ]
        necessary_dims: dict[str, tuple[str, ...]] = dict(
            cloud_wc = ("day", "plot_slice", "z", "y_plot_index"),
            heating_ts = ("day", "plot_slice", "z", "y_plot_index"),
            heating_rt = ("day", "plot_slice", "z", "y_plot_index"),
            heating_diff = ("day", "plot_slice", "z", "y_plot_index"),
            y = ("day", "plot_slice", "y_plot_index"),
            yh = ("day", "plot_slice", "yh_plot_index"),
            x = ("day", "plot_slice"),
            y_count = ("day", "plot_slice"),
            yh_count = ("day", "plot_slice"),
            x_indices = ("day", "plot_slice"),
            y_slice_start = ("day", "plot_slice"),
            y_slice_stop = ("day", "plot_slice"),
            mnn_indices = ("day", "plot_slice"),
            mnn_times = ("day", "plot_slice"),
            mnn_szas = ("day", "plot_slice")
        )

        #-----------------------------------------------------------------------
        # Read calculated fields, if available and complete
        #-----------------------------------------------------------------------
        plot_ds: xr.Dataset
        calculate_plot_data: bool = recalculate

        if calculate_plot_data:
            msg: str = "Recalculating plotted quantities..."
            print_msg(msg)
        elif not os.path.exists(plt_data_filepath):
            msg: str = "Plotted quantities not found; calculating..."
            print_msg(msg)
            calculate_plot_data = True
        else:
            msg: str = "Reading plotted quantities from {}...".format(plt_data_filepath)
            print_msg(msg)
            with xr.open_dataset(plt_data_filepath) as plot_ds_tmp:
                plot_ds = plot_ds_tmp.load()

            plot_ds_complete: bool = True

            var_name: str
            for var_name in necessary_data_vars:
                if var_name not in plot_ds.data_vars:
                    plot_ds_complete = False

            coord_name: str
            for coord_name in necessary_coords:
                if coord_name not in plot_ds.coords:
                    plot_ds_complete = False

            attr_name: str
            for attr_name in necessary_attrs:
                if attr_name not in plot_ds.attrs:
                    plot_ds_complete = False

            if plot_ds_complete:
                for var_name in necessary_data_vars:
                    if plot_ds[var_name].dims != necessary_dims[var_name]:
                        plot_ds_complete = False

            if not plot_ds_complete:
                msg: str = "Plotted quantities file does not contain all necessary information; recalculating..."
                print_msg(msg)
                calculate_plot_data = True

        #-----------------------------------------------------------------------
        # Calculate and save fields needed for plotting, if requested or needed
        #-----------------------------------------------------------------------
        if calculate_plot_data:
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
            mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
            ndays: NP_INT = NP_INT(mnn_indices.shape[0])
            nplot_slices: NP_INT = NP_INT(mnn_indices.shape[1])
            z_max_info: dict = calc_z_max_info(
                rad_tran_infile, 
                z_max = z_max
                )

            cloud_wc_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            heating_rt_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            heating_ts_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            y_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            yh_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            z_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            zh_all: list[list[XR_DATAARRAY]] = [[] for _ in range(0, ndays)]
            x_indices_all: NP_ARRAY[NP_INT] = np.zeros((ndays, nplot_slices), dtype = NP_INT)
            y_slice_start_all: NP_ARRAY[NP_INT] = np.zeros((ndays, nplot_slices), dtype = NP_INT)
            y_slice_stop_all: NP_ARRAY[NP_INT] = np.zeros((ndays, nplot_slices), dtype = NP_INT)

            #-------------------------------------------------------------------
            # Calculate fields for each MNN of each day
            #-------------------------------------------------------------------
            jj: int
            for jj in range(0, ndays):
                #---------------------------------------------------------------
                # Calculate spatial extent of plots based on maximal cloud water content
                #---------------------------------------------------------------
                msg: str = "Calculating plot spatial extent for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile,
                    time_indices = mnn_indices[jj],
                    z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
            
                x_indices: NP_ARRAY[NP_INT] = np.array([np.unravel_index(np.argmax(cloud_wc.isel(time = ll).to_numpy()), cloud_wc.shape[1:])[2] for ll in range(0, nplot_slices)])
                y_slice_width: NP_REAL = 3. * z_max_info["zh_max"] # Width of y-slices [km]
                y_islices: list[slice] = [[] for _ in range(0, nplot_slices)]
                for ll in range(0, nplot_slices):
                    y_islices[ll] = find_y_islice(
                        grid["y"],
                        cloud_wc.isel(time = ll, x = x_indices[ll]),
                        slice_width = y_slice_width
                    )

                #---------------------------------------------------------------
                # Calculate desired atmospheric and radiative quantities
                #---------------------------------------------------------------
                msg: str = "Calculating desired atmospheric quantities for day {} of {}...".format(jj, ndays - 1)
                print_msg(msg)

                cloud_wc: XR_DATAARRAY = calc_cloud_wc(
                    rad_tran_infile, 
                    time_indices = mnn_indices[jj], 
                    x_indices = x_indices, 
                    z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [slice, lay, y]
                heating_rt: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = mnn_indices[jj], 
                    x_indices = x_indices, 
                    z_max_info = z_max_info,
                    solver = "rt") # Shortwave heating rate, ray-tracer; [K d^{-1}]; [slice, lay, y]
                heating_ts: XR_DATAARRAY = calc_sw_heating(
                    rad_tran_infile,
                    rad_tran_outfile,
                    time_indices = mnn_indices[jj], 
                    x_indices = x_indices, 
                    z_max_info = z_max_info,
                    solver = "ts") # Shortwave heating rate, two-stream; [K d^{-1}]; [slice, lay, y]

                #---------------------------------------------------------------
                # Trim fields to the yz-slice
                #---------------------------------------------------------------
                cloud_wc_all[jj] = [(cloud_wc
                    .isel(y = y_islices[ll], slice = ll)
                    .load()) for ll in range(0, nplot_slices)]   
                heating_rt_all[jj] = [(heating_rt
                    .isel(y = y_islices[ll], slice = ll)
                    .load()) for ll in range(0, nplot_slices)]
                heating_ts_all[jj] = [(heating_ts
                    .isel(y = y_islices[ll], slice = ll)
                    .load()) for ll in range(0, nplot_slices)]

                #---------------------------------------------------------------
                # Trim grids to the yz-slice
                #---------------------------------------------------------------
                yh_islices: list[slice] = [slice(
                        y_islices[ll].start, y_islices[ll].stop + 1
                    ) for ll in range(0, nplot_slices)]
                yh_all[jj] = [(grid["yh"]
                    .isel(yh = yh_islices[ll])
                    .load()) * 1.e-3 for ll in range(0, nplot_slices)] # [m] => [km]

                zh_all[jj] = [(grid["zh"]
                    .isel(zh = z_max_info["isel_indexers"]["zh"])
                    .load()) * 1.e-3 for ll in range(0, nplot_slices)] # [m] => [km]

                y_all[jj] = [(grid["y"]
                    .isel(y = y_islices[ll])
                    .load()) * 1.e-3 for ll in range(0, nplot_slices)] # [m] => [km]

                z_all[jj] = [(grid["z"]
                    .isel(z = z_max_info["isel_indexers"]["z"])
                    .load()) * 1.e-3 for ll in range(0, nplot_slices)] # [m] => [km]

                x_indices_all[jj,:] = x_indices
                for ll in range(0, nplot_slices):
                    y_slice_start_all[jj,ll] = y_islices[ll].start
                    y_slice_stop_all[jj,ll] = y_islices[ll].stop

            #-------------------------------------------------------------------
            # Pack plotted quantities into a single xarray dataset
            #-------------------------------------------------------------------
            msg: str = "Packing plotted quantities into dataset..."
            print_msg(msg)

            ny_max: NP_INT = NP_INT(max([
                y_all[jj][ll].shape[0]
                for jj in range(0, ndays)
                for ll in range(0, nplot_slices)
            ]))
            nyh_max: NP_INT = NP_INT(max([
                yh_all[jj][ll].shape[0]
                for jj in range(0, ndays)
                for ll in range(0, nplot_slices)
            ]))
            nz: NP_INT = NP_INT(z_all[0][0].shape[0])
            nzh: NP_INT = NP_INT(zh_all[0][0].shape[0])

            cloud_wc_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices, nz, ny_max), np.nan, dtype = NP_REAL)
            heating_rt_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices, nz, ny_max), np.nan, dtype = NP_REAL)
            heating_ts_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices, nz, ny_max), np.nan, dtype = NP_REAL)
            heating_diff_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices, nz, ny_max), np.nan, dtype = NP_REAL)
            y_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices, ny_max), np.nan, dtype = NP_REAL)
            yh_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices, nyh_max), np.nan, dtype = NP_REAL)
            x_data: NP_ARRAY[NP_REAL] = np.full(
                (ndays, nplot_slices), np.nan, dtype = NP_REAL)
            y_count_data: NP_ARRAY[NP_INT] = np.zeros(
                (ndays, nplot_slices), dtype = NP_INT)
            yh_count_data: NP_ARRAY[NP_INT] = np.zeros(
                (ndays, nplot_slices), dtype = NP_INT)

            for jj in range(0, ndays):
                for ll in range(0, nplot_slices):
                    y_count: NP_INT = NP_INT(y_all[jj][ll].shape[0])
                    yh_count: NP_INT = NP_INT(yh_all[jj][ll].shape[0])

                    cloud_wc_data[jj,ll,:,0:y_count] = cloud_wc_all[jj][ll].to_numpy()
                    heating_rt_data[jj,ll,:,0:y_count] = heating_rt_all[jj][ll].to_numpy()
                    heating_ts_data[jj,ll,:,0:y_count] = heating_ts_all[jj][ll].to_numpy()
                    heating_diff_data[jj,ll,:,0:y_count] = (
                        heating_ts_all[jj][ll] - heating_rt_all[jj][ll]
                    ).to_numpy()
                    y_data[jj,ll,0:y_count] = y_all[jj][ll].to_numpy()
                    yh_data[jj,ll,0:yh_count] = yh_all[jj][ll].to_numpy()
                    x_data[jj,ll] = NP_REAL(grid["x"].isel(x = x_indices_all[jj,ll])) * 1.e-3 # [m] => [km]
                    y_count_data[jj,ll] = y_count
                    yh_count_data[jj,ll] = yh_count

            z_data: NP_ARRAY[NP_REAL] = z_all[0][0].to_numpy()
            zh_data: NP_ARRAY[NP_REAL] = zh_all[0][0].to_numpy()
            dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]

            plot_ds = xr.Dataset(
                data_vars = dict(
                    cloud_wc = (["day", "plot_slice", "z", "y_plot_index"], cloud_wc_data),
                    heating_ts = (["day", "plot_slice", "z", "y_plot_index"], heating_ts_data),
                    heating_rt = (["day", "plot_slice", "z", "y_plot_index"], heating_rt_data),
                    heating_diff = (["day", "plot_slice", "z", "y_plot_index"], heating_diff_data),
                    y = (["day", "plot_slice", "y_plot_index"], y_data),
                    yh = (["day", "plot_slice", "yh_plot_index"], yh_data),
                    x = (["day", "plot_slice"], x_data),
                    y_count = (["day", "plot_slice"], y_count_data),
                    yh_count = (["day", "plot_slice"], yh_count_data),
                    x_indices = (["day", "plot_slice"], x_indices_all),
                    y_slice_start = (["day", "plot_slice"], y_slice_start_all),
                    y_slice_stop = (["day", "plot_slice"], y_slice_stop_all),
                    mnn_indices = (["day", "plot_slice"], mnn_indices),
                    mnn_times = (["day", "plot_slice"], mnn_times),
                    mnn_szas = (["day", "plot_slice"], mnn_szas),
                ),
                coords = dict(
                    day = np.arange(0, ndays, dtype = NP_INT),
                    plot_slice = np.arange(0, nplot_slices, dtype = NP_INT),
                    z = z_data,
                    zh = zh_data,
                    y_plot_index = np.arange(0, ny_max, dtype = NP_INT),
                    yh_plot_index = np.arange(0, nyh_max, dtype = NP_INT),
                ),
                attrs = dict(
                    lr_str = lr_str,
                    dx_m = dx,
                    zh_max_km = NP_REAL(z_max_info["zh_max"]),
                    y_slice_width_km = NP_REAL(y_slice_width),
                    cloud_wc_units = "g m^{-3}",
                    heating_units = "K d^{-1}",
                    y_units = "km",
                    z_units = "km",
                    x_units = "km",
                    sza_units = "degrees",
                    time_units = "hours since simulation start",
                )
            )

            #-------------------------------------------------------------------
            # Save plotted quantities
            #-------------------------------------------------------------------
            msg: str = "Saving plotted quantities to {}...".format(plt_data_filepath)
            print_msg(msg)
            plot_ds.to_netcdf(plt_data_filepath)

        ndays: NP_INT = NP_INT(plot_ds.sizes["day"])

        #-----------------------------------------------------------------------
        # Plot saved data for each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            msg: str = "Plotting data for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            nrows: NP_INT = NP_INT(4)
            ncols: NP_INT = NP_INT(2)

            y_count: list[NP_INT] = [
                NP_INT(plot_ds["y_count"].isel(day = jj, plot_slice = ll).item())
                for ll in range(0, ncols)
            ]
            yh_count: list[NP_INT] = [
                NP_INT(plot_ds["yh_count"].isel(day = jj, plot_slice = ll).item())
                for ll in range(0, ncols)
            ]

            cloud_wc: list[XR_DATAARRAY] = [(plot_ds["cloud_wc"]
                .isel(day = jj, plot_slice = ll, y_plot_index = slice(0, y_count[ll]))
                .load()) for ll in range(0, ncols)]
            heating_ts: list[XR_DATAARRAY] = [(plot_ds["heating_ts"]
                .isel(day = jj, plot_slice = ll, y_plot_index = slice(0, y_count[ll]))
                .load()) for ll in range(0, ncols)]
            heating_rt: list[XR_DATAARRAY] = [(plot_ds["heating_rt"]
                .isel(day = jj, plot_slice = ll, y_plot_index = slice(0, y_count[ll]))
                .load()) for ll in range(0, ncols)]
            heating_diff: list[XR_DATAARRAY] = [(plot_ds["heating_diff"]
                .isel(day = jj, plot_slice = ll, y_plot_index = slice(0, y_count[ll]))
                .load()) for ll in range(0, ncols)]

            yh: list[XR_DATAARRAY] = [(plot_ds["yh"]
                .isel(day = jj, plot_slice = ll, yh_plot_index = slice(0, yh_count[ll]))
                .load()) for ll in range(0, ncols)]

            zh: list[XR_DATAARRAY] = [(plot_ds["zh"]
                .load()) for ll in range(0, ncols)]

            y: list[XR_DATAARRAY] = [(plot_ds["y"]
                .isel(day = jj, plot_slice = ll, y_plot_index = slice(0, y_count[ll]))
                .load()) for ll in range(0, ncols)]

            z: list[XR_DATAARRAY] = [(plot_ds["z"]
                .load()) for ll in range(0, ncols)]

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            cloud_wc_max: list[NP_REAL] = [
                NP_REAL(np.nanmax(cloud_wc[ll].to_numpy())) 
                for ll in range(0, ncols)]
            cloud_wc_min: list[NP_REAL] = [
                NP_REAL(np.nanmin(cloud_wc[ll].to_numpy())) 
                for ll in range(0, ncols)]

            heating_max: list[NP_REAL] = [max(
                NP_REAL(np.nanmax(heating_ts[ll].to_numpy())), 
                NP_REAL(np.nanmax(heating_rt[ll].to_numpy()))) 
                for ll in range(0, ncols)]
            heating_min: list[NP_REAL] = [min(
                NP_REAL(np.nanmin(heating_ts[ll].to_numpy())), 
                NP_REAL(np.nanmin(heating_rt[ll].to_numpy()))) 
                for ll in range(0, ncols)]

            heating_diff_max: list[NP_REAL] = [
                NP_REAL(np.nanmax(np.abs(heating_diff[ll].to_numpy()))) 
                for ll in range(0, ncols)]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            fig_height: NP_REAL = NP_REAL(7.0)
            panel_fig_width: NP_REAL = (
                NP_REAL(plot_ds.attrs["y_slice_width_km"]) 
                / NP_REAL(plot_ds.attrs["zh_max_km"])
                ) * (NP_REAL(ncols) / NP_REAL(nrows)) * fig_height
            fig_width: NP_REAL = panel_fig_width * NP_REAL(1.12)
            fig_size: list[NP_REAL] = [fig_width, fig_height]

            fig: MPL_FIGURE = plt.figure(figsize = fig_size)
            gs = fig.add_gridspec(
                nrows = nrows,
                ncols = ncols + 1,
                width_ratios = [1.0 for _ in range(0, ncols)] + [0.06],
                height_ratios = [1.0 for _ in range(0, nrows)],
                left = 0.08,
                right = 0.93,
                bottom = 0.08,
                top = 0.90,
                wspace = 0.12,
                hspace = 0.10
            )

            axs: MPL_AXES = np.empty((nrows, ncols), dtype = object)
            row: int
            col: int
            for row in range(0, nrows):
                for col in range(0, ncols):
                    if row == 0 and col == 0:
                        axs[row,col] = fig.add_subplot(gs[row,col])
                    elif row == 0:
                        axs[row,col] = fig.add_subplot(
                            gs[row,col],
                            sharey = axs[0,0]
                        )
                    elif col == 0:
                        axs[row,col] = fig.add_subplot(
                            gs[row,col],
                            sharex = axs[0,col],
                            sharey = axs[0,0]
                        )
                    else:
                        axs[row,col] = fig.add_subplot(
                            gs[row,col],
                            sharex = axs[0,col],
                            sharey = axs[0,0]
                        )

            cloud_wc_cax = fig.add_subplot(gs[0,ncols])
            heating_cax = fig.add_subplot(gs[1:3,ncols])
            heating_diff_cax = fig.add_subplot(gs[3,ncols])

            linthresh: NP_REAL = 1.0 # Linear threshold for symlog scale

            # Row 0: Cloud Water Content
            cloud_wc_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                cloud_wc_pcm[ll] = axs[0, ll].pcolormesh(
                    yh[ll],
                    zh[ll],
                    cloud_wc[ll],
                    norm = colors.LogNorm(
                        vmin = max(1.e-2, min(cloud_wc_min)), 
                        vmax = max(cloud_wc_max)),
                    cmap = cw_cmap, shading = "flat")

            # Row 1: Heating, Two-Stream
            heating_ts_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                heating_ts_pcm[ll] = axs[1,ll].pcolormesh(
                    yh[ll], 
                    zh[ll], 
                    heating_ts[ll],
                    norm = colors.LogNorm(
                        vmin = min(heating_min), 
                        vmax = max(heating_max)),
                    cmap = heating_cmap, shading = "flat")

            # Row 2: Heating, Ray-Tracer
            heating_rt_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                heating_rt_pcm[ll] = axs[2,ll].pcolormesh(
                    yh[ll], 
                    zh[ll], 
                    heating_rt[ll],
                    norm = colors.LogNorm(
                        vmin = min(heating_min), 
                        vmax = max(heating_max)),
                    cmap = heating_cmap, shading = "flat")

            # Row 3: Heating, Difference
            heating_diff_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                heating_diff_pcm[ll] = axs[3,ll].pcolormesh(
                    yh[ll], 
                    zh[ll], 
                    heating_diff[ll],
                    norm = colors.SymLogNorm(
                        linthresh = linthresh,
                        vmin = -max(heating_diff_max),
                        vmax = max(heating_diff_max)),
                    cmap = diff_cmap, shading = "flat")

                axs[3,ll].contour(
                    y[ll],
                    z[ll],
                    heating_diff[ll],
                    levels = [-linthresh, linthresh],
                    colors = "k",
                    linewidths = 1.0,
                    negative_linestyles = "dashed"
                )

            cloud_wc_cbar = fig.colorbar(
                cloud_wc_pcm[0],
                cax = cloud_wc_cax,
                extend = "min"
            )
            heating_cbar = fig.colorbar(
                heating_ts_pcm[0],
                cax = heating_cax
            )
            heating_diff_cbar = fig.colorbar(
                heating_diff_pcm[0],
                cax = heating_diff_cax
            )

            #-------------------------------------------------------------------
            # Plot contours at powers of 10
            #-------------------------------------------------------------------
            # CWC
            cloud_wc_levels: NP_ARRAY[NP_REAL] = np.array(
                cloud_wc_cbar.ax.get_yticks(),
                dtype = NP_REAL
            )
            # Row 0: Cloud Water Content
            row: NP_INT = NP_INT(0)
            ll: int
            for ll in range(0, ncols):
                axs[row,ll].contour(
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

            # Heating rate
            heating_levels: NP_ARRAY[NP_REAL] = np.array(
                heating_cbar.ax.get_yticks(),
                dtype = NP_REAL
            )
            # Row 1: Heating, Two-Stream
            ll: int
            row: NP_INT = NP_INT(1)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    y[ll],
                    z[ll],
                    heating_ts[ll],
                    levels = heating_levels,
                    colors = "k",
                    linewidths = 1.0
                )

            row: NP_INT = NP_INT(2)
            for ll in range(0, ncols):
                axs[row,ll].contour(
                    y[ll],
                    z[ll],
                    heating_rt[ll],
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

            #-------------------------------------------------------------------
            # Labels
            #-------------------------------------------------------------------
            dx: NP_REAL = NP_REAL(plot_ds.attrs["dx_m"]) # [m]
            dx_str: str
            if dx < 1.e3:
                dx_str = r"{:.0f} $m$".format(dx)
            else:
                dx_str = r"{:.2f} $km$".format(dx * 1.e-3)

            fig.suptitle(
                r"Heating Rate $\left[ K\,d^{-1} \right]$" + " - {}".format(dx_str),
                y = 0.975
            )
            fig.supxlabel(r"y $\left[ km \right]$", y = 0.02)
            fig.supylabel(r"z $\left[ km \right]$", x = 0.02)

            for ll in range(0, ncols):
                x_pos: NP_REAL = NP_REAL(plot_ds["x"].isel(day = jj, plot_slice = ll)) # [km]
                # col_title: str = (r"{:.2f} Hours - ".format(NP_REAL(plot_ds["mnn_times"].isel(day = jj, plot_slice = ll)))
                #     + r"Solar Zenith Angle {:.1f}$^{{\circ}}$ - ".format(NP_REAL(plot_ds["mnn_szas"].isel(day = jj, plot_slice = ll)))
                #     + r"$x$ = {:.2f} $\left[ km \right]$".format(x_pos))
                col_title: str = r"SZA {:.1f}$^{{\circ}}$".format(NP_REAL(plot_ds["mnn_szas"].isel(day = jj, plot_slice = ll)))
                axs[0,ll].set_title(col_title)
            axs[1,0].set_ylabel(r"Two-Stream")
            axs[2,0].set_ylabel(r"Ray-Tracer")
            axs[3,0].set_ylabel(r"Two-Stream - Ray-Tracer")

            cloud_wc_cbar.ax.set_ylabel(r"CWC $\left[ g\,m^{-3} \right]$")
            heating_diff_cbar.ax.set_ylabel(r"Difference")

            #-------------------------------------------------------------------
            # Additional Colorbar Elements
            #-------------------------------------------------------------------
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

            #-------------------------------------------------------------------
            # Additional figure styling
            #-------------------------------------------------------------------
            ll: int
            for ll in range(0, ncols):
                for ax in axs[:,ll]:
                    ax.set_xlim([yh[ll].min(), yh[ll].max()])

            row: int
            col: int
            for row in range(0, nrows - 1):
                for col in range(0, ncols):
                    plt.setp(axs[row,col].get_xticklabels(), visible = False)

            for row in range(0, nrows):
                for col in range(1, ncols):
                    plt.setp(axs[row,col].get_yticklabels(), visible = False)

            # Aspect ratio
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_aspect("equal")

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_snapshot_heating.{}.{}.png".format(lr_str, day_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()