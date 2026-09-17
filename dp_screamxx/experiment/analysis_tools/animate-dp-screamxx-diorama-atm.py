#-------------------------------------------------------------------------------
# Append the 'experiment' directory to the PYTHONPATH for future imports
#-------------------------------------------------------------------------------
import os, sys, time
try:
    import resource
except ImportError:
    resource = None

experiment_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
if experiment_dir not in sys.path:
    sys.path.append(experiment_dir)

# Standard Library Imports
from argparse import ArgumentParser, Namespace
from typing import Optional

# Third-Party Library Imports
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    XR_DATAARRAY, XR_DATASET, \
    MPL_AXES, MPL_FIGURE, MPL_LINEAR_SEGMENTED_COLORMAP, MPL_LOGNORM, \
    MPL_COLORBAR
from consts.visual import cloud_cmap
from dpscream import get_sort_mask, calc_cloud_wc, print_msg

# Script variables
prog_name: str = "animate-dp-screamxx-diorama-atm"
prog_desc: str = "Animate DP-SCREAM cloud water content as a 3D atmospheric diorama."

def main():
    script_start_time: float = time.perf_counter()

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb: NP_REAL = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg: str = "Starting script. Max RSS: {:.1f} MiB.".format(max_rss_mb)
    else:
        msg = "Starting script. Max RSS unavailable on this platform."
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    section_start_time: float = time.perf_counter()

    msg = "Parsing command-line input..."
    print_msg(msg)

    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--dp-scream-file", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path to DP-SCREAM output file.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", action = "store_true",
        help = "Re-calculate plotting quantities and save them to the working directory.")
    parser.add_argument("--z-max", nargs = "?", default = 16., type = float,
        help = "Maximum plotted height [km]. Use <= 0 to plot full available height.")
    parser.add_argument("--cwc-tol", nargs = "?", default = 1.e-1, type = float,
        help = "Cloud water content plotting tolerance [g m^{-3}].")
    parser.add_argument("--xy-stride", nargs = "?", default = 1, type = int,
        help = "Horizontal stride for plotting voxels.")
    parser.add_argument("--frames", nargs = "?", default = None, type = int,
        help = "Number of animation frames to use. If absent, use all time steps.")
    parser.add_argument("--voxels", nargs = "?", default = None, type = int,
        help = ("Number of voxels with maximal cloud water content to plot per frame. "
            "If absent, plot all voxels above the tolerance value."))
    parser.add_argument("--detailed-calc", action = "store_true",
        help = ("True: Compute cloud water mass using VMRs, etc. "
            "False: Compute cloud water mass using standard values."))
    parser.add_argument("--fps", nargs = "?", default = 5, type = int,
        help = "Frames per second for the output animation.")
    parser.add_argument("--bitrate", nargs = "?", default = 2400, type = int,
        help = "FFmpeg writer bitrate.")
        
    args: Namespace = parser.parse_args()

    dp_scream_file: str = os.path.normpath(args.dp_scream_file)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None
    cwc_tol: NP_REAL = NP_REAL(args.cwc_tol)
    xy_stride: NP_INT = NP_INT(args.xy_stride)
    frames: Optional[NP_INT] = NP_INT(args.frames) if args.frames is not None else None
    voxels: Optional[NP_INT] = NP_INT(args.voxels) if args.voxels is not None else None
    detailed_calc: bool = args.detailed_calc
    fps: NP_INT = NP_INT(args.fps)
    bitrate: NP_INT = NP_INT(args.bitrate)

    if not os.path.exists(dp_scream_file):
        msg = "DP-SCREAM file does not exist: {}".format(dp_scream_file)
        print_msg(msg)
        sys.exit(1)

    if xy_stride < 1:
        xy_stride = NP_INT(1)

    if frames is not None:
        if frames < 1:
            frames = None

    if voxels is not None:
        if voxels < 1:
            voxels = None

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Parsed command-line input in {:.2f} s. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                max_rss_mb)
    else:
        msg = "Parsed command-line input in {:.2f} s.".format(
            time.perf_counter() - section_start_time)
    print_msg(msg)

    msg = ("Run configuration: frames = {}, xy_stride = {}, z_max = {}, "
        "cwc_tol = {:.4e}, voxels = {}, detailed_calc = {}, fps = {}, "
        "bitrate = {}.").format(
            frames,
            xy_stride,
            z_max,
            cwc_tol,
            voxels,
            detailed_calc,
            fps,
            bitrate)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Ensure directories exist
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    dir_names: list[str] = [rad_tran_vizdir, working_dir]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Ensured output directories exist in {:.2f} s. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                max_rss_mb)
    else:
        msg = "Ensured output directories exist in {:.2f} s.".format(
            time.perf_counter() - section_start_time)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Obtain all time indices and times
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Obtaining time information..."
    print_msg(msg)

    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4",
        decode_times = False, decode_timedelta = False) as xr_dp_scream:

        n_time_full: NP_INT = NP_INT(xr_dp_scream.sizes["time"])

        time_indices: NP_ARRAY[NP_INT] = np.arange(0, n_time_full, dtype = NP_INT)

        if frames is not None:
            time_indices = time_indices[0:frames]

        time_raw: NP_ARRAY[NP_REAL] = (
            xr_dp_scream["time"]
            .isel(time = time_indices)
            .to_numpy()
            .astype(NP_REAL))

        time_units: str = xr_dp_scream["time"].attrs.get("units", "")

        if "days since" in time_units:
            animation_times: NP_ARRAY[NP_REAL] = time_raw * NP_REAL(24.) # [days] => [h]
        elif "hours since" in time_units:
            animation_times: NP_ARRAY[NP_REAL] = time_raw # [h]
        elif "minutes since" in time_units:
            animation_times: NP_ARRAY[NP_REAL] = time_raw / NP_REAL(60.) # [min] => [h]
        elif "seconds since" in time_units:
            animation_times: NP_ARRAY[NP_REAL] = time_raw / NP_REAL(3600.) # [s] => [h]
        else:
            msg = "Warning: Unrecognized time units '{}'. Assuming days.".format(time_units)
            print_msg(msg)
            animation_times = time_raw * NP_REAL(24.) # Assume [days] => [h]

    n_t: NP_INT = NP_INT(time_indices.size)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Obtained time information in {:.2f} s. "
            "Using {} frame(s) out of {} available time step(s). "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                n_t,
                n_time_full,
                max_rss_mb)
    else:
        msg = ("Obtained time information in {:.2f} s. "
            "Using {} frame(s) out of {} available time step(s).").format(
                time.perf_counter() - section_start_time,
                n_t,
                n_time_full)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Set up plotting information cache file
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Checking for saved plotting information..."
    print_msg(msg)

    z_max_attr: NP_REAL = NP_REAL(z_max) if z_max is not None else NP_REAL(-1.)
    frames_attr: NP_INT = NP_INT(frames) if frames is not None else NP_INT(-1)
    horizontal_bounds_method: str = "domain_extent"

    nc_filename: str = "dpscream_atm_diorama_plotting_info.nc"
    nc_filepath: str = os.path.join(working_dir, nc_filename)

    required_variables: list[str] = [
        "cwc",
        "zzh",
        "time_indices",
        "animation_times"
    ]
    required_coords: list[str] = [
        "x",
        "y",
        "xh",
        "yh",
        "z",
        "zh"
    ]

    read_from_file: bool = False
    if (not recalculate) and os.path.exists(nc_filepath):
        with xr.open_dataset(nc_filepath) as ds_check:
            missing_variable: bool = False
            missing_coord: bool = False

            var_name: str
            for var_name in required_variables:
                if var_name not in ds_check:
                    missing_variable = True

            coord_name: str
            for coord_name in required_coords:
                if coord_name not in ds_check.coords:
                    missing_coord = True

            attrs_ok: bool = True
            if "dp_scream_file" not in ds_check.attrs:
                attrs_ok = False
            elif os.path.normpath(ds_check.attrs["dp_scream_file"]) != dp_scream_file:
                attrs_ok = False

            if "xy_stride" not in ds_check.attrs:
                attrs_ok = False
            elif NP_INT(ds_check.attrs["xy_stride"]) != xy_stride:
                attrs_ok = False

            if "z_max" not in ds_check.attrs:
                attrs_ok = False
            elif not np.isclose(NP_REAL(ds_check.attrs["z_max"]), z_max_attr):
                attrs_ok = False

            if "frames" not in ds_check.attrs:
                attrs_ok = False
            elif NP_INT(ds_check.attrs["frames"]) != frames_attr:
                attrs_ok = False

            if "detailed_calc" not in ds_check.attrs:
                attrs_ok = False
            elif bool(NP_INT(ds_check.attrs["detailed_calc"])) != detailed_calc:
                attrs_ok = False

            if "time_units" not in ds_check.attrs:
                attrs_ok = False
            elif ds_check.attrs["time_units"] != time_units:
                attrs_ok = False

            if "horizontal_bounds_method" not in ds_check.attrs:
                attrs_ok = False
            elif ds_check.attrs["horizontal_bounds_method"] != horizontal_bounds_method:
                attrs_ok = False

            dims_ok: bool = True
            if "time" not in ds_check.sizes:
                dims_ok = False
            elif NP_INT(ds_check.sizes["time"]) != n_t:
                dims_ok = False

            time_indices_ok: bool = True
            if (not missing_variable) and ("time_indices" in ds_check):
                cached_time_indices: NP_ARRAY[NP_INT] = ds_check["time_indices"].to_numpy().astype(NP_INT)
                if cached_time_indices.size != time_indices.size:
                    time_indices_ok = False
                elif not np.all(cached_time_indices == time_indices):
                    time_indices_ok = False
            else:
                time_indices_ok = False

            if (not missing_variable) and (not missing_coord) and attrs_ok and dims_ok and time_indices_ok:
                read_from_file = True

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Checked saved plotting information in {:.2f} s. "
            "read_from_file = {}. Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                read_from_file,
                max_rss_mb)
    else:
        msg = ("Checked saved plotting information in {:.2f} s. "
            "read_from_file = {}.").format(
                time.perf_counter() - section_start_time,
                read_from_file)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Read or calculate plotting information
    #---------------------------------------------------------------------------
    cwc_np: NP_ARRAY[NP_REAL]
    zzh: NP_ARRAY[NP_REAL]
    x: NP_ARRAY[NP_REAL]
    y: NP_ARRAY[NP_REAL]
    xh: NP_ARRAY[NP_REAL]
    yh: NP_ARRAY[NP_REAL]

    section_start_time = time.perf_counter()

    if read_from_file:
        msg = "Reading plotting information from file..."
        print_msg(msg)

        with xr.open_dataset(nc_filepath) as ds_plot:
            cwc_np = ds_plot["cwc"].to_numpy().astype(NP_REAL) # [time, z, y, x]
            zzh = ds_plot["zzh"].to_numpy().astype(NP_REAL) # [time, xh, yh, zh]
            x = ds_plot["x"].to_numpy().astype(NP_REAL) # [km]
            y = ds_plot["y"].to_numpy().astype(NP_REAL) # [km]
            xh = ds_plot["xh"].to_numpy().astype(NP_REAL) # [km]
            yh = ds_plot["yh"].to_numpy().astype(NP_REAL) # [km]
            time_indices = ds_plot["time_indices"].to_numpy().astype(NP_INT)
            animation_times = ds_plot["animation_times"].to_numpy().astype(NP_REAL)

        n_t = NP_INT(time_indices.size)

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Read plotting information in {:.2f} s. "
                "cwc shape = {}, cwc size = {:.3f} GiB; "
                "zzh shape = {}, zzh size = {:.3f} GiB. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_start_time,
                    cwc_np.shape,
                    NP_REAL(cwc_np.nbytes) / NP_REAL(1024.**3),
                    zzh.shape,
                    NP_REAL(zzh.nbytes) / NP_REAL(1024.**3),
                    max_rss_mb)
        else:
            msg = ("Read plotting information in {:.2f} s. "
                "cwc shape = {}, cwc size = {:.3f} GiB; "
                "zzh shape = {}, zzh size = {:.3f} GiB.").format(
                    time.perf_counter() - section_start_time,
                    cwc_np.shape,
                    NP_REAL(cwc_np.nbytes) / NP_REAL(1024.**3),
                    zzh.shape,
                    NP_REAL(zzh.nbytes) / NP_REAL(1024.**3))
        print_msg(msg)

    else:
        if recalculate:
            msg = "Recalculating plotting information..."
        else:
            msg = "Saved plotting information unavailable or incomplete. Recalculating..."
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain horizontal sort mask
        #-----------------------------------------------------------------------
        section_sub_start_time: float = time.perf_counter()

        msg = "Obtaining horizontal sort mask..."
        print_msg(msg)

        sort_mask: NP_ARRAY[NP_INT] = get_sort_mask(dp_scream_file).astype(NP_INT)

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Obtained horizontal sort mask in {:.2f} s. "
                "sort_mask size = {}, memory = {:.3f} MiB. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    sort_mask.size,
                    NP_REAL(sort_mask.nbytes) / NP_REAL(1024.**2),
                    max_rss_mb)
        else:
            msg = ("Obtained horizontal sort mask in {:.2f} s. "
                "sort_mask size = {}, memory = {:.3f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    sort_mask.size,
                    NP_REAL(sort_mask.nbytes) / NP_REAL(1024.**2))
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Calculate cloud water content
        #-----------------------------------------------------------------------
        section_sub_start_time = time.perf_counter()

        msg = "Obtaining cloud water content info..."
        print_msg(msg)

        cwc: XR_DATAARRAY = calc_cloud_wc(
            dp_scream_file,
            sort_mask,
            time_indices,
            detailed_calc = detailed_calc) # Cloud water content; [g m^{-3}]; [time, lev, y, x]

        z_dim: str = "lev" if "lev" in cwc.dims else "lay"
        cwc = cwc.transpose("time", z_dim, "y", "x")

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Obtained cloud water content info in {:.2f} s. "
                "cwc dims = {}, cwc shape = {}. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    cwc.dims,
                    cwc.shape,
                    max_rss_mb)
        else:
            msg = ("Obtained cloud water content info in {:.2f} s. "
                "cwc dims = {}, cwc shape = {}.").format(
                    time.perf_counter() - section_sub_start_time,
                    cwc.dims,
                    cwc.shape)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain time-varying vertical interface heights
        #-----------------------------------------------------------------------
        section_sub_start_time = time.perf_counter()

        msg = "Obtaining time-varying vertical interface heights..."
        print_msg(msg)

        xr_dp_scream: XR_DATASET
        with xr.open_dataset(dp_scream_file, engine = "netcdf4",
            decode_times = False, decode_timedelta = False) as xr_dp_scream:

            if ("y" in xr_dp_scream) and ("x" in xr_dp_scream):
                y_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["y"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
                x_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["x"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
            elif ("lat" in xr_dp_scream) and ("lon" in xr_dp_scream):
                y_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["lat"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
                x_coord: NP_ARRAY[NP_REAL] = xr_dp_scream["lon"].isel(ncol = sort_mask).to_numpy().astype(NP_REAL)
            else:
                raise ValueError("DP-SCREAM file must contain either x/y or lon/lat fields on ncol.")

            z_int: XR_DATAARRAY = (xr_dp_scream["z_int"]
                .isel(time = time_indices, ncol = sort_mask)
                .assign_coords({
                    "y" : ("ncol", y_coord),
                    "x" : ("ncol", x_coord)
                })
                .set_index(ncol = ["y", "x"])
                .unstack("ncol")
                .transpose("time", "ilev", "y", "x")
                .load()) # [time, ilev, y, x]; [m]

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Obtained time-varying vertical interface heights in {:.2f} s. "
                "z_int dims = {}, z_int shape = {}. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    z_int.dims,
                    z_int.shape,
                    max_rss_mb)
        else:
            msg = ("Obtained time-varying vertical interface heights in {:.2f} s. "
                "z_int dims = {}, z_int shape = {}.").format(
                    time.perf_counter() - section_sub_start_time,
                    z_int.dims,
                    z_int.shape)
        print_msg(msg)

        # If vertical interfaces are ordered top-to-bottom, reverse them so z
        # increases with vertical index. Reverse CWC consistently.
        z_diff_mean: NP_REAL = NP_REAL((z_int.diff("ilev")).mean().to_numpy())
        if z_diff_mean < 0:
            msg = "Vertical interfaces are ordered top-to-bottom. Reversing vertical dimension..."
            print_msg(msg)
            z_int = z_int.isel(ilev = slice(None, None, -1))
            cwc = cwc.isel({z_dim : slice(None, None, -1)})

        # Make sure CWC layers and z_int interfaces are compatible
        n_z_cwc: NP_INT = NP_INT(cwc.sizes[z_dim])
        n_z_int: NP_INT = NP_INT(z_int.sizes["ilev"] - 1)
        if n_z_cwc != n_z_int:
            msg = ("CWC vertical layer count and z_int interface count do not match. "
                "Trimming to common vertical extent...")
            print_msg(msg)

            n_z_common: NP_INT = NP_INT(np.min([n_z_cwc, n_z_int]))
            cwc = cwc.isel({z_dim : slice(0, n_z_common)})
            z_int = z_int.isel(ilev = slice(0, n_z_common + 1))

        # Trim vertical extent if requested. Because z_int varies in time and
        # space, retain every layer whose lower interface is below z_max anywhere.
        if z_max is not None:
            section_sub_start_time = time.perf_counter()

            msg = "Trimming vertical extent to z_max = {:.3f} km...".format(z_max)
            print_msg(msg)

            z_int_np_tmp: NP_ARRAY[NP_REAL] = z_int.to_numpy().astype(NP_REAL) * NP_REAL(1.e-3) # [km]
            layer_keep: NP_ARRAY[NP_INT] = np.where(
                np.nanmin(z_int_np_tmp[:,0:-1,...], axis = (0, 2, 3)) <= z_max)[0].astype(NP_INT)

            if layer_keep.size > 0:
                n_z_keep: NP_INT = NP_INT(layer_keep.max() + 1)
                cwc = cwc.isel({z_dim : slice(0, n_z_keep)})
                z_int = z_int.isel(ilev = slice(0, n_z_keep + 1))

            del z_int_np_tmp

            if resource is not None:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == "darwin":
                    max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
                else:
                    max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
                msg = ("Trimmed vertical extent in {:.2f} s. "
                    "Retained {} CWC layer(s). Max RSS: {:.1f} MiB.").format(
                        time.perf_counter() - section_sub_start_time,
                        cwc.sizes[z_dim],
                        max_rss_mb)
            else:
                msg = ("Trimmed vertical extent in {:.2f} s. "
                    "Retained {} CWC layer(s).").format(
                        time.perf_counter() - section_sub_start_time,
                        cwc.sizes[z_dim])
            print_msg(msg)

        #-----------------------------------------------------------------------
        # Apply horizontal plotting stride
        #-----------------------------------------------------------------------
        section_sub_start_time = time.perf_counter()

        msg = "Applying horizontal plotting stride..."
        print_msg(msg)

        cwc = cwc.isel(
            y = slice(None, None, xy_stride),
            x = slice(None, None, xy_stride))

        z_int = z_int.isel(
            y = slice(None, None, xy_stride),
            x = slice(None, None, xy_stride))

        # Get horizontal grid-center information from cloud water content
        x = cwc["x"].to_numpy().astype(NP_REAL) * NP_REAL(1.e-3) # [m] => [km]
        y = cwc["y"].to_numpy().astype(NP_REAL) * NP_REAL(1.e-3) # [m] => [km]

        n_x: NP_INT = NP_INT(x.size)
        n_y: NP_INT = NP_INT(y.size)
        n_z: NP_INT = NP_INT(cwc.sizes[z_dim])
        n_zh: NP_INT = NP_INT(n_z + 1)

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Applied horizontal plotting stride in {:.2f} s. "
                "Plotting grid dimensions: n_t = {}, n_z = {}, n_y = {}, n_x = {}. "
                "Candidate voxels per frame = {}, total candidate voxels = {}. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    n_t,
                    n_z,
                    n_y,
                    n_x,
                    NP_INT(n_z * n_y * n_x),
                    NP_INT(n_t * n_z * n_y * n_x),
                    max_rss_mb)
        else:
            msg = ("Applied horizontal plotting stride in {:.2f} s. "
                "Plotting grid dimensions: n_t = {}, n_z = {}, n_y = {}, n_x = {}. "
                "Candidate voxels per frame = {}, total candidate voxels = {}.").format(
                    time.perf_counter() - section_sub_start_time,
                    n_t,
                    n_z,
                    n_y,
                    n_x,
                    NP_INT(n_z * n_y * n_x),
                    NP_INT(n_t * n_z * n_y * n_x))
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Construct horizontal grid corners from horizontal grid positions.
        # Use the coordinate extents as the outer plotting bounds so the rendered
        # domain width and height match the actual simulation width and height.
        #-----------------------------------------------------------------------
        section_sub_start_time = time.perf_counter()

        msg = "Constructing plotting grid corners..."
        print_msg(msg)

        xh = np.zeros([n_x + 1], dtype = NP_REAL)
        yh = np.zeros([n_y + 1], dtype = NP_REAL)

        if n_x > 1:
            xh[1:-1] = NP_REAL(0.5) * (x[0:-1] + x[1:])
            xh[0] = x[0]
            xh[-1] = x[-1]
        else:
            xh[0] = x[0]
            xh[1] = x[0]

        if n_y > 1:
            yh[1:-1] = NP_REAL(0.5) * (y[0:-1] + y[1:])
            yh[0] = y[0]
            yh[-1] = y[-1]
        else:
            yh[0] = y[0]
            yh[1] = y[0]

        # Convert z_int from [time, ilev, y, x] to [time, x, y, ilev], [km]
        z_col: NP_ARRAY[NP_REAL] = (
            np.transpose(z_int.to_numpy(), axes = (0, 3, 2, 1)).astype(NP_REAL)
            * NP_REAL(1.e-3))

        msg = ("Column-interface height array z_col shape = {}, "
            "size = {:.3f} GiB.").format(
                z_col.shape,
                NP_REAL(z_col.nbytes) / NP_REAL(1024.**3))
        print_msg(msg)

        # Interpolate column-interface heights to horizontal voxel corners by
        # averaging neighboring column-center interface heights.
        z_pad: NP_ARRAY[NP_REAL] = np.pad(
            z_col,
            pad_width = ((0, 0), (1, 1), (1, 1), (0, 0)),
            mode = "edge")

        msg = ("Padded height array z_pad shape = {}, "
            "size = {:.3f} GiB.").format(
                z_pad.shape,
                NP_REAL(z_pad.nbytes) / NP_REAL(1024.**3))
        print_msg(msg)

        zzh = NP_REAL(0.25) * (
            z_pad[:,0:n_x + 1,0:n_y + 1,:]
            + z_pad[:,1:n_x + 2,0:n_y + 1,:]
            + z_pad[:,0:n_x + 1,1:n_y + 2,:]
            + z_pad[:,1:n_x + 2,1:n_y + 2,:]) # [time, xh, yh, zh]

        del z_col
        del z_pad

        cwc_np = cwc.to_numpy().astype(NP_REAL) # [time, z, y, x]

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Constructed plotting grid corners in {:.2f} s. "
                "cwc_np shape = {}, cwc_np size = {:.3f} GiB; "
                "zzh shape = {}, zzh size = {:.3f} GiB. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    cwc_np.shape,
                    NP_REAL(cwc_np.nbytes) / NP_REAL(1024.**3),
                    zzh.shape,
                    NP_REAL(zzh.nbytes) / NP_REAL(1024.**3),
                    max_rss_mb)
        else:
            msg = ("Constructed plotting grid corners in {:.2f} s. "
                "cwc_np shape = {}, cwc_np size = {:.3f} GiB; "
                "zzh shape = {}, zzh size = {:.3f} GiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    cwc_np.shape,
                    NP_REAL(cwc_np.nbytes) / NP_REAL(1024.**3),
                    zzh.shape,
                    NP_REAL(zzh.nbytes) / NP_REAL(1024.**3))
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Save plotting information
        #-----------------------------------------------------------------------
        section_sub_start_time = time.perf_counter()

        msg = "Saving plotting information to file..."
        print_msg(msg)

        ds_save: XR_DATASET = xr.Dataset(
            data_vars = {
                "cwc" : (
                    ["time", "z", "y", "x"],
                    cwc_np,
                    {
                        "long_name" : "cloud water content",
                        "units" : "g m^{-3}"
                    }),
                "zzh" : (
                    ["time", "xh", "yh", "zh"],
                    zzh,
                    {
                        "long_name" : "voxel vertical corner heights",
                        "units" : "km"
                    }),
                "time_indices" : (
                    ["time"],
                    time_indices,
                    {
                        "long_name" : "DP-SCREAM time indices"
                    }),
                "animation_times" : (
                    ["time"],
                    animation_times,
                    {
                        "long_name" : "time since simulation start",
                        "units" : "h"
                    })
            },
            coords = {
                "time" : np.arange(0, n_t, dtype = NP_INT),
                "z" : np.arange(0, n_z, dtype = NP_INT),
                "zh" : np.arange(0, n_zh, dtype = NP_INT),
                "x" : (
                    ["x"],
                    x,
                    {
                        "long_name" : "x grid centers",
                        "units" : "km"
                    }),
                "y" : (
                    ["y"],
                    y,
                    {
                        "long_name" : "y grid centers",
                        "units" : "km"
                    }),
                "xh" : (
                    ["xh"],
                    xh,
                    {
                        "long_name" : "x grid corners",
                        "units" : "km"
                    }),
                "yh" : (
                    ["yh"],
                    yh,
                    {
                        "long_name" : "y grid corners",
                        "units" : "km"
                    })
            },
            attrs = {
                "dp_scream_file" : dp_scream_file,
                "xy_stride" : int(xy_stride),
                "z_max" : float(z_max_attr),
                "frames" : int(frames_attr),
                "detailed_calc" : int(detailed_calc),
                "time_units" : time_units,
                "horizontal_bounds_method" : horizontal_bounds_method,
                "description" : "Plotting information for DP-SCREAM atmospheric diorama animation."
            })

        ds_save.to_netcdf(nc_filepath)

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Saved plotting information in {:.2f} s. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_sub_start_time,
                    max_rss_mb)
        else:
            msg = "Saved plotting information in {:.2f} s.".format(
                time.perf_counter() - section_sub_start_time)
        print_msg(msg)

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Recalculated plotting information in {:.2f} s. "
                "Max RSS: {:.1f} MiB.").format(
                    time.perf_counter() - section_start_time,
                    max_rss_mb)
        else:
            msg = "Recalculated plotting information in {:.2f} s.".format(
                time.perf_counter() - section_start_time)
        print_msg(msg)

    #---------------------------------------------------------------------------
    # Set useful dimensions
    #---------------------------------------------------------------------------
    n_t = NP_INT(cwc_np.shape[0])
    n_z = NP_INT(cwc_np.shape[1])
    n_y = NP_INT(cwc_np.shape[2])
    n_x = NP_INT(cwc_np.shape[3])
    n_zh = NP_INT(n_z + 1)

    msg = ("Active plotting dimensions: n_t = {}, n_z = {}, n_y = {}, n_x = {}, "
        "n_zh = {}. Candidate voxels per frame = {}; total candidate voxels = {}.").format(
            n_t,
            n_z,
            n_y,
            n_x,
            n_zh,
            NP_INT(n_z * n_y * n_x),
            NP_INT(n_t * n_z * n_y * n_x))
    print_msg(msg)

    msg = ("Primary array memory: cwc_np = {:.3f} GiB, zzh = {:.3f} GiB.").format(
        NP_REAL(cwc_np.nbytes) / NP_REAL(1024.**3),
        NP_REAL(zzh.nbytes) / NP_REAL(1024.**3))
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Calculate the filled voxels and set up color mapping
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Calculating voxel masks and color mapping information..."
    print_msg(msg)

    estimated_filled_gib: NP_REAL = (
        NP_REAL(n_t * n_z * n_y * n_x * np.dtype(NP_BOOL).itemsize)
        / NP_REAL(1024.**3))
    estimated_cwc_norm_gib: NP_REAL = (
        NP_REAL(n_t * n_z * n_y * n_x * np.dtype(NP_REAL).itemsize)
        / NP_REAL(1024.**3))
    estimated_facecolors_gib: NP_REAL = (
        NP_REAL(n_t * n_z * n_y * n_x * 4 * np.dtype(NP_REAL).itemsize)
        / NP_REAL(1024.**3))

    msg = ("Estimated dense voxel-color memory if fully allocated: "
        "filled = {:.3f} GiB, cwc_norm = {:.3f} GiB, facecolors = {:.3f} GiB. "
        "This script computes RGBA colors per rendered frame to avoid allocating "
        "the dense facecolors array.").format(
            estimated_filled_gib,
            estimated_cwc_norm_gib,
            estimated_facecolors_gib)
    print_msg(msg)

    cwc_np = cwc_np.astype(NP_REAL, copy = False)
    cwc_np[cwc_np < cwc_tol] = NP_REAL(0.)

    max_cwc: NP_REAL = NP_REAL(np.nanmax(cwc_np))
    min_cwc: NP_REAL = cwc_tol

    if max_cwc <= cwc_tol:
        max_cwc = NP_REAL(10.) * cwc_tol

    msg = ("Cloud water content plotting range: min_cwc = {:.4e} g m^{{-3}}, "
        "max_cwc = {:.4e} g m^{{-3}}.").format(
            min_cwc,
            max_cwc)
    print_msg(msg)

    cwc_colormap: MPL_LINEAR_SEGMENTED_COLORMAP = mpl.colormaps[cloud_cmap]
    cwc_colormap_norm: MPL_LOGNORM = colors.LogNorm(vmin = min_cwc, vmax = max_cwc)

    filled: NP_ARRAY[NP_BOOL] = (cwc_np >= cwc_tol).astype(NP_BOOL) # [time, z, y, x]

    # If requested, only keep the voxels with the largest CWC values in each frame.
    if voxels is not None:
        msg = "Selecting voxels with maximal cloud water content..."
        print_msg(msg)

        filled_top: NP_ARRAY[NP_BOOL] = np.zeros(filled.shape, dtype = NP_BOOL)

        frame: int
        for frame in range(0, n_t):
            frame_start_time: float = time.perf_counter()

            cwc_frame: NP_ARRAY[NP_REAL] = cwc_np[frame,...].ravel()
            valid_indices: NP_ARRAY[NP_INT] = np.where(cwc_frame >= cwc_tol)[0].astype(NP_INT)
            n_valid: NP_INT = NP_INT(valid_indices.size)

            n_voxels_frame: NP_INT = NP_INT(0)
            if n_valid > 0:
                n_voxels_frame = NP_INT(np.min([voxels, n_valid]))
                valid_values: NP_ARRAY[NP_REAL] = cwc_frame[valid_indices]

                if n_voxels_frame < n_valid:
                    top_local_indices: NP_ARRAY[NP_INT] = np.argpartition(
                        valid_values, -n_voxels_frame)[-n_voxels_frame:].astype(NP_INT)
                    top_indices: NP_ARRAY[NP_INT] = valid_indices[top_local_indices].astype(NP_INT)
                else:
                    top_indices = valid_indices

                filled_frame: NP_ARRAY[NP_BOOL] = np.zeros(cwc_frame.shape, dtype = NP_BOOL)
                filled_frame[top_indices] = True
                filled_top[frame,...] = filled_frame.reshape(filled.shape[1:])

            if resource is not None:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == "darwin":
                    max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
                else:
                    max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
                msg = ("Frame {}/{} voxel selection: valid voxels = {}, "
                    "selected voxels = {}, elapsed = {:.2f} s, "
                    "Max RSS: {:.1f} MiB.").format(
                        frame + 1,
                        n_t,
                        n_valid,
                        n_voxels_frame,
                        time.perf_counter() - frame_start_time,
                        max_rss_mb)
            else:
                msg = ("Frame {}/{} voxel selection: valid voxels = {}, "
                    "selected voxels = {}, elapsed = {:.2f} s.").format(
                        frame + 1,
                        n_t,
                        n_valid,
                        n_voxels_frame,
                        time.perf_counter() - frame_start_time)
            print_msg(msg)

        filled = filled_top

    filled_counts: NP_ARRAY[NP_INT] = np.count_nonzero(
        filled,
        axis = (1, 2, 3)).astype(NP_INT)

    total_filled_voxels: NP_INT = NP_INT(np.sum(filled_counts))
    max_filled_voxels: NP_INT = NP_INT(np.max(filled_counts)) if filled_counts.size > 0 else NP_INT(0)
    min_filled_voxels: NP_INT = NP_INT(np.min(filled_counts)) if filled_counts.size > 0 else NP_INT(0)
    mean_filled_voxels: NP_REAL = NP_REAL(np.mean(filled_counts)) if filled_counts.size > 0 else NP_REAL(0.)

    msg = ("Voxel count summary after tolerance/selection: min = {}, mean = {:.1f}, "
        "max = {}, total rendered voxel-frame instances = {}.").format(
            min_filled_voxels,
            mean_filled_voxels,
            max_filled_voxels,
            total_filled_voxels)
    print_msg(msg)

    frame: int
    for frame in range(0, n_t):
        msg = "Frame {}/{} voxel count: {}.".format(
            frame + 1,
            n_t,
            filled_counts[frame])
        print_msg(msg)

    # Transpose to necessary shape for Poly3DCollection indexing
    filled = np.transpose(filled, axes = (0, 3, 2, 1)) # [time, x, y, z]

    # Set alpha
    alpha_min: NP_REAL = NP_REAL(0.1)
    alpha_max: NP_REAL = NP_REAL(0.5)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Calculated voxel masks and color mapping information in {:.2f} s. "
            "filled shape = {}, filled size = {:.3f} GiB. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                filled.shape,
                NP_REAL(filled.nbytes) / NP_REAL(1024.**3),
                max_rss_mb)
    else:
        msg = ("Calculated voxel masks and color mapping information in {:.2f} s. "
            "filled shape = {}, filled size = {:.3f} GiB.").format(
                time.perf_counter() - section_start_time,
                filled.shape,
                NP_REAL(filled.nbytes) / NP_REAL(1024.**3))
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Set up the figure
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Setting up figure..."
    print_msg(msg)

    nrows: NP_INT = NP_INT(1)
    ncols: NP_INT = NP_INT(1)

    fig_height: NP_REAL = NP_REAL(4.5)
    fig_width: NP_REAL = NP_REAL(6.5)

    subplot_top_in: NP_REAL = NP_REAL(0.24)
    subplot_top: NP_REAL = NP_REAL((fig_height - subplot_top_in) / fig_height)

    cbar_bottom_in: NP_REAL = NP_REAL(0.50)
    cbar_top_in: NP_REAL = NP_REAL(0.60)
    cbar_bottom: NP_REAL = NP_REAL(cbar_bottom_in / fig_height)
    cbar_height: NP_REAL = NP_REAL((fig_height - cbar_bottom_in - cbar_top_in) / fig_height)

    fig: MPL_FIGURE
    axs: NP_ARRAY[MPL_AXES]
    fig, axs = plt.subplots(
        nrows = nrows, ncols = ncols,
        sharex = False, sharey = False,
        constrained_layout = False,
        figsize = (fig_width, fig_height),
        subplot_kw = {"projection" : "3d"})
        
    if (ncols == 1) and (nrows == 1):
        axs = np.array([[axs]])
    elif ncols == 1:
        axs = axs[...,None]
    elif nrows == 1:
        axs = axs[None,...]

    fig.subplots_adjust(
        left = 0.04,
        right = 0.78,
        bottom = 0.02,
        top = subplot_top,
        wspace = 0.04,
        hspace = -0.16
    )

    cax = fig.add_axes([0.90, cbar_bottom, 0.02, cbar_height])

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Set up figure in {:.2f} s. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                max_rss_mb)
    else:
        msg = "Set up figure in {:.2f} s.".format(
            time.perf_counter() - section_start_time)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Plot the initial data
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Plotting the initial data..."
    print_msg(msg)

    x0: NP_ARRAY[NP_REAL] = xh[:-1]
    x1: NP_ARRAY[NP_REAL] = xh[1:]
    y0: NP_ARRAY[NP_REAL] = yh[:-1]
    y1: NP_ARRAY[NP_REAL] = yh[1:]

    filled_i: NP_ARRAY[NP_BOOL] = filled[0,...]

    poly_artist: Optional[Poly3DCollection] = None
    time_artist = axs[0,0].text2D(
        0.5, 0.86,
        r"{:.2f} $h$".format(animation_times[0]),
        transform = axs[0,0].transAxes,
        ha = "center",
        va = "top"
    )

    msg = "Initial frame voxel count: {}.".format(filled_counts[0])
    print_msg(msg)

    if np.any(filled_i):
        i_x: NP_ARRAY[NP_INT]
        i_y: NP_ARRAY[NP_INT]
        i_z: NP_ARRAY[NP_INT]
        i_x, i_y, i_z = np.where(filled_i)

        n_vox: int = int(i_x.size)

        xv0: NP_ARRAY[NP_REAL] = x0[i_x]
        xv1: NP_ARRAY[NP_REAL] = x1[i_x]
        yv0: NP_ARRAY[NP_REAL] = y0[i_y]
        yv1: NP_ARRAY[NP_REAL] = y1[i_y]

        z000: NP_ARRAY[NP_REAL] = zzh[0, i_x,     i_y,     i_z    ]
        z100: NP_ARRAY[NP_REAL] = zzh[0, i_x + 1, i_y,     i_z    ]
        z110: NP_ARRAY[NP_REAL] = zzh[0, i_x + 1, i_y + 1, i_z    ]
        z010: NP_ARRAY[NP_REAL] = zzh[0, i_x,     i_y + 1, i_z    ]

        z001: NP_ARRAY[NP_REAL] = zzh[0, i_x,     i_y,     i_z + 1]
        z101: NP_ARRAY[NP_REAL] = zzh[0, i_x + 1, i_y,     i_z + 1]
        z111: NP_ARRAY[NP_REAL] = zzh[0, i_x + 1, i_y + 1, i_z + 1]
        z011: NP_ARRAY[NP_REAL] = zzh[0, i_x,     i_y + 1, i_z + 1]

        verts: NP_ARRAY[NP_REAL] = np.empty((6 * n_vox, 4, 3), dtype = NP_REAL)

        # Bottom face
        verts[0::6,0,:] = np.stack((xv0, yv0, z000), axis = 1)
        verts[0::6,1,:] = np.stack((xv1, yv0, z100), axis = 1)
        verts[0::6,2,:] = np.stack((xv1, yv1, z110), axis = 1)
        verts[0::6,3,:] = np.stack((xv0, yv1, z010), axis = 1)

        # Top face
        verts[1::6,0,:] = np.stack((xv0, yv0, z001), axis = 1)
        verts[1::6,1,:] = np.stack((xv1, yv0, z101), axis = 1)
        verts[1::6,2,:] = np.stack((xv1, yv1, z111), axis = 1)
        verts[1::6,3,:] = np.stack((xv0, yv1, z011), axis = 1)

        # y-min face
        verts[2::6,0,:] = np.stack((xv0, yv0, z000), axis = 1)
        verts[2::6,1,:] = np.stack((xv1, yv0, z100), axis = 1)
        verts[2::6,2,:] = np.stack((xv1, yv0, z101), axis = 1)
        verts[2::6,3,:] = np.stack((xv0, yv0, z001), axis = 1)

        # y-max face
        verts[3::6,0,:] = np.stack((xv0, yv1, z010), axis = 1)
        verts[3::6,1,:] = np.stack((xv1, yv1, z110), axis = 1)
        verts[3::6,2,:] = np.stack((xv1, yv1, z111), axis = 1)
        verts[3::6,3,:] = np.stack((xv0, yv1, z011), axis = 1)

        # x-min face
        verts[4::6,0,:] = np.stack((xv0, yv0, z000), axis = 1)
        verts[4::6,1,:] = np.stack((xv0, yv1, z010), axis = 1)
        verts[4::6,2,:] = np.stack((xv0, yv1, z011), axis = 1)
        verts[4::6,3,:] = np.stack((xv0, yv0, z001), axis = 1)

        # x-max face
        verts[5::6,0,:] = np.stack((xv1, yv0, z100), axis = 1)
        verts[5::6,1,:] = np.stack((xv1, yv1, z110), axis = 1)
        verts[5::6,2,:] = np.stack((xv1, yv1, z111), axis = 1)
        verts[5::6,3,:] = np.stack((xv1, yv0, z101), axis = 1)

        cwc_values: NP_ARRAY[NP_REAL] = cwc_np[0, i_z, i_y, i_x]
        voxel_facecolors: NP_ARRAY[NP_REAL] = cwc_colormap(
            cwc_colormap_norm(np.maximum(cwc_values, cwc_tol))).astype(NP_REAL)

        voxel_alpha: NP_ARRAY[NP_REAL] = (
            ((alpha_max - alpha_min) * (cwc_values - cwc_tol) / (max_cwc - cwc_tol))
            + alpha_min).astype(NP_REAL)
        voxel_alpha = np.clip(voxel_alpha, NP_REAL(0.), NP_REAL(1.))
        voxel_facecolors[:,3] = voxel_alpha

        poly_facecolors: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors, 6, axis = 0)

        poly_artist = Poly3DCollection(
            verts,
            facecolors = poly_facecolors,
            edgecolors = poly_facecolors,
            linewidths = 0
        )
        axs[0,0].add_collection3d(poly_artist)

        msg = ("Initial frame geometry: voxels = {}, faces = {}, "
            "verts size = {:.3f} MiB, poly_facecolors size = {:.3f} MiB.").format(
                n_vox,
                6 * n_vox,
                NP_REAL(verts.nbytes) / NP_REAL(1024.**2),
                NP_REAL(poly_facecolors.nbytes) / NP_REAL(1024.**2))
        print_msg(msg)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Plotted initial data in {:.2f} s. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                max_rss_mb)
    else:
        msg = "Plotted initial data in {:.2f} s.".format(
            time.perf_counter() - section_start_time)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Set up colorbar
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Setting up colorbar..."
    print_msg(msg)

    cwc_mappable = mpl.cm.ScalarMappable(
        norm = cwc_colormap_norm,
        cmap = cwc_colormap)

    cwc_colorbar: MPL_COLORBAR = fig.colorbar(
        mappable = cwc_mappable,
        cax = cax)
    cwc_colorbar.ax.set_yscale("log")

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Set up colorbar in {:.2f} s. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                max_rss_mb)
    else:
        msg = "Set up colorbar in {:.2f} s.".format(
            time.perf_counter() - section_start_time)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Set style elements
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Setting style elements..."
    print_msg(msg)

    # Background Panes
    pane_color: list[float] = [0.0, 0.0, 0.0, 0.0]
    ax: MPL_AXES
    for ax in axs.flatten():
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)

    # Set grid linewidth
    ax: MPL_AXES
    for ax in axs.flatten():
        ax.grid(False)
        ax.xaxis._axinfo["grid"]["linewidth"] = 0
        ax.yaxis._axinfo["grid"]["linewidth"] = 0
        ax.zaxis._axinfo["grid"]["linewidth"] = 0
        ax.xaxis._axinfo["grid"]["color"] = (1., 1., 1., 0.)
        ax.yaxis._axinfo["grid"]["color"] = (1., 1., 1., 0.)
        ax.zaxis._axinfo["grid"]["color"] = (1., 1., 1., 0.)

    # Axis limits
    z_plot_min: NP_REAL = NP_REAL(0.)
    z_plot_max: NP_REAL = NP_REAL(z_max) if z_max is not None else NP_REAL(np.nanmax(zzh))
    z_plot_max = NP_REAL(np.ceil(z_plot_max))

    if z_plot_max <= z_plot_min:
        z_plot_max = NP_REAL(1.)

    ax: MPL_AXES
    for ax in axs.flatten():
        ax.set_xlim((xh.min(), xh.max()))
        ax.set_ylim((yh.min(), yh.max()))
        ax.set_zlim((z_plot_min, z_plot_max))

    # Aspect Ratio
    xh_len: NP_REAL = NP_REAL(xh.max() - xh.min())
    yh_len: NP_REAL = NP_REAL(yh.max() - yh.min())
    zh_len: NP_REAL = NP_REAL(z_plot_max - z_plot_min)
    ax: MPL_AXES
    for ax in axs.flatten():
        ax.set_box_aspect([xh_len, yh_len, zh_len])

    # Tick Labels
    x_ticks: NP_ARRAY[NP_REAL] = MaxNLocator(nbins = 4).tick_values(
        xh.min(), xh.max()).astype(NP_REAL)
    y_ticks: NP_ARRAY[NP_REAL] = MaxNLocator(nbins = 4).tick_values(
        yh.min(), yh.max()).astype(NP_REAL)
    z_ticks: NP_ARRAY[NP_REAL] = np.array([z_plot_min, z_plot_max], dtype = NP_REAL)

    ax: MPL_AXES
    for ax in axs.flatten():
        ax.xaxis.set_ticks(x_ticks)
        ax.yaxis.set_ticks(y_ticks)
        ax.zaxis.set_ticks(z_ticks)

    # Axis labels
    axs[-1,-1].set_ylabel(r"y $\left[ km \right]$")
    for ax in (axs[:,-1]).flatten():
        ax.set_zlabel(r"z $\left[ km \right]$")
    for ax in (axs[-1,:]).flatten():
        ax.set_xlabel(r"x $\left[ km \right]$")

    # Suplabels
    fig.suptitle(r"Cloud Water Content $\left[ g\,m^{-3} \right]$" + " - Native Resolution")
    fig.supxlabel(" ") # Add for padding at bottom

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Set style elements in {:.2f} s. "
            "Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                max_rss_mb)
    else:
        msg = "Set style elements in {:.2f} s.".format(
            time.perf_counter() - section_start_time)
    print_msg(msg)

    #---------------------------------------------------------------------------
    # Animate the data
    #---------------------------------------------------------------------------
    def update(frame):
        nonlocal poly_artist

        frame_start_time: float = time.perf_counter()

        msg = ("Rendering frame {}/{}: {} voxel(s), {} face(s).").format(
            frame + 1,
            n_t,
            filled_counts[frame],
            NP_INT(6 * filled_counts[frame]))
        print_msg(msg)

        if poly_artist is not None:
            poly_artist.remove()
            poly_artist = None

        filled_i: NP_ARRAY[NP_BOOL] = filled[frame,...]

        if np.any(filled_i):
            i_x: NP_ARRAY[NP_INT]
            i_y: NP_ARRAY[NP_INT]
            i_z: NP_ARRAY[NP_INT]
            i_x, i_y, i_z = np.where(filled_i)

            n_vox: int = int(i_x.size)

            xv0: NP_ARRAY[NP_REAL] = x0[i_x]
            xv1: NP_ARRAY[NP_REAL] = x1[i_x]
            yv0: NP_ARRAY[NP_REAL] = y0[i_y]
            yv1: NP_ARRAY[NP_REAL] = y1[i_y]

            z000: NP_ARRAY[NP_REAL] = zzh[frame, i_x,     i_y,     i_z    ]
            z100: NP_ARRAY[NP_REAL] = zzh[frame, i_x + 1, i_y,     i_z    ]
            z110: NP_ARRAY[NP_REAL] = zzh[frame, i_x + 1, i_y + 1, i_z    ]
            z010: NP_ARRAY[NP_REAL] = zzh[frame, i_x,     i_y + 1, i_z    ]

            z001: NP_ARRAY[NP_REAL] = zzh[frame, i_x,     i_y,     i_z + 1]
            z101: NP_ARRAY[NP_REAL] = zzh[frame, i_x + 1, i_y,     i_z + 1]
            z111: NP_ARRAY[NP_REAL] = zzh[frame, i_x + 1, i_y + 1, i_z + 1]
            z011: NP_ARRAY[NP_REAL] = zzh[frame, i_x,     i_y + 1, i_z + 1]

            verts: NP_ARRAY[NP_REAL] = np.empty((6 * n_vox, 4, 3), dtype = NP_REAL)

            # Bottom face
            verts[0::6,0,:] = np.stack((xv0, yv0, z000), axis = 1)
            verts[0::6,1,:] = np.stack((xv1, yv0, z100), axis = 1)
            verts[0::6,2,:] = np.stack((xv1, yv1, z110), axis = 1)
            verts[0::6,3,:] = np.stack((xv0, yv1, z010), axis = 1)

            # Top face
            verts[1::6,0,:] = np.stack((xv0, yv0, z001), axis = 1)
            verts[1::6,1,:] = np.stack((xv1, yv0, z101), axis = 1)
            verts[1::6,2,:] = np.stack((xv1, yv1, z111), axis = 1)
            verts[1::6,3,:] = np.stack((xv0, yv1, z011), axis = 1)

            # y-min face
            verts[2::6,0,:] = np.stack((xv0, yv0, z000), axis = 1)
            verts[2::6,1,:] = np.stack((xv1, yv0, z100), axis = 1)
            verts[2::6,2,:] = np.stack((xv1, yv0, z101), axis = 1)
            verts[2::6,3,:] = np.stack((xv0, yv0, z001), axis = 1)

            # y-max face
            verts[3::6,0,:] = np.stack((xv0, yv1, z010), axis = 1)
            verts[3::6,1,:] = np.stack((xv1, yv1, z110), axis = 1)
            verts[3::6,2,:] = np.stack((xv1, yv1, z111), axis = 1)
            verts[3::6,3,:] = np.stack((xv0, yv1, z011), axis = 1)

            # x-min face
            verts[4::6,0,:] = np.stack((xv0, yv0, z000), axis = 1)
            verts[4::6,1,:] = np.stack((xv0, yv1, z010), axis = 1)
            verts[4::6,2,:] = np.stack((xv0, yv1, z011), axis = 1)
            verts[4::6,3,:] = np.stack((xv0, yv0, z001), axis = 1)

            # x-max face
            verts[5::6,0,:] = np.stack((xv1, yv0, z100), axis = 1)
            verts[5::6,1,:] = np.stack((xv1, yv1, z110), axis = 1)
            verts[5::6,2,:] = np.stack((xv1, yv1, z111), axis = 1)
            verts[5::6,3,:] = np.stack((xv1, yv0, z101), axis = 1)

            cwc_values: NP_ARRAY[NP_REAL] = cwc_np[frame, i_z, i_y, i_x]
            voxel_facecolors: NP_ARRAY[NP_REAL] = cwc_colormap(
                cwc_colormap_norm(np.maximum(cwc_values, cwc_tol))).astype(NP_REAL)

            voxel_alpha: NP_ARRAY[NP_REAL] = (
                ((alpha_max - alpha_min) * (cwc_values - cwc_tol) / (max_cwc - cwc_tol))
                + alpha_min).astype(NP_REAL)
            voxel_alpha = np.clip(voxel_alpha, NP_REAL(0.), NP_REAL(1.))
            voxel_facecolors[:,3] = voxel_alpha

            poly_facecolors: NP_ARRAY[NP_REAL] = np.repeat(voxel_facecolors, 6, axis = 0)

            poly_artist = Poly3DCollection(
                verts,
                facecolors = poly_facecolors,
                edgecolors = poly_facecolors,
                linewidths = 0
            )
            axs[0,0].add_collection3d(poly_artist)

            msg = ("Frame {}/{} geometry memory: verts = {:.3f} MiB, "
                "poly_facecolors = {:.3f} MiB.").format(
                    frame + 1,
                    n_t,
                    NP_REAL(verts.nbytes) / NP_REAL(1024.**2),
                    NP_REAL(poly_facecolors.nbytes) / NP_REAL(1024.**2))
            print_msg(msg)

        # Update time label
        time_artist.set_text(r"{:.2f} $h$".format(animation_times[frame]))

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
            else:
                max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
            msg = ("Rendered frame {}/{} in {:.2f} s. "
                "Max RSS: {:.1f} MiB.").format(
                    frame + 1,
                    n_t,
                    time.perf_counter() - frame_start_time,
                    max_rss_mb)
        else:
            msg = "Rendered frame {}/{} in {:.2f} s.".format(
                frame + 1,
                n_t,
                time.perf_counter() - frame_start_time)
        print_msg(msg)

    n_frames: NP_INT = NP_INT(n_t)
    interval: NP_REAL = NP_REAL(1.e3) / NP_REAL(fps)

    msg = ("Creating animation object with {} frame(s), fps = {}, "
        "interval = {:.2f} ms.").format(
            n_frames,
            fps,
            interval)
    print_msg(msg)

    ani = animation.FuncAnimation(
        fig = fig,
        func = update,
        frames = n_frames,
        interval = interval)

    #---------------------------------------------------------------------------
    # Save the animation to file
    #---------------------------------------------------------------------------
    section_start_time = time.perf_counter()

    msg = "Saving animation to file..."
    print_msg(msg)

    writer = animation.FFMpegWriter(fps = fps, bitrate = bitrate)

    ani_filename: str = "dp_screamxx_diorama_atm.mp4"
    ani_filepath: str = os.path.join(rad_tran_vizdir, ani_filename)

    msg = "Animation output path: {}".format(ani_filepath)
    print_msg(msg)

    ani.save(
        filename = ani_filepath,
        writer = writer,
        dpi = 256
    )
    plt.close(fig)

    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024. * 1024.)
        else:
            max_rss_mb = NP_REAL(usage.ru_maxrss) / NP_REAL(1024.)
        msg = ("Saved animation to file in {:.2f} s. "
            "Total runtime = {:.2f} s. Max RSS: {:.1f} MiB.").format(
                time.perf_counter() - section_start_time,
                time.perf_counter() - script_start_time,
                max_rss_mb)
    else:
        msg = ("Saved animation to file in {:.2f} s. "
            "Total runtime = {:.2f} s.").format(
                time.perf_counter() - section_start_time,
                time.perf_counter() - script_start_time)
    print_msg(msg)

if __name__ == "__main__":
    main()