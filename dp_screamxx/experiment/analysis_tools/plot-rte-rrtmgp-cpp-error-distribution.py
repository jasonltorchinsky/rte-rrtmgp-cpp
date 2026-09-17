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
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATAARRAY, \
    MPL_FIGURE, MPL_AXES, MPL_PCOLORMESH
from consts.numeric import NP_EPS, NP_SMALL, NP_LARGE
from consts.visual import count_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_daytime_indices, find_szas, find_times, \
    calc_sw_reflectance, calc_sw_heating, calc_sw_flux_sfc_dn, \
    find_grid, calc_z_max_info, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-error-distribution"
prog_desc: str = "Visualize distributions of two-stream and ray-tracer solver differences for RTE-RRTMGP-CPP."

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
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--z-max", nargs = "?", default = 0., type = float,
        help = "Maximum height for calculations [km].")
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
    parser.add_argument("--bins", nargs = "?", default = 10, type = int,
        help = "Number of bins in each histogram.")

    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate
    z_max: Optional[NP_REAL] = NP_REAL(args.z_max) if args.z_max > 0 else None
    bins: NP_INT = NP_INT(args.bins)

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
    # Calculate quantities that should be common across all resolutions
    #---------------------------------------------------------------------------
    msg: str = "Calculating quantities common across all resolutions..."
    print_msg(msg)

    daytime_indices: NP_ARRAY[NP_INT] = find_daytime_indices(
        rad_tran_infiles[0]) # Time indices for each day; [ndays; time_per_day]
    daytime_times: NP_ARRAY[NP_REAL] = find_times(
        rad_tran_infiles[0], 
        daytime_indices) # Time since simulation start; [h]; [ndays, time_per_day]
    daytime_szas: NP_ARRAY[NP_REAL] = find_szas(
        rad_tran_infiles[0], 
        daytime_indices) # Solar zenith angle (SZA); [degrees]; [ndays, time_per_day]
    ndays: NP_INT = NP_INT(daytime_indices.shape[0])
    ntimes_per_day: NP_INT = NP_INT(daytime_times[0,:].size)
    z_max_info: dict = calc_z_max_info(
        rad_tran_infiles[0], 
        z_max = z_max) # 

    #---------------------------------------------------------------------------
    # Calculate relevant quantities at each resolution for each day
    #---------------------------------------------------------------------------
    reflectance_diff: dict = {}
    heating_diff: dict = {}
    flux_sfc_dn_diff: dict = {}

    reflectance_diff_max: NP_REAL = -NP_LARGE
    heating_diff_max: NP_REAL = -NP_LARGE
    flux_sfc_dn_diff_max: NP_REAL = -NP_LARGE

    reflectance_diff_hist: dict = {}
    heating_diff_hist: dict = {}
    flux_sfc_dn_diff_hist: dict = {}

    #---------------------------------------------------------------------------
    # Loop through resolutions - Calculate differences
    #---------------------------------------------------------------------------
    msg: str = "Calculating differences..."
    print_msg(msg)

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(coarse_factor_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Set up dicts for this resolution
        #-----------------------------------------------------------------------
        reflectance_diff[coarse_factor_str] = {}
        heating_diff[coarse_factor_str] = {}
        flux_sfc_dn_diff[coarse_factor_str] = {}
        
        #-----------------------------------------------------------------------
        # Calculate fields for each each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            msg: str = "- Day {}...".format(jj)
            print_msg(msg)
            
            #-------------------------------------------------------------------
            # Calculate reflectance
            #-------------------------------------------------------------------
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

            reflectance_diff[coarse_factor_str][jj] = (
                (reflectance_ts - reflectance_rt)
                .stack(spatial = ("y", "x"))
                .reset_index("spatial")
            )

            reflectance_diff_max = max(
                reflectance_diff_max, 
                NP_REAL(np.max(np.abs(reflectance_diff[coarse_factor_str][jj])))
            )

            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
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

            heating_diff[coarse_factor_str][jj] = (
                (heating_ts - heating_rt)
                .stack(spatial = ("lay", "y", "x"))
                .reset_index("spatial")
            )

            heating_diff_max = max(
                heating_diff_max, 
                NP_REAL(np.max(np.abs(heating_diff[coarse_factor_str][jj])))
            )

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
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

            flux_sfc_dn_diff[coarse_factor_str][jj] = (
                (flux_sfc_dn_ts - flux_sfc_dn_rt)
                .stack(spatial = ("y", "x"))
                .reset_index("spatial")
            )

            flux_sfc_dn_diff_max = max(
                flux_sfc_dn_diff_max, 
                NP_REAL(np.max(np.abs(flux_sfc_dn_diff[coarse_factor_str][jj])))
            )

    #---------------------------------------------------------------------------
    # Calculate bin egdes for the histograms
    #---------------------------------------------------------------------------
    daytime_time_bin_widths: NP_ARRAY[NP_REAL] = daytime_times[:,1] - daytime_times[:,0]
    daytime_time_bin_centers: NP_ARRAY[NP_REAL] = daytime_times[...]
    daytime_time_bin_edges: NP_ARRAY[NP_REAL] = np.zeros(
        (ndays, ntimes_per_day + 1), 
        dtype = NP_REAL
    )
    jj: int
    for jj in range(0, ndays):
        daytime_time_bin_edges[jj] = np.arange(
            daytime_times[jj,0] - daytime_time_bin_widths[jj] / 2., 
            daytime_times[jj,-1] + daytime_time_bin_widths[jj] / 2. + NP_SMALL, 
            step = daytime_time_bin_widths[jj],
            dtype = NP_REAL
        )

    # Set differences that we ignore
    reflectance_eps: NP_REAL = NP_REAL(0.01)
    heating_eps: NP_REAL = NP_REAL(1.0)
    flux_sfc_dn_eps: NP_REAL = NP_REAL(100.0)

    eps_array: NP_ARRAY[NP_REAL] = np.array(
        [reflectance_eps, heating_eps, flux_sfc_dn_eps], 
        dtype = NP_REAL)

    reflectance_diff_pos_bin_edges: NP_ARRAY[NP_REAL] = np.linspace(
        reflectance_eps, 
        reflectance_diff_max, 
        num = NP_INT(NP_REAL(bins) / 2.),
        dtype = NP_REAL
    )
    reflectance_diff_bin_edges: NP_ARRAY[NP_REAL] = np.concatenate(
        (-reflectance_diff_pos_bin_edges[::-1], reflectance_diff_pos_bin_edges)
    )

    heating_diff_pos_bin_edges: NP_ARRAY[NP_REAL] = np.linspace(
        heating_eps, 
        heating_diff_max, 
        num = NP_INT(NP_REAL(bins) / 2.),
        dtype = NP_REAL
    )
    heating_diff_bin_edges: NP_ARRAY[NP_REAL] = np.concatenate(
        (-heating_diff_pos_bin_edges[::-1], heating_diff_pos_bin_edges)
    )

    flux_sfc_dn_diff_pos_bin_edges: NP_ARRAY[NP_REAL] = np.linspace(
        flux_sfc_dn_eps, 
        flux_sfc_dn_diff_max, 
        num = NP_INT(NP_REAL(bins) / 2.),
        dtype = NP_REAL
    )
    flux_sfc_dn_diff_bin_edges: NP_ARRAY[NP_REAL] = np.concatenate(
        (-flux_sfc_dn_diff_pos_bin_edges[::-1], flux_sfc_dn_diff_pos_bin_edges)
    )
            
    #---------------------------------------------------------------------------
    # Loop through resolutions - calculate histograms
    #---------------------------------------------------------------------------
    msg: str = "Calculating histograms..."
    print_msg(msg)

    # Save for colorbar limits
    reflectance_diff_hist_max: NP_REAL = -NP_LARGE
    reflectance_diff_hist_min: NP_REAL = NP_LARGE

    heating_diff_hist_max: NP_REAL = -NP_LARGE
    heating_diff_hist_min: NP_REAL = NP_LARGE

    flux_sfc_dn_diff_hist_max: NP_REAL = -NP_LARGE
    flux_sfc_dn_diff_hist_min: NP_REAL = NP_LARGE

    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(coarse_factor_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Calculate quantities common to histograms
        #-----------------------------------------------------------------------
        nelem_xy: NP_INT = NP_INT(reflectance_diff[coarse_factor_str][0]["spatial"].size)
        nelem_xyz: NP_INT = NP_INT(heating_diff[coarse_factor_str][0]["spatial"].size)

        #-----------------------------------------------------------------------
        # Set up dicts for this resolution
        #-----------------------------------------------------------------------
        reflectance_diff_hist[coarse_factor_str] = {}
        heating_diff_hist[coarse_factor_str] = {}
        flux_sfc_dn_diff_hist[coarse_factor_str] = {}
        
        #-----------------------------------------------------------------------
        # Calculate histograms for each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            msg: str = "- Day {}...".format(jj)
            print_msg(msg)

            #-------------------------------------------------------------------
            # Reflectance
            #-------------------------------------------------------------------
            msg: str = "-- Reflectance..."
            print_msg(msg)

            x: NP_ARRAY[NP_REAL] = np.transpose(
                np.tile(
                    NP_REAL(reflectance_diff[coarse_factor_str][jj]["time"]),
                    (nelem_xy, 1)
                )
            ).flatten()
            y: NP_ARRAY[NP_REAL] = NP_REAL(reflectance_diff[coarse_factor_str][jj]).flatten()

            reflectance_diff_hist[coarse_factor_str][jj] = list(
                np.histogram2d(
                    x,
                    y,
                    bins = (daytime_time_bin_edges[jj,...], reflectance_diff_bin_edges)
                )
            )
            reflectance_diff_hist[coarse_factor_str][jj][0] = \
                NP_REAL(reflectance_diff_hist[coarse_factor_str][jj][0]) / NP_REAL(nelem_xy)
            reflectance_diff_hist[coarse_factor_str][jj][0][:, bins // 2 - 1] = 0.0

            reflectance_diff_hist_max = max(
                reflectance_diff_hist_max, 
                np.max(reflectance_diff_hist[coarse_factor_str][jj][0])
            )
            reflectance_diff_hist_min = min(
                reflectance_diff_hist_min, 
                np.min(reflectance_diff_hist[coarse_factor_str][jj][0])
            )

            reflectance_diff_hist[coarse_factor_str][jj][0][:, bins // 2 - 1] = np.nan
            zero_mask: NP_ARRAY[NP_BOOL] = reflectance_diff_hist[coarse_factor_str][jj][0] < NP_EPS
            reflectance_diff_hist[coarse_factor_str][jj][0][zero_mask] = np.nan

            #-------------------------------------------------------------------
            # Calculate heating rates
            #-------------------------------------------------------------------
            msg: str = "-- Heating ..."
            print_msg(msg)

            x: NP_ARRAY[NP_REAL] = np.transpose(
                np.tile(
                    NP_REAL(heating_diff[coarse_factor_str][jj]["time"]),
                    (nelem_xyz, 1)
                )
            ).flatten()
            y: NP_ARRAY[NP_REAL] = NP_REAL(heating_diff[coarse_factor_str][jj]).flatten()

            heating_diff_hist[coarse_factor_str][jj] = list(
                np.histogram2d(
                    x,
                    y,
                    bins = (daytime_time_bin_edges[jj,...], heating_diff_bin_edges)
                )
            )
            heating_diff_hist[coarse_factor_str][jj][0] = \
                NP_REAL(heating_diff_hist[coarse_factor_str][jj][0]) / NP_REAL(nelem_xyz)
            heating_diff_hist[coarse_factor_str][jj][0][:, bins // 2 - 1] = 0.0

            heating_diff_hist_max = max(
                heating_diff_hist_max, 
                np.max(heating_diff_hist[coarse_factor_str][jj][0])
            )
            heating_diff_hist_min = min(
                heating_diff_hist_min, 
                np.min(heating_diff_hist[coarse_factor_str][jj][0])
            )

            heating_diff_hist[coarse_factor_str][jj][0][:, bins // 2 - 1] = np.nan
            zero_mask: NP_ARRAY[NP_BOOL] = heating_diff_hist[coarse_factor_str][jj][0] < NP_EPS
            heating_diff_hist[coarse_factor_str][jj][0][zero_mask] = np.nan

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
            msg: str = "-- Downwelling Surface Flux..."
            print_msg(msg)

            x: NP_ARRAY[NP_REAL] = np.transpose(
                np.tile(
                    NP_REAL(flux_sfc_dn_diff[coarse_factor_str][jj]["time"]),
                    (nelem_xy, 1)
                )
            ).flatten()
            y: NP_ARRAY[NP_REAL] = NP_REAL(flux_sfc_dn_diff[coarse_factor_str][jj]).flatten()

            flux_sfc_dn_diff_hist[coarse_factor_str][jj] = list(
                np.histogram2d(
                    x,
                    y,
                    bins = (daytime_time_bin_edges[jj,...], flux_sfc_dn_diff_bin_edges)
                )
            )
            flux_sfc_dn_diff_hist[coarse_factor_str][jj][0] = \
                NP_REAL(flux_sfc_dn_diff_hist[coarse_factor_str][jj][0]) / NP_REAL(nelem_xy)
            flux_sfc_dn_diff_hist[coarse_factor_str][jj][0][:, bins // 2 - 1] = 0.0

            flux_sfc_dn_diff_hist_max = max(
                flux_sfc_dn_diff_hist_max, 
                np.max(flux_sfc_dn_diff_hist[coarse_factor_str][jj][0])
            )
            flux_sfc_dn_diff_hist_min = min(
                flux_sfc_dn_diff_hist_min, 
                np.min(flux_sfc_dn_diff_hist[coarse_factor_str][jj][0])
            )

            flux_sfc_dn_diff_hist[coarse_factor_str][jj][0][:, bins // 2 - 1] = np.nan
            zero_mask: NP_ARRAY[NP_BOOL] = flux_sfc_dn_diff_hist[coarse_factor_str][jj][0] < NP_EPS
            flux_sfc_dn_diff_hist[coarse_factor_str][jj][0][zero_mask] = np.nan

    #---------------------------------------------------------------------------
    # Set up figure for plotting
    #---------------------------------------------------------------------------
    ii: int
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        coarse_factor_str: str = lr_re.search(rad_tran_infile).group()

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)
        grid: dict = find_grid(rad_tran_infile)

        msg: str = "Setting up figure for {}...".format(coarse_factor_str)
        print_msg(msg)

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

        #-------------------------------------------------------------------
        # Plot histograms across each day
        #-------------------------------------------------------------------
        msg: str = "Plotting histograms..."
        print_msg(msg)

        jj: int
        for jj in range(0, ndays):
            msg: str = "- Day {}...".format(jj)
            print_msg(msg)

            # Row 0 - Reflectance
            row: NP_INT = NP_INT(0)
            reflectance_diff_pcm: MPL_PCOLORMESH = axs[row,jj].pcolormesh(
                reflectance_diff_hist[coarse_factor_str][jj][1], 
                reflectance_diff_hist[coarse_factor_str][jj][2], 
                np.transpose(reflectance_diff_hist[coarse_factor_str][jj][0]),
                norm = colors.LogNorm(
                    vmin = 1.e-5, 
                    vmax = reflectance_diff_hist_max,
                ),
                cmap = count_cmap, shading = "flat"
            )

            # Row 1 - Heating
            row: NP_INT = NP_INT(1)
            heating_diff_pcm: MPL_PCOLORMESH = axs[row,jj].pcolormesh(
                heating_diff_hist[coarse_factor_str][jj][1], 
                heating_diff_hist[coarse_factor_str][jj][2], 
                np.transpose(heating_diff_hist[coarse_factor_str][jj][0]),
                norm = colors.LogNorm(
                    vmin = 1.e-8, 
                    vmax = heating_diff_hist_max,
                ),
                cmap = count_cmap, shading = "flat"
            )

            # Row 2 - Downwelling Surface Flux
            row: NP_INT = NP_INT(2)
            flux_sfc_dn_diff_pcm: MPL_PCOLORMESH = axs[row,jj].pcolormesh(
                flux_sfc_dn_diff_hist[coarse_factor_str][jj][1], 
                flux_sfc_dn_diff_hist[coarse_factor_str][jj][2], 
                np.transpose(flux_sfc_dn_diff_hist[coarse_factor_str][jj][0]),
                norm = colors.LogNorm(
                    vmin = 1.e-5, 
                    vmax = flux_sfc_dn_diff_hist_max,
                ),
                cmap = count_cmap, shading = "flat"
            )
            
            #-------------------------------------------------------------------
            # Add common column-wise plot elements
            #-------------------------------------------------------------------
            # x-ticks
            time: NP_ARRAY[NP_REAL] = daytime_times[jj]
            sza: NP_ARRAY[NP_REAL] = daytime_szas[jj]
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
        dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
        dx_str: str
        if dx < 1.e3:
            dx_str = r"{:.0f} $m$".format(dx)
        else:
            dx_str = r"{:.1f} $km$".format(dx * 1.e-3)

        fig.suptitle(r"Difference Distribution" + " - {}".format(dx_str))
        fig.supxlabel(r"Time $\left[ h \right]$")

        axs[0,0].set_ylabel(r"Reflectance")
        axs[1,0].set_ylabel(r"Heating Rate $\left[ K\,d^{-1} \right]$")
        axs[2,0].set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")

        # Colorbars
        reflectance_diff_cbar = fig.colorbar(reflectance_diff_pcm, ax = axs[0,...])
        heating_diff_cbar = fig.colorbar(heating_diff_pcm, ax = axs[1,...])
        flux_sfc_dn_diff_cbar = fig.colorbar(flux_sfc_dn_diff_pcm, ax = axs[2,...])

        # Eps hlines
        kk: int
        for kk in range(0, nrows):
            jj: int
            for jj in range(0, ndays):
                axs[kk,jj].axhline(-eps_array[kk], 
                    color = "gray", 
                    linestyle = "solid", 
                    linewidth = 0.5)
                axs[kk,jj].axhline(eps_array[kk], 
                    color = "gray", 
                    linestyle = "solid", 
                    linewidth = 0.5)
        
        # Axis scaling
        kk: int
        for kk in range(1, nrows):
            ax: MPL_AXES
            for ax in axs[kk,:]:
                ax.set_yscale("symlog", linthresh = eps_array[kk])

        #-----------------------------------------------------------------------
        # Plot contours at powers of 10
        #-----------------------------------------------------------------------
        msg: str = "Adding contours at powers of 10..."
        print_msg(msg)

        # Get levels and add horizontal lines to colorbars
        reflectance_diff_levels: NP_ARRAY[NP_REAL] = NP_REAL(reflectance_diff_cbar.ax.get_yticks())
        heating_diff_levels: NP_ARRAY[NP_REAL] = NP_REAL(heating_diff_cbar.ax.get_yticks())
        flux_sfc_dn_diff_levels: NP_ARRAY[NP_REAL] = NP_REAL(flux_sfc_dn_diff_cbar.ax.get_yticks())

        level: NP_REAL
        for level in reflectance_diff_levels:
            reflectance_diff_cbar.ax.axhline(
                level,
                color = "k",
                linewidth = 1.0,
                linestyle = "solid"
            )
        for level in heating_diff_levels:
            heating_diff_cbar.ax.axhline(
                level,
                color = "k",
                linewidth = 1.0,
                linestyle = "solid"
            )
        for level in flux_sfc_dn_diff_levels:
            flux_sfc_dn_diff_cbar.ax.axhline(
                level,
                color = "k",
                linewidth = 1.0,
                linestyle = "solid"
            )

        # Create contour plots
        jj: int
        for jj in range(0, ndays):
            msg: str = "- Day {}...".format(jj)
            print_msg(msg)

            daytime_time_bin_centers: NP_ARRAY[NP_REAL] = (
                reflectance_diff_hist[coarse_factor_str][jj][1][1:]
                + reflectance_diff_hist[coarse_factor_str][jj][1][:-1]
            ) / 2.
            
            # Row 0 - Reflectance
            row: NP_INT = NP_INT(0)
            reflectance_diff_bin_centers: NP_ARRAY[NP_REAL] = (
                reflectance_diff_hist[coarse_factor_str][jj][2][1:]
                + reflectance_diff_hist[coarse_factor_str][jj][2][:-1]
            ) / 2.

            axs[row,jj].contour(
                daytime_time_bin_centers, 
                reflectance_diff_bin_centers, 
                np.transpose(reflectance_diff_hist[coarse_factor_str][jj][0]),
                levels = reflectance_diff_levels,
                colors = "k",
                linewidths = 1.0
            )

            # Row 1 - Heating
            row: NP_INT = NP_INT(1)
            heating_diff_bin_centers: NP_ARRAY[NP_REAL] = (
                heating_diff_hist[coarse_factor_str][jj][2][1:]
                + heating_diff_hist[coarse_factor_str][jj][2][:-1]
            ) / 2.

            axs[row,jj].contour(
                daytime_time_bin_centers, 
                heating_diff_bin_centers, 
                np.transpose(heating_diff_hist[coarse_factor_str][jj][0]),
                levels = heating_diff_levels,
                colors = "k",
                linewidths = 1.0
            )

            # Row 2 - Downwelling Surface Flux
            row: NP_INT = NP_INT(2)
            flux_sfc_dn_diff_bin_centers: NP_ARRAY[NP_REAL] = (
                flux_sfc_dn_diff_hist[coarse_factor_str][jj][2][1:]
                + flux_sfc_dn_diff_hist[coarse_factor_str][jj][2][:-1]
            ) / 2.

            axs[row,jj].contour(
                daytime_time_bin_centers, 
                flux_sfc_dn_diff_bin_centers,  
                np.transpose(flux_sfc_dn_diff_hist[coarse_factor_str][jj][0]),
                levels = flux_sfc_dn_diff_levels,
                colors = "k",
                linewidths = 1.0
            )

        #---------------------------------------------------------------------------
        # Save the plot to file
        #---------------------------------------------------------------------------
        plt_filename = "rte_rrtmgp_cpp_error_distribution.{}.png".format(coarse_factor_str)
        plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
        fig.savefig(plt_filepath, dpi = 512)
        plt.close(fig)

if __name__ == "__main__":
    main()