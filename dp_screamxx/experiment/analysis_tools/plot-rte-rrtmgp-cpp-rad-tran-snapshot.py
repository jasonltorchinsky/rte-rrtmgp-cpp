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

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_PCOLORMESH
from consts.visual import flux_cmap, heating_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_heating, calc_sw_flux_abs

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-rad-tran-snapshot"
prog_desc: str = "Visualize absorbed shortwave flux and atmospheric heating rates for RTE-RRTMGP-CPP."

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
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--zmax", nargs = "?", default = 16., type = float,
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
    zmax: NP_REAL = NP_REAL(args.zmax)

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

    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]
        rad_tran_outfile: str = rad_tran_outfiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, lr_str)
        print(msg, flush = True)

        #-----------------------------------------------------------------------
        # Obtain Morning-Noon-Night time indices, times, SZAs, zmax index
        #-----------------------------------------------------------------------
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Obtaining morning-noon-night information...".format(current_time)
        print(msg, flush = True)

        mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
        mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = mnn_indices.shape[0]

        #-----------------------------------------------------------------------
        # Calculate fields for each MNN of each day
        #-----------------------------------------------------------------------
        int: jj
        for jj in range(0, ndays):
            #-------------------------------------------------------------------
            # Calculate cloud water content, x-indices for yz-slices for calculations
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating cloud water content for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj], zmax = zmax) # Cloud water content; [g m^{-3}]; [3, lev, y, x]
            x_indices: NP_ARRAY[NP_INT] = np.array([np.unravel_index(np.argmax(cloud_wc.isel(time = ll).to_numpy()), cloud_wc.shape[1:])[2] for ll in range(3)]) # [3]
            yz_slices_x: NP_ARRAY[NP_REAL] = cloud_wc["x"][x_indices].to_numpy().astype(NP_REAL) * 1.e-3 # x-location of yz-slices; [km]; [3]

            # Sneak in getting grid information before converting cloud_wc
            dx: NP_REAL = NP_REAL(cloud_wc["x"][1] - cloud_wc["x"][0]) # [m]
            y: NP_ARRAY[NP_REAL] = cloud_wc["y"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]
            z: NP_ARRAY[NP_REAL] = cloud_wc["lay"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]

            cloud_wc: list[NP_ARRAY[NP_REAL]] = [cloud_wc.isel(time = ll, x = x_indices[ll]).to_numpy().astype(NP_REAL) for ll in range(0, 3)] # [g m^{-3}]; 3 * [lev, y]

            #-------------------------------------------------------------------
            # Calculate radiative quantities
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating heating rates for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_heating_rt: list[NP_ARRAY[NP_REAL]] = calc_sw_heating(rad_tran_infile, rad_tran_outfile,
                mnn_indices[jj], x_indices, solver = "rt", zmax = zmax) # Shortwave heating rate, ray-tracer; [K d^{-1}]; 3 * [lev, y]

            sw_heating_ts: list[NP_ARRAY[NP_REAL]] = calc_sw_heating(rad_tran_infile, rad_tran_outfile,
                mnn_indices[jj], x_indices, solver = "ts", zmax = zmax) # Shortwave heating rate, two-stream; [K d^{-1}]; 3 * [lev, y]

            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating absorbed fluxes for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            sw_flux_abs_rt: list[NP_ARRAY[NP_REAL]] = calc_sw_flux_abs(rad_tran_infile, rad_tran_outfile,
                mnn_indices[jj], x_indices, solver = "rt", zmax = zmax) # Shortwave absorbed flux, ray-tracer; [W m^{-3}]; 3 * [lev, y]

            sw_flux_abs_ts: list[NP_ARRAY[NP_REAL]] = calc_sw_flux_abs(rad_tran_infile, rad_tran_outfile,
                mnn_indices[jj], x_indices, solver = "ts", zmax = zmax) # Shortwave absorbed flux, two-stream; [W m^{-3}]; 3 * [lev, y]

            #-------------------------------------------------------------------
            # Obtain plotting bounds
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Obtaining plotting bounds...".format(current_time)
            print(msg, flush = True)

            y_window_width: NP_REAL = 3. * zmax # Width of y window [km]
            
            y_bounds: NP_ARRAY[NP_REAL]
            y_bound_indices: list[NP_ARRAY[NP_INT]]
            if y_window_width > y.max() - y.min():
                y_bounds = np.tile(np.array([y.min(), y.max()]), [3, 1]) # [3, 2]
                y_bound_indices = [np.array([0, -1], dtype = NP_INT) for _ in range(0, 3)]
            else:
                y_bounds = np.zeros([3, 2], dtype = NP_REAL) # Limits for horizontal plotting axis (called xlim in matplotlib)
                y_bound_indices = [np.zeros([2], dtype = NP_INT) for _ in range(0, 3)]
                for ll in range(3):
                    cloud_wc_max_index: NP_ARRAY[NP_INT] = np.unravel_index(np.argmax(cloud_wc[ll]), cloud_wc[ll].shape) # [lay_index, y_index] of maximal cloud_wc
                    y_loc: NP_REAL = y[cloud_wc_max_index[1]] # Y-location of maximal cloud_wc [km]
                    y_bounds[ll,:] = [y_loc - y_window_width / 2., y_loc + y_window_width / 2.]

                    # Don't have to worry about shifting window past the edge after this block
                    # because the window is not as wide as the domain
                    if y_bounds[ll,0] < y.min():
                        y_bounds[ll,:] += (y.min() - y_bounds[ll,0])

                    if y_bounds[ll,1] > y.max():
                        y_bounds[ll,:] += (y.max() - y_bounds[ll,1])

                    y_bound_indices[ll][0] = np.max(np.where(y - y_bounds[ll,0] <= 0)[0])
                    y_bound_indices[ll][1] = np.min(np.where(y_bounds[ll,1] - y <= 0)[0]) + 1 # To include endpoint, add 1
            y_slices = [y[y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(0, 3)]

            #-------------------------------------------------------------------
            # Trim fields to the yz-slice
            #-------------------------------------------------------------------
            cloud_wc = [cloud_wc[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
            sw_heating_ts = [sw_heating_ts[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
            sw_heating_rt = [sw_heating_rt[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
            sw_flux_abs_ts = [sw_flux_abs_ts[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
            sw_flux_abs_rt = [sw_flux_abs_rt[ll][...,y_bound_indices[ll][0]:y_bound_indices[ll][1]] for ll in range(3)]
            
            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            cloud_wc_max: list[NP_REAL] = [cloud_wc[ll].max() for ll in range(3)]
            cloud_wc_min: list[NP_REAL] = [cloud_wc[ll].min() for ll in range(3)]

            sw_heating_max: list[NP_REAL] = [max(sw_heating_ts[ll].max(), sw_heating_rt[ll].max()) for ll in range(3)]
            sw_heating_min: list[NP_REAL] = [min(sw_heating_ts[ll].min(), sw_heating_rt[ll].min()) for ll in range(3)]

            sw_flux_abs_max: list[NP_REAL] = [max(sw_flux_abs_ts[ll].max(), sw_flux_abs_rt[ll].max()) for ll in range(3)]
            sw_flux_abs_min: list[NP_REAL] = [min(sw_flux_abs_ts[ll].min(), sw_flux_abs_rt[ll].min()) for ll in range(3)]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting data...".format(current_time)
            print(msg, flush = True)

            nrows: NP_INT = NP_INT(5)
            ncols: NP_INT = NP_INT(3)
            fig_height: NP_REAL = NP_REAL(6.)
            fig_base_size = np.array([(ncols / nrows) * (y_window_width / zmax) * fig_height, fig_height])
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
                sharex = "col", sharey = True,
                constrained_layout = True,
                figsize = 3. * fig_base_size)

            # Row 0: Horizontal Water Path
            cloud_wc_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                cloud_wc_pcm[ll] = axs[0, ll].pcolormesh(y_slices[ll], z, cloud_wc[ll],
                    vmin = cloud_wc_min[ll], vmax = cloud_wc_max[ll],
                    cmap = cw_cmap)

            # Row 1: Heating, Two-Stream
            sw_heating_ts_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                sw_heating_ts_pcm[ll] = axs[1, ll].pcolormesh(y_slices[ll], z, sw_heating_ts[ll],
                    norm = colors.LogNorm(vmin = sw_heating_min[ll], vmax = sw_heating_max[ll]),
                    cmap = heating_cmap)

            # Row 2: Heating, Ray-Tracer
            sw_heating_rt_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                sw_heating_rt_pcm[ll] = axs[2, ll].pcolormesh(y_slices[ll], z, sw_heating_rt[ll],
                    norm = colors.LogNorm(vmin = sw_heating_min[ll], vmax = sw_heating_max[ll]),
                    cmap = heating_cmap)

            # Row 3: Absorbed Flux, Two-Stream
            sw_flux_abs_ts_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                sw_flux_abs_ts_pcm[ll] = axs[3, ll].pcolormesh(y_slices[ll], z, sw_flux_abs_ts[ll],
                    norm = colors.LogNorm(vmin = sw_flux_abs_min[ll], vmax = sw_flux_abs_max[ll]),
                    cmap = flux_cmap)

            # Row 4: Absorbed Flux, Ray-Tracer
            sw_flux_abs_rt_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                sw_flux_abs_rt_pcm[ll] = axs[4, ll].pcolormesh(y_slices[ll], z, sw_flux_abs_rt[ll],
                    norm = colors.LogNorm(vmin = sw_flux_abs_min[ll], vmax = sw_flux_abs_max[ll]),
                    cmap = flux_cmap)

            # Colorbars
            for ll in range(0, ncols):
                cloud_wc_cbar = fig.colorbar(cloud_wc_pcm[ll], ax = axs[0,ll])
                sw_heating_cbar = fig.colorbar(sw_heating_ts_pcm[ll], ax = axs[1:3,ll])
                sw_flux_abs_cbar = fig.colorbar(sw_flux_abs_ts_pcm[ll], ax = axs[3:5,ll])

            # Labels
            fig.suptitle("RTE-RRTMGP-CPP Atmospheric Radiative Transfer")
            fig.supxlabel(r"y $\left[ km \right]$")
            fig.supylabel(r"z $\left[ km \right]$")

            for ll in range(3):
                col_title: str = (r"{:.2f} Hours - ".format(mnn_times[jj,ll])
                    + r"Solar Zenith Angle {:.1f}$^{{\circ}}$ - ".format(mnn_szas[jj,ll])
                    + r"$x$ = {:.2f} $\left[ km \right]$".format(yz_slices_x[ll]))
                axs[0,ll].set_title(col_title)
            axs[1,0].set_ylabel(r"Two-Stream")
            axs[2,0].set_ylabel(r"Ray-Tracer")
            axs[3,0].set_ylabel(r"Two-Stream")
            axs[4,0].set_ylabel(r"Ray-Tracer")

            cloud_wc_cbar.ax.set_ylabel(r"Cloud Water Content $\left[ g\,m^{-3} \right]$")
            sw_heating_cbar.ax.set_ylabel(r"Atmospheric Heating Rate $\left[ K\,d^{-1} \right]$")
            sw_flux_abs_cbar.ax.set_ylabel(r"Absorbed Shortwave Flux $\left[ W\,m^{-3} \right]$")

            # Set horizontal limits
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_xlim(y_bounds[mm,:])
                    axs[ll,mm].set_ylim((0.0, zmax))

            # Aspect ratio
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_aspect("equal")

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_rad_tran_day_{}.{}.png".format(jj, lr_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()