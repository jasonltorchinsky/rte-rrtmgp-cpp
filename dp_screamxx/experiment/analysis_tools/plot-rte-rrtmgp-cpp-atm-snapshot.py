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

# Third-Party Library Imports
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPL_PCOLORMESH
from consts.numeric import NP_SMALL
from consts.visual import lw_cmap, iw_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_dei, calc_rel, calc_z_max_info, find_grid, find_y_islice, print_msg

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-atm-snapshot"
prog_desc: str = "Visualize atmospheric state for RTE-RRTMGP-CPP."

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
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
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

    int: ii
    for ii in range(0, nfiles):
        rad_tran_infile: str = rad_tran_infiles[ii]

        lr_str: str = lr_re.search(rad_tran_infile).group()

        msg: str = "Processing {}...".format(lr_str)
        print_msg(msg)

        #-----------------------------------------------------------------------
        # Obtain grid information
        #-----------------------------------------------------------------------
        msg: str = "Obtaining grid information..."
        print_msg(msg)
        grid: dict = find_grid(rad_tran_infile)

        #-----------------------------------------------------------------------
        # Obtain Morning-Noon-Night time indices, times, SZAs, z_max_info
        #-----------------------------------------------------------------------
        msg: str = "Obtaining morning-noon-night information..."
        print_msg(msg)

        mnn_indices: NP_ARRAY[NP_INT] = find_mnn_indices(rad_tran_infile) # [ndays, 3]
        mnn_times: NP_ARRAY[NP_REAL] = find_times(rad_tran_infile, mnn_indices) # Time since simulation start; [h]; [ndays, 3]
        mnn_szas: NP_ARRAY[NP_REAL] = find_szas(rad_tran_infile, mnn_indices) # Solar zenith angle (SZA); [degrees]; [ndays, 3]
        ndays: NP_INT = NP_INT(mnn_indices.shape[0])
        z_max_info: dict = calc_z_max_info(rad_tran_infile, method = "cloud_top")

        #-----------------------------------------------------------------------
        # Calculate fields for each MNN of each day
        #-----------------------------------------------------------------------
        jj: int
        for jj in range(0, ndays):
            day_str: str = "day_{}".format(jj)

            #-------------------------------------------------------------------
            # Calculate spatial extent of plots based on maximal cloud water content
            #-------------------------------------------------------------------
            msg: str = "Calculating plot spatial extent for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, 
                time_indices = mnn_indices[jj],
                z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [time, lay, y, x]
            
            x_indices: NP_ARRAY[NP_INT] = np.array([np.unravel_index(np.argmax(cloud_wc.isel(time = ll).to_numpy()), cloud_wc.shape[1:])[2] for ll in range(0, 3)])
            y_slice_width: NP_REAL = 3. * z_max_info["zh_max"] # Width of y-slices [km]
            y_islices: list[slice] = [[] for _ in range(0, 3)]
            for ll in range(0, 3):
                y_islices[ll] = find_y_islice(grid["y"], cloud_wc.isel(time = ll, x = x_indices[ll]), slice_width = y_slice_width)

            #-------------------------------------------------------------------
            # Calculate desired atmospheric quantities
            #-------------------------------------------------------------------
            msg: str = "Calculating desired atmospheric quantities for day {} of {}...".format(jj, ndays - 1)
            print_msg(msg)

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, 
                time_indices = mnn_indices[jj], 
                x_indices = x_indices, 
                z_max_info = z_max_info) # Cloud water content; [g m^{-3}]; [slice, lay, y]
            rel: XR_DATAARRAY = calc_rel(rad_tran_infile, 
                time_indices = mnn_indices[jj], 
                x_indices = x_indices, 
                z_max_info = z_max_info) # Cloud liquid water effective radius; [μm]; [slice, lay, y]
            dei: XR_DATAARRAY = calc_dei(rad_tran_infile, 
                time_indices = mnn_indices[jj], 
                x_indices = x_indices, 
                z_max_info = z_max_info) # Cloud ice water effective diameter; [μm]; [slice, lay, y]

            #-------------------------------------------------------------------
            # Trim fields to the yz-slice
            #-------------------------------------------------------------------
            cloud_wc: list[XR_DATAARRAY] = [(cloud_wc
                .isel(y = y_islices[ll], slice = ll)
                .load()) for ll in range(0, 3)]   
            rel: list[XR_DATAARRAY] = [(rel
                .isel(y = y_islices[ll], slice = ll)
                .load()) for ll in range(0, 3)]
            dei: list[XR_DATAARRAY] = [(dei
                .isel(y = y_islices[ll], slice = ll)
                .load()) for ll in range(0, 3)]

            #-------------------------------------------------------------------
            # Trim grids to the yz-slice
            #-------------------------------------------------------------------
            yh_islices: list[slice] = [slice(
                    y_islices[ll].start, y_islices[ll].stop + 1
                ) for ll in range(0, 3)]
            yh: list[XR_DATAARRAY] = [(grid["yh"]
                .isel(yh = yh_islices[ll])
                .load()) * 1.e-3 for ll in range(0, 3)] # [m] => [km]

            zh: list[XR_DATAARRAY] = [(grid["zh"]
                .isel(zh = z_max_info["isel_indexers"]["zh"])
                .load()) * 1.e-3 for ll in range(0, 3)] # [m] => [km]
            
            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            cloud_wc_max: list[NP_REAL] = [NP_REAL(cloud_wc[ll].max()) for ll in range(0,3)]
            cloud_wc_min: list[NP_REAL] = [NP_REAL(cloud_wc[ll].min()) for ll in range(0,3)]

            rel_max: list[NP_REAL] = [NP_REAL(rel[ll].max()) for ll in range(0,3)]
            rel_min: list[NP_REAL] = [NP_REAL(rel[ll].min()) for ll in range(0,3)]

            dei_max: list[NP_REAL] = [NP_REAL(dei[ll].max()) for ll in range(0,3)]
            dei_min: list[NP_REAL] = [NP_REAL(dei[ll].min()) for ll in range(0,3)]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            msg: str = "Plotting data..."
            print_msg(msg)

            nrows: NP_INT = NP_INT(3)
            ncols: NP_INT = NP_INT(3)
            fig_height: NP_REAL = NP_REAL(3.)
            fig_base_size = np.array([(y_slice_width / z_max_info["z_max"]) * fig_height, fig_height])
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
                sharex = "col", sharey = True,
                constrained_layout = True,
                figsize = 3. * fig_base_size)

            # Row 0: Cloud Water Content
            cloud_wc_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                cloud_wc_pcm[ll] = axs[0, ll].pcolormesh(yh[ll], zh[ll], cloud_wc[ll],
                    norm = colors.LogNorm(vmin = max(1.e-6, cloud_wc_min[ll]), vmax = cloud_wc_max[ll]),
                    cmap = cw_cmap, shading = "flat")

            # Row 1: Cloud Liquid Water Effective Radius
            rel_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                rel_pcm[ll] = axs[1, ll].pcolormesh(yh[ll], zh[ll], rel[ll],
                    vmin = rel_min[ll], vmax = rel_max[ll],
                    cmap = lw_cmap, shading = "flat")

            # Row 2: Cloud Ice Water Effective Diameter
            dei_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                dei_pcm[ll] = axs[2, ll].pcolormesh(yh[ll], zh[ll], dei[ll],
                    vmin = rel_min[ll], vmax = rel_max[ll],
                    cmap = iw_cmap, shading = "flat")

            # Colorbars
            for ll in range(0, ncols):
                cloud_wc_cbar = fig.colorbar(cloud_wc_pcm[ll], ax = axs[0,ll], extend = "min")
                rel_cbar = fig.colorbar(rel_pcm[ll], ax = axs[1,ll])
                dei_cbar = fig.colorbar(dei_pcm[ll], ax = axs[2,ll])

            # Labels
            dx: NP_REAL = NP_REAL(grid["xh"][1] - grid["xh"][0]) # [m]
            dx_str: str
            if dx < 1.e3:
                dx_str = r"{:.0f} $m$".format(dx)
            else:
                dx_str = r"{:.1f} $km$".format(dx * 1.e-3)

            fig.suptitle(r"RTE-RRTMGP-CPP Atmospheric Snapshots - {}".format(dx_str))
            fig.supxlabel(r"y $\left[ km \right]$")
            fig.supylabel(r"z $\left[ km \right]$")

            for ll in range(0, ncols):
                x_pos: NP_REAL = NP_REAL(grid["x"].isel(x = x_indices[ll])) * 1.e-3 # [m] => [km]
                col_title: str = (r"{:.2f} Hours - ".format(mnn_times[jj,ll])
                    + r"$x$ = {:.2f} $\left[ km \right]$".format(x_pos))
                axs[0,ll].set_title(col_title)

            cloud_wc_cbar.ax.set_ylabel(r"Cloud Water Content $\left[ g\,m^{-3} \right]$")          
            rel_cbar.ax.set_ylabel(r"$rel$ $\left[ \mu m \right]$")
            dei_cbar.ax.set_ylabel(r"$dei$ $\left[ \mu m \right]$")

            # Aspect ratio
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_aspect("equal")

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_atm_snapshot.{}.{}.png".format(lr_str, day_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()