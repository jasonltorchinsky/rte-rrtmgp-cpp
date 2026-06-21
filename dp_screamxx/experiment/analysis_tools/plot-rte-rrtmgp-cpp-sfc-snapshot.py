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
from consts.visual import flux_cmap, cw_cmap
from rte_rrtmgp_cpp import find_inout_pairs, find_mnn_indices, find_szas, find_times, \
    calc_cloud_wc, calc_sw_flux_sfc_dn

# Script variables
prog_name: str = "plot-rte-rrtmgp-cpp-sfc-snapshot"
prog_desc: str = "Visualize surface state for RTE-RRTMGP-CPP."

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
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64.")
        
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir: str = os.path.normpath(args.rad_tran_vizdir)
    working_dir: str = os.path.join(rad_tran_vizdir, os.path.normpath(args.working_dir))
    recalculate: bool = args.recalculate

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

            cloud_wc: XR_DATAARRAY = calc_cloud_wc(rad_tran_infile, mnn_indices[jj]) # Cloud water content; [g m^{-3}]; [3, lay, y, x]

            # Sneak in getting grid information before converting cloud_wc
            dz: NP_REAL = NP_REAL(cloud_wc["lay"][1] - cloud_wc["lay"][0]) # [m]
            x: NP_ARRAY[NP_REAL] = cloud_wc["x"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]
            y: NP_ARRAY[NP_REAL] = cloud_wc["y"].to_numpy().astype(NP_REAL) * 1.e-3 # [km]

            cloud_wc: list[NP_ARRAY[NP_REAL]] = [cloud_wc.isel(time = ll).to_numpy().astype(NP_REAL) for ll in range(0, 3)] # [g m^{-3}]; 3 * [lay, y, x]

            #-------------------------------------------------------------------
            # Calculate vertical cloud water path at each time
            #-------------------------------------------------------------------
            vwp: list[NP_ARRAY[NP_REAL]] = [np.transpose(dz * np.sum(cloud_wc[ll], axis = 0), axes = (1, 0)) for ll in range(0, 3)] # [g m^{-2}], 3 * [y, x]

            #-------------------------------------------------------------------
            # Calculate downwelling surface flux
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Calculating downwelling surface flux for day {} of {}...".format(current_time, jj, ndays - 1)
            print(msg, flush = True)

            flux_sfc_dn_rt: list[NP_ARRAY[NP_REAL]] = calc_sw_flux_sfc_dn(rad_tran_infile, rad_tran_outfile,
                mnn_indices[jj], solver = "rt") # Downwelling surface flux; [W m^{-2}]; [time, y, x]
            flux_sfc_dn_rt: list[NP_ARRAY[NP_REAL]] = [np.transpose(flux_sfc_dn_rt.isel(time = ll).to_numpy().astype(NP_REAL), axes = (1, 0)) for ll in range(0, 3)]

            flux_sfc_dn_ts: list[NP_ARRAY[NP_REAL]] = calc_sw_flux_sfc_dn(rad_tran_infile, rad_tran_outfile,
                mnn_indices[jj], solver = "ts") # Downwelling surface flux; [W m^{-2}]; [time, y, x]
            flux_sfc_dn_ts: list[NP_ARRAY[NP_REAL]] = [np.transpose(flux_sfc_dn_ts.isel(time = ll).to_numpy().astype(NP_REAL), axes = (1, 0)) for ll in range(0, 3)]
            
            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max: list[NP_REAL] = [vwp[ll].max() for ll in range(3)]
            vwp_min: list[NP_REAL] = [vwp[ll].min() for ll in range(3)]

            flux_sfc_dn_max: list[NP_REAL] = [max(flux_sfc_dn_rt[ll].max(), flux_sfc_dn_ts[ll].max()) for ll in range(3)]
            flux_sfc_dn_min: list[NP_REAL] = [min(flux_sfc_dn_rt[ll].min(), flux_sfc_dn_ts[ll].min()) for ll in range(3)]

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Plotting data...".format(current_time)
            print(msg, flush = True)

            nrows: NP_INT = NP_INT(3)
            ncols: NP_INT = NP_INT(3)
            fig_height: NP_REAL = NP_REAL(5.)
            fig_base_size = np.array([(ncols / nrows) * fig_height, fig_height])
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
                sharex = "col", sharey = True,
                constrained_layout = True,
                figsize = 3. * fig_base_size)

            # Row 0: Vertical Water Path
            vwp_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                vwp_pcm[ll] = axs[0, ll].pcolormesh(y, x, vwp[ll] * 1.e-3,
                    vmin = vwp_min[ll], vmax = vwp_max[ll],
                    cmap = cw_cmap)

            # Row 1: Two-Stream
            flux_sfc_dn_ts_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                flux_sfc_dn_ts_pcm[ll] = axs[1, ll].pcolormesh(y, x, flux_sfc_dn_ts[ll],
                    norm = colors.LogNorm(vmin = flux_sfc_dn_min[ll], vmax = flux_sfc_dn_max[ll]),
                    cmap = flux_cmap)

            # Row 2: Ray-Tracer
            flux_sfc_dn_rt_pcm: list[MPL_PCOLORMESH] = [[] for _ in range(0, ncols)]
            ll: int
            for ll in range(0, ncols):
                flux_sfc_dn_rt_pcm[ll] = axs[2, ll].pcolormesh(y, x, flux_sfc_dn_rt[ll],
                    norm = colors.LogNorm(vmin = flux_sfc_dn_min[ll], vmax = flux_sfc_dn_max[ll]),
                    cmap = flux_cmap)

            # Colorbars
            for ll in range(0, ncols):
                vwp_cbar = fig.colorbar(vwp_pcm[ll], ax = axs[0,ll])
                flux_sfc_dn_cbar = fig.colorbar(flux_sfc_dn_ts_pcm[ll], ax = axs[1:3,ll])

            # Labels
            fig.suptitle("RTE-RRTMGP-CPP Atmospheric Radiative Transfer")
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"y $\left[ km \right]$")

            for ll in range(0, ncols):
                col_title: str = (r"{:.2f} Hours - ".format(mnn_times[jj,ll])
                    + r"Solar Zenith Angle {:.1f}$^{{\circ}}$ - ".format(mnn_szas[jj,ll]))
                axs[0,ll].set_title(col_title)
            axs[1,0].set_ylabel(r"Two-Stream")
            axs[2,0].set_ylabel(r"Ray-Tracer")

            vwp_cbar.ax.set_ylabel(r"Vertical Cloud Water Path $\left[ kg\,m^{-2} \right]$")
            flux_sfc_dn_cbar.ax.set_ylabel(r"Surface Downwelling Flux $\left[ W\,m^{-2} \right]$")

            # Aspect ratio
            ll: int
            mm: int
            for ll in range(0, nrows):
                for mm in range(0, ncols):
                    axs[ll,mm].set_aspect("equal")

            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "rte_rrtmgp_cpp_sfc_day_{}.{}.png".format(jj, lr_str)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()