# Library imports
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import xarray as xr

# Constants
g = 9.80665 # Acceleration due to gravity at equator at sea level [m s^{-2}]
R_d = 287.047 # Gas constant for dry air [J kg^{-1} K]

def find_pairs(input_dir, output_dir, lrs):
    infiles = sorted(glob.glob(os.path.join(input_dir, "*.in.nc")))
    outfiles = sorted(glob.glob(os.path.join(output_dir, "*.out.nc")))

    paired_infiles = []
    paired_outfiles = []

    for ii in range(len(infiles)):
        infile_name = os.path.basename(infiles[ii])
        ext_re = re.compile(".in.nc")
        infile_base = re.sub(ext_re, "", infile_name)

        for lr in lrs:
            if lr in infile_base:
                for jj in range(len(outfiles)):
                    if infile_base in outfiles[jj]:
                        paired_infiles += [infiles[ii]]
                        paired_outfiles += [outfiles[jj]]
                        break

    return [paired_infiles, paired_outfiles]

def find_daytime_slices(mu0, tol = 1.e-3):
    daytime_mask = np.isfinite(mu0) & (mu0 > tol)
    daystart_indices = np.where(~daytime_mask.shift(time = 1, fill_value = False) & daytime_mask)[0]
    dayend_indices = np.where(~daytime_mask.shift(time = -1, fill_value = False) & daytime_mask)[0]

    ndays = daystart_indices.size

    daytime_slices = []
    for ii in range(ndays):
        daytime_slices += [slice(daystart_indices[ii], dayend_indices[ii] + 1)]

    return daytime_slices

def find_mnn_indices(mu0, tol = 1.e-3): # Morning, Noon, Night indices for a given day
    daytime_mask = np.isfinite(mu0) & (mu0 > tol)
    daystart_indices = np.where(~daytime_mask.shift(time = 1, fill_value = False) & daytime_mask)[0]
    dayend_indices = np.where(~daytime_mask.shift(time = -1, fill_value = False) & daytime_mask)[0]

    ndays = daystart_indices.size

    mnn_indices = np.zeros((ndays, 3), dtype = np.int32)
    for ii in range(ndays):
        daystart_index = daystart_indices[ii]
        dayend_index = dayend_indices[ii]
        index_range = dayend_index - daystart_index
        mnn_indices[ii,:] = np.array([int(np.round(0.15 * index_range + daystart_index)), # Morning
            int(np.round(0.5 * index_range + daystart_index)), # Noon
            int(np.round(0.67 * index_range + daystart_index))]) # Night

    return mnn_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rad-tran-indir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer input file directory.")
    parser.add_argument("--rad-tran-outdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer output file directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    parser.add_argument("--working-dir", nargs = "?", default = ".working", type = str,
        help = "Working directory to output calculated values.")
    parser.add_argument("--recalculate", nargs = "?", default = False, type = bool,
        help = "Re-calculate surface heating rates.")
    parser.add_argument("--lr", nargs = "?", default = "", type = str,
        help = "Resolution factor tag.")
    parser.add_argument("--case", nargs = "?", default = "", type = str,
        help = "Case to determine heating rate calculation parameters.")
    parser.add_argument("--detailed-calc", nargs = "?", default = False, type = bool,
        help = ("True: Compute cloud water mass using VMRs, etc. "
            "False: Compute cloud water mass using standard values."))
    args = parser.parse_args()

    rad_tran_indir  = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir = os.path.normpath(args.rad_tran_outdir)
    rad_tran_vizdir = os.path.normpath(args.rad_tran_vizdir)
    working_dir = args.working_dir
    recalculate = args.recalculate
    lrs = [str(lr) for lr in args.lr.split(",")]
    case = args.case
    detailed_calc = args.detailed_calc

    dirs = [rad_tran_vizdir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    #---------------------------------------------------------------------------
    # Verify necessary files are present.
    #---------------------------------------------------------------------------
    [rad_tran_infiles, rad_tran_outfiles] = find_pairs(rad_tran_indir, rad_tran_outdir, lrs)

    #---------------------------------------------------------------------------
    # Read files.
    #---------------------------------------------------------------------------
    for ii in range(len(rad_tran_infiles)):
        rad_tran_infile  = rad_tran_infiles[ii]
        rad_tran_outfile = rad_tran_outfiles[ii]

        #-----------------------------------------------------------------------
        # Get information for plot name
        #-----------------------------------------------------------------------
        lr_re = re.compile("lr_..")
        lr_str = re.search(lr_re, rad_tran_infile).group()

        #-----------------------------------------------------------------------
        # Determine morning, noon, night indices
        #-----------------------------------------------------------------------
        mu0_ds = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0)
        in_mnn_indices = find_mnn_indices(mu0_ds) # mnn_indices for rad_tran_input

        # ASSUME: That rad_tran_outfile time dimension has indexes of timesteps,
        # which may be off-set.
        out_mnn_index_offset = int(xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = 0))
        out_mnn_indices = in_mnn_indices - out_mnn_index_offset

        #-----------------------------------------------------------------------
        # Obtain information common across each day
        #-----------------------------------------------------------------------
        x = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["x"].values / 1000 # [km]
        y = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["y"].values / 1000 # [km]

        dx = x[1] - x[0] # Grid spacing ASSUME: Constant and same in y

        ndays = in_mnn_indices.shape[0]
        for ii in range(ndays):
            in_mnn_index = in_mnn_indices[ii]
            out_mnn_index = out_mnn_indices[ii]

            #-------------------------------------------------------------------
            # Obtain day-specific information for plot labels
            #-------------------------------------------------------------------
            mnn_szas = np.rad2deg(np.acos(xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["mu0"].isel(x = 0, y = 0, time = in_mnn_index))).values
            mnn_times = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["time"].isel(time = in_mnn_index).values

            #-------------------------------------------------------------------
            # Obtain cloud liquid- and ice-water mixing ratios
            #-------------------------------------------------------------------
            lwp = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["lwp"].isel(time = in_mnn_index) # [time, lay, y, x]
            iwp = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["iwp"].isel(time = in_mnn_index) # [time, lay, y, x]
            p_lev = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["p_lev"].isel(time = in_mnn_index) # [time, lev, y, x]

            dp = -p_lev.diff("lev").rename({"lev": "lay"}).rename("dp").assign_coords(lay = lwp["lay"].values) # Pressure thickness # [time, lay, y, x]
            qc = g * lwp / dp # Cloud Liquid-Water Mass-Mixing Ratio [time, lay, y, x]
            qi = g * iwp / dp # Cloud Ice Water Mass-Mixing Ratio [time, lay, y, x]

            # ASSUME: Volume of each element is constant, dx = dy
            lev = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["lev"].isel(lev = [0, 1]).values / 1000. # [km]
            dz = lev[1] - lev[0] # [km]
            vol = dx**2 * dz * 1.e9 # Convert [km^{3}] to [m^{3}]

            t_lay = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["t_lay"].isel(time = in_mnn_index) # [time, lay, y, x]
            p_lay = xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False)["p_lay"].isel(time = in_mnn_index) # [time, lay, y, x]

            if detailed_calc:
                continue
                #TO-DO: IMPLEMENT THIS
            else:
                R = R_d
            mass_air = (p_lay * vol) / (R * t_lay) # [kg]

            lwc = qc * mass_air / vol # [kg m^{-3}]
            iwc = qi * mass_air / vol # [kg m^{-3}]
            wc = lwc + iwc

            vwp = (dz * 1.e3) * wc.sum(dim = "lay") * 1.e3 # Convert to [g m^{-2}]

            #-------------------------------------------------------------------
            # Obtain two-stream (ts) and ray-tracer (rt) data
            #-------------------------------------------------------------------
            ts_sfc_dn = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["sw_flux_dn"].isel(lev = 0, time = out_mnn_index) # [time, y, x]

            rt_sfc_dif = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["rt_flux_sfc_dif"].isel(time = out_mnn_index)
            rt_sfc_dir = xr.open_dataset(rad_tran_outfile, engine = "netcdf4", decode_timedelta = False)["rt_flux_sfc_dir"].isel(time = out_mnn_index)
            rt_sfc_dn = rt_sfc_dir + rt_sfc_dif # [time, y, x]

            #-------------------------------------------------------------------
            # Obtain data bounds
            #-------------------------------------------------------------------
            vwp_max = vwp.max(dim = ["x", "y"]).values
            vwp_min = vwp.min(dim = ["x", "y"]).values

            mnn_max = np.max(np.stack([ts_sfc_dn.max(dim = ["x", "y"]), rt_sfc_dn.max(dim = ["x", "y"])]), axis = 0)
            mnn_min = np.min(np.stack([ts_sfc_dn.min(dim = ["x", "y"]), rt_sfc_dn.min(dim = ["x", "y"])]), axis = 0)

            diff_max = np.abs(rt_sfc_dn - ts_sfc_dn).max(dim = ["x", "y"]).values
            diff_min = -diff_max

            #-------------------------------------------------------------------
            # Prepare data for plotting
            #-------------------------------------------------------------------
            vwp = np.transpose(vwp, axes = [0, 2, 1]) # [time, x, y]
            ts_sfc_dn = np.transpose(ts_sfc_dn.values, axes = [0, 2, 1]) # [time, x, y]
            rt_sfc_dn = np.transpose(rt_sfc_dn.values, axes = [0, 2, 1]) # [time, x, y]
            diff_sfc_dn = rt_sfc_dn - ts_sfc_dn

            #-------------------------------------------------------------------
            # Plot the data
            #-------------------------------------------------------------------
            fig, axs = plt.subplots(nrows = 4, ncols = 3,
                sharex = True, sharey = True,
                constrained_layout = True,
                figsize = (14, 14))

            heating_cmap = "hot"
            flux_cmap = "magma"

            # Row 1: Vertical Water Path
            vwp_pcm = [[] for jj in range(3)]
            for jj in range(3):
                vwp_pcm[jj] = axs[0, jj].pcolormesh(x, y, vwp[jj,:],
                    vmin = vwp_min[jj], vmax = vwp_max[jj],
                    cmap = "Blues")

            # Row 1: Two-Stream
            ts_pcm = [[] for jj in range(3)]
            for jj in range(3):
                ts_pcm[jj] = axs[1, jj].pcolormesh(x, y, ts_sfc_dn[jj,:],
                    vmin = mnn_min[jj], vmax = mnn_max[jj],
                    cmap = flux_cmap)

            # Row 2: Ray-Tracer
            rt_pcm = [[] for jj in range(3)]
            for jj in range(3):
                rt_pcm[jj] = axs[2, jj].pcolormesh(x, y, rt_sfc_dn[jj,:],
                    vmin = mnn_min[jj], vmax = mnn_max[jj],
                    cmap = flux_cmap)
            
            # Row 2: Ray-Tracer - Two-Stream
            diff_pcm = [[] for jj in range(3)]
            for jj in range(3):
                diff_pcm[jj] = axs[3, jj].pcolormesh(x, y, diff_sfc_dn[jj,:],
                    vmin = diff_min[jj], vmax = diff_max[jj],
                    cmap = "RdBu")

            # Colorbars
            for jj in range(3):
                vwp_cbar = fig.colorbar(vwp_pcm[jj], ax = axs[0,jj])
                rt_cbar = fig.colorbar(rt_pcm[jj], ax = axs[1:3,jj])
                diff_cbar = fig.colorbar(diff_pcm[jj], ax = axs[3,jj])

            # Labels
            lr_label = r"{:0.0f} $m$".format(dx * 1000.) if dx < 1.0 else r"{:0.2f} $km$".format(dx)
            fig.suptitle("Horizontal Resolution - {}".format(lr_label))
            fig.supxlabel(r"x $\left[ km \right]$")
            fig.supylabel(r"y $\left[ km \right]$")

            for jj in range(3):
                axs[0,jj].set_title(r"{:.2f} Hours - Solar Zenith Angle {:.1f}$^{{\circ}}$".format(mnn_times[jj], mnn_szas[jj]))
            axs[1,0].set_ylabel("Two-Stream")
            axs[2,0].set_ylabel("Ray-Tracer")
            axs[3,0].set_ylabel("Ray-Tracer - Two-Stream")
            
            vwp_cbar.ax.set_ylabel(r"Vertical Water Path $\left[ g\,m^{-2} \right]$")
            rt_cbar.ax.set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")
            diff_cbar.ax.set_ylabel(r"Downwelling Surface Flux $\left[ W\,m^{-2} \right]$")


            #-------------------------------------------------------------------
            # Save the plot to file
            #-------------------------------------------------------------------
            plt_filename = "{}_sfc_heating_day_{}.png".format(lr_str, ii)
            plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
            fig.savefig(plt_filepath, dpi = 200)
            plt.close(fig)

if __name__ == "__main__":
    main()