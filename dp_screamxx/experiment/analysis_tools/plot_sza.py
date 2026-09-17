# Library imports
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr

# Local imports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-scream-file", nargs = "?", required = True, type = str,
        help = "DP-SCREAM file path.")
    parser.add_argument("--rad-tran-indir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer input file directory.")
    parser.add_argument("--rad-tran-vizdir", nargs = "?", required = True, type = str,
        help = "Radiative Transfer visualization file directory.")
    args = parser.parse_args()

    dp_scream_file  = os.path.normpath(args.dp_scream_file)
    rad_tran_indir  = os.path.normpath(args.rad_tran_indir)
    rad_tran_vizdir = os.path.normpath(args.rad_tran_vizdir)

    dirs = [rad_tran_vizdir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    #---------------------------------------------------------------------------
    # Verify necessary files are present.
    #---------------------------------------------------------------------------
    rad_tran_infiles = sorted(glob.glob(os.path.join(rad_tran_indir, "*.in.nc")), reverse = True)
    assert(len(rad_tran_infiles) > 0)
    assert(os.path.isfile(dp_scream_file))

    #---------------------------------------------------------------------------
    # Extract times and solar zenith angles.
    #---------------------------------------------------------------------------
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        dp_scream_times = (xr_dp_scream["time"] - xr_dp_scream["time"][0]).to_numpy() / (3600.e9) # [ns] => [h]
        dp_scream_times = dp_scream_times.astype(np.float64)
        dp_scream_mu0 = xr_dp_scream["cosine_solar_zenith_angle"].isel(ncol = 0) # [time]
    dp_scream_sza = np.rad2deg(np.acos(dp_scream_mu0))

    with xr.open_dataset(rad_tran_infiles[0], engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        rad_tran_times = xr_rad_tran["time"] # [h]
        rad_tran_mu0 = xr_rad_tran["mu0"].isel(y = 0, x = 0) # [time]
    rad_tran_sza = np.rad2deg(np.acos(rad_tran_mu0))

    #---------------------------------------------------------------------------
    # Plot the solar zenith angles
    #---------------------------------------------------------------------------
    fig, axs = plt.subplots(constrained_layout = True,
        figsize = (14, 14))

    # Plot the data
    axs.plot(dp_scream_times, dp_scream_sza, color = "black", label = "DP-SCREAM")
    axs.plot(rad_tran_times, rad_tran_sza, color = "red", label = "RTE-RRTMGP-CPP")

    # Set Labels
    axs.set_xlabel(r"Time Since Simulation Start $\left[ h \right]$")
    axs.set_ylabel(r"Solar Zenith Angle $\left[ ^{\circ} \right]$")

    # Create legend
    axs.legend()

    plt_filename = "sza.png"
    plt_filepath = os.path.join(rad_tran_vizdir, plt_filename)
    fig.savefig(plt_filepath, dpi = 200)
    plt.close(fig)


if __name__ == "__main__":
    main()