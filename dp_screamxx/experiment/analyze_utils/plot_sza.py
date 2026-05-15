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
    # Extract solar zenith angles.
    #---------------------------------------------------------------------------
    dp_scream_mu0 = xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False)["cosine_solar_zenith_angle"].isel(ncol = 0) # [time]
    rad_tran_mu0 = xr.open_dataset(rad_tran_infiles[0], engine = "netcdf4", decode_timedelta = False)["mu0"].isel(y = 0, x = 0) # [time]

if __name__ == "__main__":
    main()