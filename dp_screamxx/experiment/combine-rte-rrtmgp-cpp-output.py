# Standard Library Imports
import os
import re
import shutil
import subprocess

from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_COMM, MPI_ROOT, XR_DATASET, XR_DATAARRAY

# Script variables
prog_name: str = "combine-rte-rrtmgp-cpp-output.py"
prog_desc: str = "Combine RTE-RRTMGP-CPP+RT output into a single time-series file."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line arguments.".format(current_time)
    print(msg, flush = True)
    
    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory."
    )
    parser.add_argument("--rad-tran-separate-outdir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT separate output directory."
    )
    parser.add_argument("--rad-tran-combined-outdir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT combined output directory."
    )
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64."
    )
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_separate_outdir: str = os.path.normpath(args.rad_tran_separate_outdir)
    rad_tran_combined_outdir: str = os.path.normpath(args.rad_tran_combined_outdir)

    coarse_factors: Optional[NP_ARRAY[NP_INT]]
    if args.coarse_factors is None:
        coarse_factors = None
    else:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]

    #---------------------------------------------------------------------------
    # Get RTE-RRTMGP-CPP+RT input files
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Getting relevant RTE-RRTMGP-CPP+RT input file paths.".format(current_time)
    print(msg, flush = True)

    rad_tran_indir_list: list[str] = os.listdir(rad_tran_indir)
    rad_tran_infiles: list[str] = []
    if coarse_factors is not None:
        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            coarse_str: str = "lr_{:02}".format(coarse_factor)
            for file_name in rad_tran_indir_list:
                if coarse_str in file_name:
                    rad_tran_infiles += [os.path.join(rad_tran_indir, file_name)]
                    break
    else:
        coarse_factors: list[NP_INT] = []
        for file_name in rad_tran_indir_list:
            if ".in.nc" in file_name:
                rad_tran_infiles += [os.path.join(rad_tran_indir, file_name)]
                coarse_str = re.search(r'lr_(\d{2})', file_name).group()[3:]
                coarse_factors.append(NP_INT(coarse_str))
        coarse_factors: NP_ARRAY[NP_INT] = np.array(coarse_factors, dtype = NP_INT)

    coarse_factors = np.sort(coarse_factors)[::-1]
    rad_tran_infiles = sorted(rad_tran_infiles, reverse = True)

    #---------------------------------------------------------------------------
    # Loop through input files and collect all associated output files
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Looping through input files.".format(current_time)
    print(msg, flush = True)

    # Get coords that are common across all resolutions
    with xr.open_dataset(rad_tran_infiles[0], engine = "netcdf4", decode_timedelta = False) as xr_rad_tran_in:
            time: XR_DATAARRAY = xr_rad_tran_in["time"]
            z: XR_DATARRAY = xr_rad_tran_in["z"]
            lay: XR_DATARRAY = xr_rad_tran_in["z_lay"].rename("lay").rename({"z_lay" : "lay"})
            lev: XR_DATARRAY = xr_rad_tran_in["z_lev"].rename("lev").rename({"z_lev" : "lev"})

    t_re: re.Pattern = re.compile("t_...")
    rad_tran_separate_outdir_list: list[str] = os.listdir(rad_tran_separate_outdir)
    for rad_tran_infile in rad_tran_infiles:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}.".format(current_time, rad_tran_infile)
        print(msg, flush = True)

        #-----------------------------------------------------------------------
        # Get list of output files
        #-----------------------------------------------------------------------
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Finding corresponding output files.".format(current_time)
        print(msg, flush = True)

        rad_tran_file_basename: str = os.path.basename(rad_tran_infile)[:-6]
        coarse_str: str = rad_tran_file_basename[-5:]

        rad_tran_outfiles: list[str] = []
        for rad_tran_outfile in rad_tran_separate_outdir_list:
            if (coarse_str + ".t") in rad_tran_outfile:
                rad_tran_outfiles += [os.path.join(rad_tran_separate_outdir, rad_tran_outfile)]
        
        rad_tran_outfiles = sorted(rad_tran_outfiles)
        n_rad_tran_outfiles: int = len(rad_tran_outfiles)
        t_idxs: list[int] = [int(t_re.search(rad_tran_outfiles[ii]).group(0)[-3:])
            for ii in range(0, n_rad_tran_outfiles)]
        
        #-----------------------------------------------------------------------
        # Set up coordinates for combined output file
        #-----------------------------------------------------------------------
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Extracting time-independent fields.".format(current_time)
        print(msg, flush = True)

        with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran_in:
            x: XR_DATARRAY = xr_rad_tran_in["x"]
            y: XR_DATARRAY = xr_rad_tran_in["y"]

        xr_rad_tran_out_coords: dict = {
            "time" : time,
            "x" : x,
            "y" : y,
            "z" : z,
            "lay" : lay,
            "lev" : lev
        }

        with xr.open_dataset(rad_tran_outfiles[0], engine = "netcdf4", decode_timedelta = False) as xr_rad_tran_out:
            sw_band_lims_wvn: XR_DATAARRAY = xr_rad_tran_out["sw_band_lims_wvn"]

        xr_rad_tran_out_data_vars: dict = {"sw_band_lims_wvn" : sw_band_lims_wvn} # Initialize with time-independent fields

        #-----------------------------------------------------------------------
        # Extract time-dependent fields
        #-----------------------------------------------------------------------
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Extracting time-dependent fields.".format(current_time)
        print(msg, flush = True)

        rad_tran_out_keys: list[str] = ['p_lay', 'p_lev', 'sw_flux_up', 
            'sw_flux_dn', 'sw_flux_dn_dir', 'sw_flux_net', 'rt_flux_tod_up', 
            'rt_flux_sfc_dir', 'rt_flux_sfc_dif', 'rt_flux_sfc_up', 
            'rt_flux_abs_dir', 'rt_flux_abs_dif']
        n_rad_tran_out_keys: int = len(rad_tran_out_keys)

        rad_tran_out_key: str
        for rad_tran_out_key in rad_tran_out_keys:
            xr_rad_tran_out_data_vars[rad_tran_out_key] = xr.concat(
                [xr.open_dataset(rad_tran_outfiles[ii], engine = "netcdf4", decode_timedelta = False)[rad_tran_out_key]
                    for ii in range(0, n_rad_tran_outfiles)],
                dim = time[t_idxs]
            )

        #-----------------------------------------------------------------------
        # Writing combined output to file
        #-----------------------------------------------------------------------
        rad_tran_outfile: str = os.path.join(rad_tran_combined_outdir, rad_tran_file_basename + ".out.nc")

        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Writing combined output to {}.".format(current_time, rad_tran_outfile)
        print(msg, flush = True)

        xr_rad_tran_out: XR_DATASET = xr.Dataset(
            data_vars = xr_rad_tran_out_data_vars,
            coords = xr_rad_tran_out_coords
        )

        rad_tran_outfile: str = os.path.join(rad_tran_combined_outdir, rad_tran_file_basename + ".out.nc")

        xr_rad_tran_out.to_netcdf(rad_tran_outfile)

if __name__ == "__main__":
    main()