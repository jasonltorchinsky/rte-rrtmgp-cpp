# Standard Library Imports
import os
import re
import resource
import shutil
import subprocess

from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

# Script variables
prog_name: str = "combine-rte-rrtmgp-cpp-output"
prog_desc: str = "Combine RTE-RRTMGP-CPP+RT output into a single time-series file."

def log_mem(label: str):
    rss_kib: NP_REAL = NP_REAL(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) # [KiB]

    # On Linux, ru_maxrss is usually KiB.
    rss_mib: NP_REAL = rss_kib / 1024.0 # [KiB] => [MiB]

    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: [{}] : Memory - max RSS = {:0.2f} MiB".format(current_time, label, rss_mib)
    print(msg, flush = True)

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line arguments.".format(current_time)
    print(msg, flush = True)
    
    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-combined-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT combined input directory."
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

    rad_tran_combined_indir: str = os.path.normpath(args.rad_tran_combined_indir)
    rad_tran_separate_outdir: str = os.path.normpath(args.rad_tran_separate_outdir)
    rad_tran_combined_outdir: str = os.path.normpath(args.rad_tran_combined_outdir)

    coarse_factors: Optional[NP_ARRAY[NP_INT]] = None
    if args.coarse_factors is not None:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]

    #---------------------------------------------------------------------------
    # Collect RTE-RRTMGP-CPP separate output files
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Getting relevant RTE-RRTMGP-CPP+RT input file paths.".format(current_time)
    print(msg, flush = True)

    rad_tran_separate_outdir_list: list[str] = os.listdir(rad_tran_separate_outdir)
    rad_tran_separate_outfiles: dict = {}

    if coarse_factors is None:
        coarse_factors: list[NP_INT] = []
        file_name: str
        for file_name in rad_tran_separate_outdir_list:
            if ".out.nc" in file_name:
                coarse_factor_str: str = re.search(r'lr_(\d{2})', file_name).group()
                coarse_factor: NP_INT = NP_INT(coarse_factor_str[3:])
                coarse_factors.append(coarse_factor)
        coarse_factors: NP_ARRAY[NP_INT] = np.array(coarse_factors, dtype = NP_INT)
    coarse_factors = np.sort(coarse_factors)[::-1]
    
    coarse_factor: NP_INT
    for coarse_factor in coarse_factors:
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)
        rad_tran_separate_outfiles[coarse_factor_str] = []
        for file_name in rad_tran_separate_outdir_list:
            if coarse_factor_str in file_name:
                rad_tran_separate_outfiles[coarse_factor_str] += [os.path.join(rad_tran_separate_outdir, file_name)]
        rad_tran_separate_outfiles[coarse_factor_str] = sorted(rad_tran_separate_outfiles[coarse_factor_str])

    #---------------------------------------------------------------------------
    # Loop through coarsening factors
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Looping through coarsening factors...".format(current_time)
    print(msg, flush = True)

    log_mem("Pre-Coarse Factor Loop")

    coarse_factor: NP_INT
    for coarse_factor in coarse_factors:
        #-----------------------------------------------------------------------
        # Set up combined RTE-RRTMGP-CPP infile
        #-----------------------------------------------------------------------
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, coarse_factor_str)
        print(msg, flush = True)

        rad_tran_combined_infile_name: str = [file_name for file_name in os.listdir(rad_tran_combined_indir) if coarse_factor_str in file_name][0]
        rad_tran_combined_infile_path: str = os.path.join(rad_tran_combined_indir, rad_tran_combined_infile_name)
        with xr.open_dataset(rad_tran_combined_infile_path, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran_combined_in:
            time: XR_DATARRAY = xr_rad_tran_combined_in["time"].load()
        ntime: NP_INT = NP_INT(time.size)

        log_mem("Pre-Output Opening")

        xr_rad_tran_separate_out: list[XR_DATASET] = [
            (xr.open_dataset(rad_tran_separate_outfiles[coarse_factor_str][ii],
                engine = "netcdf4",
                drop_variables = ["p_lay", "p_lev", 
                    "tot_tau", "tot_ssa", 
                    "cld_tau", "cld_ssa", "cld_asy",
                    "aer_tau", "aer_ssa", "aer_asy",
                    "sw_gpt_flux_up", "sw_gpt_flux_dn", "sw_gpt_flux_dn_dir", "sw_gpt_flux_net"])
            .expand_dims(time = [NP_REAL(time[ii])]))
            for ii in range(0, ntime)]
        for ii in range(0, ntime):
            xr_rad_tran_separate_out[ii]["time"].attrs.update(time.attrs)

        log_mem("Post-Output Opening")

        xr_rad_tran_combined_out: XR_DATASET = xr.combine_by_coords(xr_rad_tran_separate_out, data_vars = "all")

        log_mem("Post-Output Combining")
        
        rad_tran_combined_outfile_name: str = re.sub(".t_...", "", 
            os.path.basename(rad_tran_separate_outfiles[coarse_factor_str][0]))
        rad_tran_combined_outfile_path: str = os.path.join(rad_tran_combined_outdir,
            rad_tran_combined_outfile_name)

        xr_rad_tran_combined_out.to_netcdf(rad_tran_combined_outfile_path)

        log_mem("Post-Output Saving")

if __name__ == "__main__":
    main()