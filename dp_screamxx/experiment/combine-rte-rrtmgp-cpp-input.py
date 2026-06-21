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
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM, XR_DATASET, XR_DATAARRAY
from consts.numeric import MPI_ROOT

# Script variables
prog_name: str = "combine-rte-rrtmgp-cpp-input"
prog_desc: str = "Combine RTE-RRTMGP-CPP+RT input into a single time-series file."

def main():
    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Parsing command-line arguments.".format(current_time)
    print(msg, flush = True)
    
    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-separate-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT separate input directory."
    )
    parser.add_argument("--rad-tran-combined-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT combined input directory."
    )
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64."
    )
    args: Namespace = parser.parse_args()

    rad_tran_separate_indir: str = os.path.normpath(args.rad_tran_separate_indir)
    rad_tran_combined_indir: str = os.path.normpath(args.rad_tran_combined_indir)

    coarse_factors: Optional[NP_ARRAY[NP_INT]] = None
    if args.coarse_factors is not None:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]

    #---------------------------------------------------------------------------
    # Collect RTE-RRTMGP-CPP separate input files
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Getting relevant RTE-RRTMGP-CPP+RT input file paths.".format(current_time)
    print(msg, flush = True)

    rad_tran_separate_indir_list: list[str] = os.listdir(rad_tran_separate_indir)
    rad_tran_separate_infiles: dict = {}

    if coarse_factors is None:
        coarse_factors: list[NP_INT] = []
        file_name: str
        for file_name in rad_tran_separate_indir_list:
            if ".in.nc" in file_name:
                coarse_factor_str: str = re.search(r'lr_(\d{2})', file_name).group()
                coarse_factor: NP_INT = NP_INT(coarse_factor_str[3:])
                coarse_factors.append(coarse_factor)
        coarse_factors: NP_ARRAY[NP_INT] = np.array(coarse_factors, dtype = NP_INT)
    coarse_factors = np.sort(coarse_factors)[::-1]
    
    coarse_factor: NP_INT
    for coarse_factor in coarse_factors:
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)
        rad_tran_separate_infiles[coarse_factor_str] = []
        for file_name in rad_tran_separate_indir_list:
            if coarse_factor_str in file_name:
                rad_tran_separate_infiles[coarse_factor_str] += [os.path.join(rad_tran_separate_indir, file_name)]
        rad_tran_separate_infiles[coarse_factor_str] = sorted(rad_tran_separate_infiles[coarse_factor_str])

    #---------------------------------------------------------------------------
    # Loop through coarsening factors
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}]: Looping through coarsening factors...".format(current_time)
    print(msg, flush = True)

    coarse_factor: NP_INT
    for coarse_factor in coarse_factors:
        #-----------------------------------------------------------------------
        # Set up combined RTE-RRTMGP-CPP infile
        #-----------------------------------------------------------------------
        coarse_factor_str: str = "lr_{:02}".format(coarse_factor)

        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Processing {}...".format(current_time, coarse_factor_str)
        print(msg, flush = True)

        xr_rad_tran_combined_in: XR_DATASET = xr.open_mfdataset(rad_tran_separate_infiles[coarse_factor_str],
            data_vars = "all")
        
        rad_tran_combined_infile_name: str = re.sub(".t_...", "", 
            os.path.basename(rad_tran_separate_infiles[coarse_factor_str][0]))
        rad_tran_combined_infile_path: str = os.path.join(rad_tran_combined_indir,
            rad_tran_combined_infile_name)

        xr_rad_tran_combined_in.to_netcdf(rad_tran_combined_infile_path)

if __name__ == "__main__":
    main()