# Standard Library Imports
import os
import re
import shutil
import subprocess

from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import xarray as xr

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM, XR_DATASET
from consts.numeric import MPI_ROOT

# Script variables
prog_name: str = "run-rte-rrtmgp-cpp.py"
prog_desc: str = "Run RTE-RRTMGP-CPP+RT across multiple GPUs, per time-slice."

def main():
    #---------------------------------------------------------------------------
    # Set up MPI communicator
    #---------------------------------------------------------------------------
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    #---------------------------------------------------------------------------
    # Parse command-line input
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Parsing command-line arguments.".format(current_time)
        print(msg, flush = True)
    parser: ArgumentParser = ArgumentParser(prog = prog_name,
        description = prog_desc)
    parser.add_argument("--rad-tran-indir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT input directory."
    )
    parser.add_argument("--rad-tran-outdir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Path for RTE-RRTMGP-CPP+RT output directory."
    )
    parser.add_argument("--rad-tran-exec", action = "store",
        nargs = "?", type = str, required = True, 
        help = "Path for RTE-RRTMGP-CPP+RT executable."
    )
    parser.add_argument("--coarse-factors", action = "store",
        nargs = "?", type = str, required = False, default = None,
        help = "Coarsening factors to process, e.g., 1,2,8,64."
    )
    parser.add_argument("--rrtmgp-data-dir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Directory containing rrtmgp-clouds-*.nc and rrtmgp-gas-*.nc."
    )
    parser.add_argument("--rte-data-dir", action = "store",
        nargs = "?", type = str, required = True,
        help = "Directory containing aerosol_optics.nc."
    )
    parser.add_argument("--gpus", action = "store",
        nargs = "?", type = str, required = True,
        help = "GPUs to utilize, e.g., 0,1,2,7."
    )
    parser.add_argument("--raytracing", action = "store",
        nargs = "?", type = int, required = False, default = 128,
        help = "Number of rays-per-pixel: Default 128."
    )
    parser.add_argument("--work-dir", action = "store",
        nargs = "?", type = str, required = False, default = ".working",
        help = "Relative path for workers to store intermediate files."
    )
    args: Namespace = parser.parse_args()

    rad_tran_indir: str = os.path.normpath(args.rad_tran_indir)
    rad_tran_outdir: str = os.path.normpath(args.rad_tran_outdir)
    rad_tran_exec: str = os.path.normpath(args.rad_tran_exec)
    rrtmgp_data_dir: str = os.path.normpath(args.rrtmgp_data_dir)
    rte_data_dir: str = os.path.normpath(args.rte_data_dir)
    work_dir: str = os.path.join(rad_tran_outdir, os.path.normpath(args.work_dir))
    raytracing: int = args.raytracing
    gpus: NP_ARRAY[NP_INT] = np.sort(np.array(args.gpus.split(","), dtype = NP_INT))

    coarse_factors: Optional[NP_ARRAY[NP_INT]]
    if args.coarse_factors is None:
        coarse_factors = None
    else:
        coarse_factors = np.sort(np.array(args.coarse_factors.split(","), dtype = NP_INT))[::-1]

    assert(comm_size <= gpus.size)

    #---------------------------------------------------------------------------
    # Create directories that don't exist
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        dir_names: list[str] = [rad_tran_outdir, work_dir]
        dir_name: str
        for dir_name in dir_names:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    #---------------------------------------------------------------------------
    # Get RTE-RRTMGP-CPP+RT input files
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
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
    # Distribute work load across processes - ASSUME: time is same across
    # each coarsening
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Distributing workload.".format(current_time)
        print(msg, flush = True)
    nt: NP_INT = NP_INT(xr.open_dataset(rad_tran_infiles[0], engine = "netcdf4", decode_timedelta = False)["time"].size)
    rank_nts: NP_ARRAY[NP_INT] = (nt // comm_size) \
        + (np.arange(comm_size, dtype = NP_INT) < (nt - comm_size * (nt // comm_size)))
    l_time_st_idx: NP_INT = np.sum(rank_nts[:l_rank])
    l_time_idxs = np.arange(l_time_st_idx, l_time_st_idx + rank_nts[l_rank])

    #---------------------------------------------------------------------------
    # Each rank gets its own GPU, creates its own working directory,
    # creates necessary symbolic links
    #---------------------------------------------------------------------------
    current_time = datetime.now().strftime("%H:%M:%S")
    msg: str = "[{}], [Rank {}]: Setting up working directory.".format(current_time, l_rank)
    print(msg, flush = True)

    l_gpu: NP_INT = gpus[l_rank]
    l_env = os.environ.copy()
    l_env["CUDA_VISIBLE_DEVICES"] = str(l_gpu)

    l_work_dir: str = os.path.join(work_dir, str(l_rank))

    if not os.path.exists(l_work_dir):
        os.makedirs(l_work_dir)

    # Create symbolic links for executable
    rrtmgp_filenames_src: list[str] = ["rrtmgp-clouds-sw.nc", "rrtmgp-clouds-lw.nc",
        "rrtmgp-gas-sw-g224.nc", "rrtmgp-gas-lw-g256.nc"]
    rrtmgp_filenames_tgt: list[str] = ["cloud_coefficients_sw.nc", "cloud_coefficients_lw.nc",
        "coefficients_sw.nc", "coefficients_lw.nc"]
    for ii in range(0, len(rrtmgp_filenames_src)):
        if not os.path.exists(os.path.join(l_work_dir, rrtmgp_filenames_tgt[ii])):
            os.symlink(os.path.join(rrtmgp_data_dir, rrtmgp_filenames_src[ii]),
                os.path.join(l_work_dir, rrtmgp_filenames_tgt[ii]))

    rte_filenames_src: list[str] = ["aerosol_optics.nc", "mie_lut_broadband.nc"]
    rte_filenames_tgt: list[str] = ["aerosol_optics.nc", "mie_lut_broadband.nc"]
    for ii in range(0, len(rte_filenames_src)):
        if not os.path.exists(os.path.join(l_work_dir, rte_filenames_tgt[ii])):
            os.symlink(os.path.join(rte_data_dir, rte_filenames_src[ii]),
                os.path.join(l_work_dir, rte_filenames_tgt[ii]))

    #---------------------------------------------------------------------------
    # Loop through local queue and run RTE-RRTMGP-CPP+RT on the input
    # Have to move output file when process is done
    #---------------------------------------------------------------------------
    cmd: list[str] = [rad_tran_exec, "--cloud-optics", "--single-gpt", "--raytracing", str(raytracing)]
    stdout: str = os.path.join(l_work_dir, os.path.basename(rad_tran_exec) + ".out")
    stderr: str = os.path.join(l_work_dir, os.path.basename(rad_tran_exec) + ".err")
    for rad_tran_infile in rad_tran_infiles:
        for tt in l_time_idxs:
            current_time = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}], [Rank {}]: Processing {}, time index {}.".format(current_time, l_rank, os.path.basename(rad_tran_infile), tt)
            print(msg, flush = True)

            with (xr.open_dataset(rad_tran_infile, engine = "netcdf4", 
                decode_timedelta = False).isel(time = tt)) as xr_rad_tran_in:
                xr_rad_tran_in.to_netcdf(os.path.join(l_work_dir, "rte_rrtmgp_input.nc"))
            
            with open(stdout, "w") as f_out, open(stderr, "w") as f_err:
                proc: subprocess.Popen = subprocess.Popen(cmd, env = l_env,
                    cwd = l_work_dir, stdout = f_out, stderr = f_err,
                    text = True, bufsize = 1)

                current_time = datetime.now().strftime("%H:%M:%S")
                msg: str = "[{}], [Rank {}]: Waiting on subprocess {}, time index {}.".format(current_time, l_rank, os.path.basename(rad_tran_infile), tt)
                print(msg, flush = True)

                proc.wait()

            rte_rrtmgp_output_src: str = os.path.join(l_work_dir, "rte_rrtmgp_output.nc")
            rte_rrtmgp_output_tgt_filename: str = os.path.basename(rad_tran_infile)[:-6] + ".t_{:03}".format(tt) + ".out.nc"
            rte_rrtmgp_output_tgt: str = os.path.join(rad_tran_outdir, rte_rrtmgp_output_tgt_filename)
            shutil.move(rte_rrtmgp_output_src, rte_rrtmgp_output_tgt)

            current_time = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}], [Rank {}]: Output moved to {}.".format(current_time, l_rank, rte_rrtmgp_output_tgt_filename)
            print(msg, flush = True)

if __name__ == "__main__":
    main()