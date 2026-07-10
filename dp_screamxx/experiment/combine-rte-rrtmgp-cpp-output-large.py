# Standard Library Imports
import gc
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

from netCDF4 import Dataset

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY, \
    NC_DATASET, NC_REAL, NC_DIMENSION, NC_VARIABLE

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
        time_data: NP_ARRAY[NP_REAL] = NP_REAL(time.to_numpy())
        time_attrs: dict = dict(time.attrs)
        ntime: NP_INT = NP_INT(time.size)

        #-----------------------------------------------------------------------
        # Set up combined RTE-RRTMGP-CPP outfile
        #-----------------------------------------------------------------------
        rad_tran_combined_outfile_name: str = re.sub(".t_...", "", 
            os.path.basename(rad_tran_separate_outfiles[coarse_factor_str][0]))
        rad_tran_combined_outfile_path: str = os.path.join(rad_tran_combined_outdir,
            rad_tran_combined_outfile_name)

        log_mem("Pre-Combined Output Initialization")

        #-----------------------------------------------------------------------
        # Create base combined output file 
        #-----------------------------------------------------------------------
        keep_variables: list[str] = ["x", "y", "z",
            "sw_flux_up", "sw_flux_dn",
            "rt_flux_tod_up", "rt_flux_sfc_dir", "rt_flux_sfc_dif", "rt_flux_sfc_up",
            "rt_flux_abs_dir", "rt_flux_abs_dif"]

        rad_tran_separate_outfile_paths: list[str] = rad_tran_separate_outfiles[coarse_factor_str]

        with NC_DATASET(rad_tran_separate_outfile_paths[0], "r") as nc_src, \
            NC_DATASET(rad_tran_combined_outfile_path, "w", format = "NETCDF4") as nc_dst:
            nc_src.set_auto_maskandscale(False)

            #-------------------------------------------------------------------
            # Copy global attributes from first file.
            #-------------------------------------------------------------------
            for attr_name in nc_src.ncattrs():
                nc_dst.setncattr(attr_name, nc_src.getncattr(attr_name))

            #-------------------------------------------------------------------
            # Create dimensions / copy dimensions from first file.
            #-------------------------------------------------------------------
            nc_dst.createDimension("time", ntime)

            dim_name: str
            dim: NC_DIMENSION
            for dim_name, dim in nc_src.dimensions.items():
                if dim_name != "time":
                    dim_len: Optional[NP_INT] = None
                    if not dim.isunlimited():
                        dim_len = len(dim)
                    
                    nc_dst.createDimension(dim_name, len(dim))

            #-------------------------------------------------------------------
            # Create time variable.
            #-------------------------------------------------------------------
            time_var: NC_VARIABLE = nc_dst.createVariable("time", NC_REAL, ("time",))
            time_var[:] = time_data

            attr_name: str
            attr_value: str
            for attr_name, attr_value in time_attrs.items():
                time_var.setncattr(attr_name, attr_value)

            #-------------------------------------------------------------------
            # Create output variables.
            #-------------------------------------------------------------------
            out_vars: dict = {}

            var_name: str
            src_var: NC_VARIABLE
            for var_name, src_var in nc_src.variables.items():
                if (var_name in keep_variables):
                    # Get fill value, if present. Must be supplied at variable creation.
                    fill_value: Optional[NP_REAL] = None
                    if "_FillValue" in src_var.ncattrs():
                        fill_value = src_var.getncattr("_FillValue")

                    # Coordinate variable: e.g., lev(lev), col(col), etc.
                    # Copy once without adding time.
                    is_coordinate_var: bool = (
                        var_name in nc_src.dimensions
                        and src_var.dimensions == (var_name,)
                    )

                    if is_coordinate_var:
                        dst_var: NC_VARIABLE = nc_dst.createVariable(
                            var_name,
                            src_var.datatype,
                            src_var.dimensions,
                            fill_value = fill_value,
                        )

                        for attr_name in src_var.ncattrs():
                            if attr_name == "_FillValue":
                                continue
                            dst_var.setncattr(attr_name, src_var.getncattr(attr_name))

                        dst_var[:] = src_var[:]

                        continue

                    # Normal data variable: add leading time dimension.
                    out_dims = ("time",) + src_var.dimensions

                    # Chunking aligned with one-timestep writes.
                    # This is optional but often helpful.
                    chunksizes: Optional[tuple] = None
                    try:
                        chunksizes = (1,) + tuple(len(src0.dimensions[d]) for d in src_var.dimensions)
                    except Exception:
                        chunksizes = None

                    dst_var: NC_VARIABLE
                    if chunksizes is not None:
                        dst_var = nc_dst.createVariable(
                            var_name,
                            src_var.datatype,
                            out_dims,
                            fill_value = fill_value,
                            zlib = False,
                            chunksizes = chunksizes,
                        )
                    else:
                        dst_var = nc_dst.createVariable(
                            var_name,
                            src_var.datatype,
                            out_dims,
                            fill_value = fill_value,
                            zlib = False,
                        )

                    for attr_name in src_var.ncattrs():
                        if attr_name == "_FillValue":
                            continue
                        dst_var.setncattr(attr_name, src_var.getncattr(attr_name))

                    out_vars[var_name] = dst_var

            #--------------------------------------------------------------------------
            # Stream one input file at a time.
            #--------------------------------------------------------------------------
            for ii, file_path in enumerate(rad_tran_separate_outfile_paths):
                current_time: str = datetime.now().strftime("%H:%M:%S")
                msg: str = "[{}]: Streaming {}, file {} of {}...".format(current_time, coarse_factor_str, ii, ntime)
                print(msg, flush = True)

                with Dataset(file_path, "r") as src_i:
                    src_i.set_auto_maskandscale(False)

                    for var_name, dst_var in out_vars.items():
                        if var_name not in src_i.variables:
                            raise RuntimeError(
                                f"Variable {var_name} missing from {file_path}"
                            )

                        src_var = src_i.variables[var_name]

                        dst_var[ii, ...] = src_var[...]

                if ii % 5 == 0:
                    log_mem(f"Post-Streaming ii = {ii}")
                    gc.collect()

            nc_dst.sync()

    log_mem("Post-Streaming-Output")

if __name__ == "__main__":
    main()