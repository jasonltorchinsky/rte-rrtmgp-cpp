# Standard Library Imports
import glob
import os
import re

# Third-Party Library Imports

# Local imports
from consts.dtypes import NP_INT, NP_ARRAY

"""
Find pairs of input and output files and reverse-sort by resolution
"""
def find_inout_pairs(rad_tran_indir: str, rad_tran_outdir: str, 
    coarse_factors: Optional[NP_ARRAY[NP_INT]] = None) -> list[list[str], list[str]]:
    
    rad_tran_infiles: list[str] = sorted(glob.glob(os.path.join(rad_tran_indir, "*.in.nc")), reverse = True)
    rad_tran_outfiles: list[str] = sorted(glob.glob(os.path.join(rad_tran_outdir, "*.out.nc")), reverse = True)

    # Get list of coarse factor strings
    coarse_strs: list[str]
    if coarse_factors is not None:
        coarse_strs = sorted(["lr_{:02}".format(coarse_factors[ii]) for ii in range(0, coarse_factors.size)], reverse = True)
    else:
        lr_re: re.Pattern = re.compile("lr_..")

        coarse_strs = []

        ii: int
        for ii in range(0, len(rad_tran_infiles)):
            rad_tran_infile_name: str = os.path.basename(rad_tran_infiles[ii])

            lr_Match: Optional[re.Match] = lr_re.search(rad_tran_infile_name)
            if lr_Match is not None:
                coarse_strs += [lr_Match.group()]

    paired_rad_tran_infiles: list[str] = []
    paired_rad_tran_outfiles: list[str] = []

    ii: int
    for ii in range(0, len(rad_tran_infiles)):
        rad_tran_infile_name: str = os.path.basename(rad_tran_infiles[ii])
        ext_re: re.Pattern = re.compile(".in.nc")
        rad_tran_infile_base: str = re.sub(ext_re, "", rad_tran_infile_name)

        coarse_str: str
        for coarse_str in coarse_strs:
            if coarse_str in rad_tran_infile_name:
                jj: int
                for jj in range(0, len(rad_tran_outfiles)):
                    if rad_tran_infile_base in rad_tran_outfiles[jj]:
                        paired_rad_tran_infiles += [rad_tran_infiles[ii]]
                        paired_rad_tran_outfiles += [rad_tran_outfiles[jj]]
                        break

    paired_rad_tran_infiles = sorted(paired_rad_tran_infiles, reverse = True)
    paired_rad_tran_outfiles = sorted(paired_rad_tran_outfiles, reverse = True)

    return [paired_rad_tran_infiles, paired_rad_tran_outfiles]