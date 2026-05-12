import glob
import os
import re

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