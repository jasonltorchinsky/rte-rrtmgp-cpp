#! /bin/sh

## Link netCDF data files

TIME="[$(date '+%T')]"
printf "${TIME}: LINKING netCDF DATA FILES...\n\n"

ln -sf ../rrtmgp-data/rrtmgp-clouds-sw.nc cloud_coefficients_sw.nc
ln -sf ../rrtmgp-data/rrtmgp-clouds-lw.nc cloud_coefficients_lw.nc
ln -sf ../rrtmgp-data/rrtmgp-gas-sw-g224.nc coefficients_sw.nc
ln -sf ../rrtmgp-data/rrtmgp-gas-lw-g256.nc coefficients_lw.nc
ln -sf ../data/aerosol_optics.nc
ln -sf ../data/mie_lut_broadband.nc

TIME="[$(date '+%T')]"
printf "${TIME}: LINKED netCDF DATA FILES\n\n"

# ASSUME: Python libraries installed
## Create input file

TIME="[$(date '+%T')]"
printf "${TIME}: CREATING ATMOSPHERE STATE INPUT FILE...\n\n"

python test_sandbox_input.py

TIME="[$(date '+%T')]"
printf "${TIME}: CREATED ATMOSPHERE STATE INPUT FILE\n\n"

## Visualize input

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZING ATMOSPHERE STATE...\n\n"

#python plot_input.py

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZED ATMOSPHERE STATE\n\n"

# ASSUME: Executable built in 'build' directory
## Run RT executable

TIME="[$(date '+%T')]"
printf "${TIME}: RUNNING RTE-RRTMGP-CPP WITH RAY-TRACING...\n\n"

bsub -I -n 1 -W 00:10 -gpu num=1 ../build/test_rte_rrtmgp_rt_gpu --cloud-optics

TIME="[$(date '+%T')]"
printf "${TIME}: RTE_RRTMGP-CPP WITH RAY-TRACING COMPLETE\n\n"

## Visualize output

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZING OUTPUT...\n\n"

#python plot_output.py
python plot_comparison.py

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZED OUTPUT\n\n"
