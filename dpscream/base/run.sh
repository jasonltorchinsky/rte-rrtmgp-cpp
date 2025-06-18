#! /bin/sh


# ASSUME: Python libraries installed
ROOT_DIR="../.."
BUILD_DIR=${ROOT_DIR}/"build"
VIZ_DIR=${ROOT_DIR}/"viz"
DATA_DIR=${ROOT_DIR}/"data"
RRTMGP_DATA_DIR=${ROOT_DIR}/"rrtmgp-data"
DPSCREAM_DATA_DIR=${ROOT_DIR}/"dpscream-data"

DPSCREAM_FILE_NAME_ROOT="scream_dpxx_RICO.scream.INSTANT.nhours_x1.2004-12-16-00000"
DPSCREAM_FILE_NAME=${DPSCREAM_FILE_NAME_ROOT}".nc"
DPSCREAM_FILE_PATH=${DPSCREAM_DATA_DIR}/${DPSCREAM_FILE_NAME}

PARAMETER_FILE="parameters.json"
OUT_DIR=${DPSCREAM_FILE_NAME_ROOT}
INPUT_FILE=${OUT_DIR}/${DPSCREAM_FILE_NAME_ROOT}".in.nc"
OUTPUT_FILE=${OUT_DIR}/${DPSCREAM_FILE_NAME_ROOT}".out.nc"

DEFAULT_INPUT_FILE="rte_rrtmgp_input.nc"
DEFAULT_OUTPUT_FILE="rte_rrtmgp_output.nc"

OUT_VIZ_DIR=${OUT_DIR}/"viz"
INPUT_VIZ_DIR=${OUT_VIZ_DIR}/"input"
OUTPUT_VIZ_DIR=${OUT_VIZ_DIR}/"output"
COMPARISON_VIZ_DIR=${OUT_VIZ_DIR}/"comparison"

## Create subdirectories
mkdir -p ${OUT_DIR}
mkdir -p ${OUT_VIZ_DIR}
mkdir -p ${INPUT_VIZ_DIR}
mkdir -p ${OUTPUT_VIZ_DIR}
mkdir -p ${COMPARISON_VIZ_DIR}

## Link netCDF data files from ROOT/data and ROOT/rrtmgp-data

TIME="[$(date '+%T')]"
printf "${TIME}: LINKING netCDF DATA FILES...\n\n"

ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-clouds-sw.nc cloud_coefficients_sw.nc
ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-clouds-lw.nc cloud_coefficients_lw.nc
ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-gas-sw-g224.nc coefficients_sw.nc
ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-gas-lw-g256.nc coefficients_lw.nc
ln -sf ${DATA_DIR}/aerosol_optics.nc
ln -sf ${DATA_DIR}/mie_lut_broadband.nc

TIME="[$(date '+%T')]"
printf "${TIME}: LINKED netCDF DATA FILES\n\n"

## Create input file

TIME="[$(date '+%T')]"
printf "${TIME}: CONVERTING DPSCREAM OUTPUT TO RTE-RRTMGP-CPP INPUT...\n\n"

eval 'python test_dpscream_input.py --input "${DPSCREAM_FILE_PATH}" '\
     '--output "${INPUT_FILE}" '
ln -sf ${INPUT_FILE} ${DEFAULT_INPUT_FILE}

TIME="[$(date '+%T')]"
printf "${TIME}: CONVERTED DPSCREAM OUTPUT TO RTE-RRTMGP-CPP INPUT\n\n"

## Visualize input

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZING ATMOSPHERE STATE...\n\n"

eval 'python "${VIZ_DIR}"/plot_input.py --input "${INPUT_FILE}" --outdir "${INPUT_VIZ_DIR}"'

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZED ATMOSPHERE STATE\n\n"

# ASSUME: Executable built in 'build' directory
## Run RT executable

TIME="[$(date '+%T')]"
printf "${TIME}: RUNNING RTE+RRTMGP-CPP...\n\n"

bsub -I -n 1 -W 00:10 -gpu num=1 ${BUILD_DIR}/test_rte_rrtmgp_rt_gpu --cloud-optics
mv ${DEFAULT_OUTPUT_FILE} ${OUTPUT_FILE}
ln -sf ${OUTPUT_FILE} ${DEFAULT_OUTPUT_FILE}

TIME="[$(date '+%T')]"
printf "${TIME}: RTE+RRTMGP-CPP COMPLETE\n\n"

## Visualize output

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZING OUTPUT...\n\n"

eval 'python "${VIZ_DIR}"/plot_output.py --input "${INPUT_FILE}" '\
     '--output "${OUTPUT_FILE}" --outdir "${OUTPUT_VIZ_DIR}"'

eval 'python "${VIZ_DIR}"/plot_comparison.py --input "${INPUT_FILE}" '\
     '--output "${OUTPUT_FILE}" --outdir "${COMPARISON_VIZ_DIR}" '

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZED OUTPUT\n\n"
