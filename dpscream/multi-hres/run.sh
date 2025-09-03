#! /bin/sh


# ASSUME: Python libraries installed
ROOT_DIR="../.."
BUILD_DIR=${ROOT_DIR}/"build"
VIZ_DIR=${ROOT_DIR}/"viz"
DATA_DIR=${ROOT_DIR}/"data"
RRTMGP_DATA_DIR=${ROOT_DIR}/"rrtmgp-data"
DPSCREAM_DATA_DIR=${ROOT_DIR}/"dpscream-data"

DPSCREAM_TEST_NAME_PREFIX="scream_dpxx_RICO_doSW"
DPSCREAM_TEST_NAME_SUFFIX="scream.INSTANT.nhours_x1.2004-12-16-00000"

PARAMETER_FILE="parameters.json"
OUT_DIR=${DPSCREAM_TEST_NAME_PREFIX}
OUT_VIZ_DIR=${OUT_DIR}/viz
INPUT_FILE_ROOT=${OUT_DIR}/${DPSCREAM_TEST_NAME_PREFIX}
OUTPUT_FILE_ROOT=${OUT_DIR}/${DPSCREAM_TEST_NAME_PREFIX}

DEFAULT_INPUT_FILE="rte_rrtmgp_input.nc"
DEFAULT_OUTPUT_FILE="rte_rrtmgp_output.nc"

## Input variables
NX_SH="5 10 15 20"
NX_PYTHON="[5, 10, 15, 20]"
SZAS_SH="85 80 70 60 50 40 30 20 10 0"
SZAS_PYTHON="[85., 80., 70., 60., 50., 40., 30., 20., 10., 0.]"

## Create subdirectories
mkdir -p ${OUT_DIR}
mkdir -p ${OUT_VIZ_DIR}

### Resolution-specific driectories
for NX in ${NX_SH};
do
     PADDED_NX=$(printf "%02d" "${NX}")
     NX_OUT_DIR=${OUT_DIR}/${PADDED_NX}

     NX_OUT_VIZ_DIR=${NX_OUT_DIR}/"viz"
     NX_INPUT_VIZ_DIR=${NX_OUT_VIZ_DIR}/"input"
     NX_OUTPUT_VIZ_DIR=${NX_OUT_VIZ_DIR}/"output"
     NX_COMPARISON_VIZ_DIR=${NX_OUT_VIZ_DIR}/"comparison"
     
     mkdir -p ${NX_OUT_DIR}
     mkdir -p ${NX_OUT_VIZ_DIR}
     mkdir -p ${NX_INPUT_VIZ_DIR}
     mkdir -p ${NX_OUTPUT_VIZ_DIR}
     mkdir -p ${NX_COMPARISON_VIZ_DIR}
done

## Link netCDF data files from ROOT/data and ROOT/rrtmgp-data

TIME="[$(date '+%T')]"
printf "${TIME}: LINKING netCDF DATA FILES...\n\n"

#ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-clouds-sw.nc cloud_coefficients_sw.nc
#ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-clouds-lw.nc cloud_coefficients_lw.nc
#ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-gas-sw-g224.nc coefficients_sw.nc
#ln -sf ${RRTMGP_DATA_DIR}/rrtmgp-gas-lw-g256.nc coefficients_lw.nc
#ln -sf ${DATA_DIR}/aerosol_optics.nc
#ln -sf ${DATA_DIR}/mie_lut_broadband.nc

TIME="[$(date '+%T')]"
printf "${TIME}: LINKED netCDF DATA FILES\n\n"

## Create input file

TIME="[$(date '+%T')]"
printf "${TIME}: CONVERTING DPSCREAM OUTPUT TO RTE-RRTMGP-CPP INPUT...\n\n"

#for NX in ${NX_SH};
#do
#     PADDED_NX=$(printf "%02d" "${NX}")
#     NXxNX=${PADDED_NX}x${PADDED_NX}
#     DPSCREAM_FILE_ROOT=${DPSCREAM_TEST_NAME_PREFIX}_${NXxNX}.${DPSCREAM_TEST_NAME_SUFFIX}
#     DPSCREAM_FILE_ROOT_PATH=${DPSCREAM_DATA_DIR}/${DPSCREAM_FILE_ROOT}

#     RTE_RRTMGP_INPUT_FILE_ROOT=${DPSCREAM_FILE_ROOT}
#     RTE_RRTMGP_INPUT_FILE_BASE_PATH=${OUT_DIR}/${RTE_RRTMGP_INPUT_FILE_ROOT}

#     eval 'python test_multi_hres_input.py --input_root "${DPSCREAM_FILE_ROOT_PATH}" '\
#          '--output_root "${RTE_RRTMGP_INPUT_FILE_BASE_PATH}" --szas "${SZAS_PYTHON}" '
#done

TIME="[$(date '+%T')]"
printf "${TIME}: CONVERTED DPSCREAM OUTPUT TO RTE-RRTMGP-CPP INPUT\n\n"

## Visualize input

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZING ATMOSPHERE STATE...\n\n"

#for NX in ${NX_SH};
#do
#     PADDED_NX=$(printf "%02d" "${NX}")
#     NXxNX=${PADDED_NX}x${PADDED_NX}
#     DPSCREAM_FILE_ROOT=${DPSCREAM_TEST_NAME_PREFIX}_${NXxNX}.${DPSCREAM_TEST_NAME_SUFFIX}

#     NX_OUT_DIR=${OUT_DIR}/${PADDED_NX}
#     NX_OUT_VIZ_DIR=${NX_OUT_DIR}/"viz"
#     NX_INPUT_VIZ_DIR=${NX_OUT_VIZ_DIR}/"input"

#     RTE_RRTMGP_INPUT_FILE_ROOT=${DPSCREAM_FILE_ROOT}
#     RTE_RRTMGP_INPUT_FILE_BASE_PATH=${OUT_DIR}/${RTE_RRTMGP_INPUT_FILE_ROOT}

#     SZA=$(echo ${SZAS_SH} | awk '{print $1}')
#     PADDED_SZA=$(printf "%04d" "${SZA}")
#     INPUT_FILE=${RTE_RRTMGP_INPUT_FILE_BASE_PATH}.${PADDED_SZA}.in.nc
     
#     eval 'python "${VIZ_DIR}"/plot_input.py --input "${INPUT_FILE}" --outdir "${NX_INPUT_VIZ_DIR}"'
#done

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZED ATMOSPHERE STATE\n\n"

# ASSUME: Executable built in 'build' directory
## Run RT executable

TIME="[$(date '+%T')]"
printf "${TIME}: RUNNING RTE+RRTMGP-CPP...\n\n"

#for NX in ${NX_SH};
#do
#      PADDED_NX=$(printf "%02d" "${NX}")
#      NXxNX=${PADDED_NX}x${PADDED_NX}
#      DPSCREAM_FILE_ROOT=${DPSCREAM_TEST_NAME_PREFIX}_${NXxNX}.${DPSCREAM_TEST_NAME_SUFFIX}

#      RTE_RRTMGP_FILE_ROOT=${DPSCREAM_FILE_ROOT}
#      RTE_RRTMGP_FILE_BASE_PATH=${OUT_DIR}/${RTE_RRTMGP_FILE_ROOT}

#      for SZA in ${SZAS_SH};
#      do
#           PADDED_SZA=$(printf "%04d" "${SZA}")
#           INPUT_FILE=${RTE_RRTMGP_FILE_BASE_PATH}.${PADDED_SZA}.in.nc
#           echo ${INPUT_FILE}
#           ln -sf ${INPUT_FILE} ${DEFAULT_INPUT_FILE}
	
#           bsub -I -n 1 -W 00:10 -gpu num=1 ${BUILD_DIR}/test_rte_rrtmgp_rt_gpu --cloud-optics --raytracing 512

#           OUTPUT_FILE=${RTE_RRTMGP_FILE_BASE_PATH}.${PADDED_SZA}.out.nc
#           mv ${DEFAULT_OUTPUT_FILE} ${OUTPUT_FILE}
#      done
#done

TIME="[$(date '+%T')]"
printf "${TIME}: RTE+RRTMGP-CPP COMPLETE\n\n"

## Visualize output

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZING OUTPUT...\n\n"

#for NX in ${NX_SH};
#do
#      PADDED_NX=$(printf "%02d" "${NX}")
#      NXxNX=${PADDED_NX}x${PADDED_NX}
#      DPSCREAM_FILE_ROOT=${DPSCREAM_TEST_NAME_PREFIX}_${NXxNX}.${DPSCREAM_TEST_NAME_SUFFIX}

#      RTE_RRTMGP_FILE_ROOT=${DPSCREAM_FILE_ROOT}
#      RTE_RRTMGP_FILE_BASE_PATH=${OUT_DIR}/${RTE_RRTMGP_FILE_ROOT}

#      NX_OUT_DIR=${OUT_DIR}/${PADDED_NX}
#      NX_OUT_VIZ_DIR=${NX_OUT_DIR}/"viz"
#      NX_COMP_VIZ_DIR=${NX_OUT_VIZ_DIR}/"comparison"

#      for SZA in ${SZAS_SH};
#      do
#      	PADDED_SZA=$(printf "%04d" "${SZA}")
#      	INPUT_FILE=${RTE_RRTMGP_FILE_BASE_PATH}.${PADDED_SZA}.in.nc
#        OUTPUT_FILE=${RTE_RRTMGP_FILE_BASE_PATH}.${PADDED_SZA}.out.nc

#      	NX_OUT_VIZ_SZA_DIR=${NX_OUT_VIZ_DIR}/${PADDED_SZA}
#      	NX_COMP_VIZ_SZA_DIR=${NX_COMP_VIZ_DIR}/${PADDED_SZA}

#      	mkdir -p ${NX_OUT_VIZ_SZA_DIR}
#      	mkdir -p ${NX_COMP_VIZ_SZA_DIR}

#      	eval 'python "${VIZ_DIR}"/plot_output.py --input "${INPUT_FILE}" '\
#      	     '--output "${OUTPUT_FILE}" --outdir "${NX_OUT_VIZ_SZA_DIR}"'

#      	eval 'python "${VIZ_DIR}"/plot_comparison.py --input "${INPUT_FILE}" '\
#      	     '--output "${OUTPUT_FILE}" --outdir "${NX_COMP_VIZ_SZA_DIR}" '
#      done
#done

#for NX in ${NX_SH};
#do
#      PADDED_NX=$(printf "%02d" "${NX}")
#      NXxNX=${PADDED_NX}x${PADDED_NX}
#      DPSCREAM_FILE_ROOT=${DPSCREAM_TEST_NAME_PREFIX}_${NXxNX}.${DPSCREAM_TEST_NAME_SUFFIX}

#      RTE_RRTMGP_FILE_ROOT=${DPSCREAM_FILE_ROOT}
#      RTE_RRTMGP_FILE_BASE_PATH=${OUT_DIR}/${RTE_RRTMGP_FILE_ROOT}

#      NX_OUT_DIR=${OUT_DIR}/${PADDED_NX}
#      NX_OUT_VIZ_DIR=${NX_OUT_DIR}/"viz"
#      NX_COMP_VIZ_DIR=${NX_OUT_VIZ_DIR}/"comparison"

#      SZA=$(echo ${SZAS_SH} | awk '{print $1}')
#      PADDED_SZA=$(printf "%04d" "${SZA}")
#      INPUT_FILE=${RTE_RRTMGP_FILE_BASE_PATH}.${PADDED_SZA}.in.nc
#      eval 'python "${VIZ_DIR}"/plot_sza_comparison.py --input "${INPUT_FILE}" '\
#           ' --output "${RTE_RRTMGP_FILE_BASE_PATH}" --outdir "${NX_OUT_VIZ_DIR}" '\
#           ' --szas "${SZAS_PYTHON}" '

#      eval 'python "${VIZ_DIR}"/plot_sza_statistics.py --input "${INPUT_FILE}" '\
#           ' --output "${RTE_RRTMGP_FILE_BASE_PATH}" --outdir "${NX_OUT_VIZ_DIR}" '\
#           ' --szas "${SZAS_PYTHON}" '
#done

#eval 'python "${VIZ_DIR}"/plot_hres_sza_comparison.py --rteprefix "${INPUT_FILE_ROOT}" '\
#      ' --rtesuffix "${DPSCREAM_TEST_NAME_SUFFIX}" --nxs "${NX_PYTHON}" '\
#      ' --szas "${SZAS_PYTHON}" --outdir "${OUT_VIZ_DIR}" '

eval 'python "${VIZ_DIR}"/plot_hres_sza.py --rteprefix "${INPUT_FILE_ROOT}" '\
      ' --rtesuffix "${DPSCREAM_TEST_NAME_SUFFIX}" --nxs "${NX_PYTHON}" '\
      ' --szas "${SZAS_PYTHON}" --outdir "${OUT_VIZ_DIR}" '

TIME="[$(date '+%T')]"
printf "${TIME}: VISUALIZED OUTPUT\n\n"
