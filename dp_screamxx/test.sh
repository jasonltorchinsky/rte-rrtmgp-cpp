# Translate DP-SCREAMXX output to RTE-RRTMGP-CPP input
DP_SCREAMXX_DIR=/global/cfs/cdirs/m4815/ml3drt/dp_screamxx
DP_SCREAMXX_RUN_NAME=scream_dpxx_GATEIII_20x20
DP_SCREAMXX_RUN_DIR=${DP_SCREAMXX_DIR}/${DP_SCREAMXX_RUN_NAME}/run
SCREAM_OUTPUT_FILE_BASE=scream.INSTANT.nhours_x1.1974-08-30-00000
SCREAM_OUTPUT_FILE_PATH=${DP_SCREAMXX_RUN_DIR}/${DP_SCREAMXX_RUN_NAME}.${SCREAM_OUTPUT_FILE_BASE}.nc

RTE_RRTMGP_DIR=/global/cfs/cdirs/m4815/ml3drt/rte_rrtmgp_cpp
RTE_RRTMGP_RUN_NAME=t_360
RTE_RRTMGP_RUN_DIR=${RTE_RRTMGP_DIR}/${DP_SCREAMXX_RUN_NAME}/${RTE_RRTMGP_RUN_NAME}
RTE_RRTMGP_INPUT_DIR=${RTE_RRTMGP_RUN_DIR}/input
RTE_RRTMGP_OUTPUT_DIR=${RTE_RRTMGP_RUN_DIR}/output
RTE_RRTMGP_VIZ_DIR=${RTE_RRTMGP_RUN_DIR}/viz
RTE_RRTMGP_VIZ_INPUT_DIR=${RTE_RRTMGP_VIZ_DIR}/input
RTE_RRTMGP_FILE_BASE=${DP_SCREAMXX_RUN_NAME}.${SCREAM_OUTPUT_FILE_BASE}
RTE_RRTMGP_INPUT_FILE_PATH_BASE=${RTE_RRTMGP_INPUT_DIR}/${RTE_RRTMGP_FILE_BASE}

mkdir -p ${RTE_RRTMGP_INPUT_DIR}
mkdir -p ${RTE_RRTMGP_OUTPUT_DIR}
mkdir -p ${RTE_RRTMGP_VIZ_DIR}

# Set SZAs, times, and coarsening factors
SZAS_PYTHON="[0, 85]"
COARSE_FACTORS_PYTHON="[4]"
TIMES_PYTHON="[360]"
T0_PYTHON=-10
TF_PYTHON=-1

mpirun -np 8 python exp_hres/convert_dp_screamxx_output_new.py \
    --dpscream_file_path ${SCREAM_OUTPUT_FILE_PATH} \
    --rte_rrtmgp_cpp_dir_path ${RTE_RRTMGP_INPUT_DIR} \
    --coarse_factors "${COARSE_FACTORS_PYTHON}" \
    --szas "${SZAS_PYTHON}" \
    --times "${TIMES_PYTHON}"

#mpirun -np 8 python exp_hres/viz_tools/plot_input.py \
#    --rte_rrtmgp_cpp_input_dir_path ${RTE_RRTMGP_INPUT_DIR} \
#    --rte_rrtmgp_cpp_viz_dir_path ${RTE_RRTMGP_VIZ_DIR}
#mpirun -np 1 python exp_hres/viz_tools/plot_output.py \
#    --rte_rrtmgp_cpp_input_dir_path ${RTE_RRTMGP_INPUT_DIR} \
#    --rte_rrtmgp_cpp_output_dir_path ${RTE_RRTMGP_OUTPUT_DIR} \
#    --rte_rrtmgp_cpp_viz_dir_path ${RTE_RRTMGP_VIZ_DIR}
#mpirun -np 8 python exp_hres/viz_tools/plot_comparison.py \
#    --rte_rrtmgp_cpp_input_dir_path ${RTE_RRTMGP_INPUT_DIR} \
#    --rte_rrtmgp_cpp_output_dir_path ${RTE_RRTMGP_OUTPUT_DIR} \
#    --rte_rrtmgp_cpp_viz_dir_path ${RTE_RRTMGP_VIZ_DIR}