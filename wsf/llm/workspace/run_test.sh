#!/bin/bash -e

DIR="$( cd "$( dirname "$0" )" &> /dev/null && pwd )"

trap WL_unsetenv EXIT

source settings.sh
WL_setenv

export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
# export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

echo "WL_PATH: ${WL_PATH}"
echo "PLATFORM: ${TARGET_PLATFORM}"
echo "RECIPE_TYPE: ${RECIPE_TYPE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "MODE: ${MODE}"
echo "STEPS: ${STEPS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "PRECISION: ${PRECISION}"
echo "INPUT_TOKENS: ${INPUT_TOKENS}"
echo "OUTPUT_TOKENS: ${OUTPUT_TOKENS}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RANK_USE: ${RANK_USE}"
echo "USE_DEEPSPEED": ${USE_DEEPSPEED}

if [ "$ONEDNN_VERBOSE" == "1" ]; then
    export ONEDNN_VERBOSE=1
fi

#pytorch version
python -c "import torch; print(\"torch.version: \"+torch.__version__)"

# Set ISA
if [ "${TARGET_PLATFORM}" == "SRF" ]; then
    export ONEDNN_MAX_CPU_ISA="AVX2_VNNI_2"
else
    ISA=$(echo ${PRECISION} | cut -d_ -f1)
    REAL_PRECISION=$(echo ${PRECISION} | cut -d_ -f2)
    if [ "${ISA}" == "avx" ]; then
        export ONEDNN_MAX_CPU_ISA="AVX512_CORE_VNNI"
        if [ "${PRECISION}" == "avx_int8" ]; then
            export ATEN_CPU_CAPABILITY="avx512_vnni"
            export _DNNL_GRAPH_DISABLE_COMPILER_BACKEND="1"
        fi
    elif [ "${PRECISION}" == "amx_bfloat16" ] || [ "${PRECISION}" == "amx_bfloat32" ] || [ "${PRECISION}" == "amx_int8" ]; then
        export ONEDNN_MAX_CPU_ISA="AVX512_CORE_AMX"
    elif [ "${PRECISION}" == "amx_fp16" ]; then
        export ONEDNN_MAX_CPU_ISA="AVX512_CORE_AMX_FP16"
        export DNNL_MAX_CPU_ISA="AVX512_CORE_AMX_FP16"
    else
        echo "Not support precision ${PRECISION}."
        exit 1
    fi
fi

export HF_HOME="${HOME}/.cache/huggingface/"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=${TRANSFORMERS_CACHE}

# WL specific settings
if [ ! -h ${WL_PATH}/prompt.json ]; then
    ln -s ${DIR}/../common/prompt.json ${WL_PATH}/prompt.json
fi

if [[ "${USE_DEEPSPEED}" == "True" ]]; then
    ./run_test_deepspeed.sh
else
    ./run_test_general.sh
fi

