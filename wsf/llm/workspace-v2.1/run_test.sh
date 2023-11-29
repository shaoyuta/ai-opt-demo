#!/bin/bash -e

set -e

DIR="$( cd "$( dirname "$0" )" &> /dev/null && pwd )"
echo ${DIR}
trap WL_unsetenv EXIT

source ${DIR}/settings.sh
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
echo "MODEL_NAME: ${MODEL_NAME}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RANK_USE: ${RANK_USE}"
echo "USE_DEEPSPEED": ${USE_DEEPSPEED}
echo "MODEL_SIZE: ${MODEL_SIZE}"

if [ "$ONEDNN_VERBOSE" == "1" ]; then
    export ONEDNN_VERBOSE=1
fi

#pytorch version
python -c "import torch; print(\"torch.version: \"+torch.__version__)"

export HF_HOME="${HOME}/.cache/huggingface/"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=${TRANSFORMERS_CACHE}

# WL specific settings
if [ ! -e ${WL_PATH}/prompt.json ]; then
    ln -s ${DIR}/../common/prompt.json ${WL_PATH}/prompt.json
fi


# quantization
if [ ${PRECISION} == "woq_int8" ] || [ ${PRECISION} == "woq_int4" ] || [ ${PRECISION} == "static_int8" ]; then
    if [ ! -f "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}/best_model.pt" ]; then
        QUANT_ARGS="-m ${MODEL_NAME} --int8-bf16-mixed --output-dir "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}" "
        QUANT_SCRIPT="run_llama_quantization.py"
        if [ $PRECISION == "woq_int8" ]; then
            QUANT_ARGS+="--ipex-weight-only-quantization  "
        fi
        if [ $PRECISION == "static_int8" ]; then
            QUANT_ARGS+="--ipex-smooth-quant --alpha 0.5 --dataset lambada "
        fi
        if [ $PRECISION == "woq_int4" ]; then
            # Step 1: Generate modified weights and quantization info
            if [ ! -f "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}/gptq_checkpoint.pt" ]; then
                python utils/run_gptq.py --model ${MODEL_NAME} --output-dir ${TRANSFORMERS_CACHE}/saved_results_${PRECISION}
            fi
            # Step 3: Run quantized model for latency benchmark
            QUANT_ARGS+="--ipex-weight-only-quantization --low-precision-checkpoint "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}/gptq_checkpoint.pt"" 
        fi
        cd ./single_instance
        echo "Run quantization cmd: python ${QUANT_SCRIPT} ${QUANT_ARGS}"
        eval python ${QUANT_SCRIPT} ${QUANT_ARGS}
        cd ..
    fi
fi

if [ $USE_DEEPSPEED == "True" ]; then
#    cd ./distributed
    if [ $MODE == "accuracy" ]; then
        ./distributed/run_distributed_accuracy.sh
    else
        ./distributed/run_distributed.sh
    fi
else
#    cd ./single_instance
    if [ $MODE == "accuracy" ]; then
        ./single_instance/run_accuracy.sh
    else
        ./single_instance/run_single_instance.sh
    fi
fi