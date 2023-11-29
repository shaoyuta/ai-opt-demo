#!/bin/bash -e

SOCKETS=`lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}' | tr -d "[:space:]"`
CORES_PER_NUMA=$(( $SOCKETS * $CORES_PER_SOCKET / $NUMA_NODES ))
echo "CORES_PER_INSTANCE: ${CORES_PER_NUMA}"

# woq for precision int8
if [[ "${PRECISION}" == *"int8"* ]]; then
    if [[ "${PRECISION}" == *"avx"* ]]; then
        echo "Precision int8 only supports amx so far."
        exit 1
    fi
    # quantization execute argments
    QUANTIZATION_ARGS=" --ipex-weight-only-quantization --output-dir "${TRANSFORMERS_CACHE}/saved_results" --jit -m ${MODEL_NAME}"
    if [[ "${PRECISION}" == *"amx"* ]]; then
        QUANTIZATION_ARGS+=" --int8-bf16-mixed"
    fi
    # run quantization
    if [ ! -f "${TRANSFORMERS_CACHE}/saved_results/best_model.pt" ]; then
        mkdir -p ${TRANSFORMERS_CACHE}/saved_results
        echo "Run quantization: python run_llama_int8.py ${QUANTIZATION_ARGS}"
        eval python run_llama_int8.py ${QUANTIZATION_ARGS}
    fi
fi

# general args
if [ "${MODE}" == "accuracy" ]; then
    EXEC_ARGS="--device cpu \
               --accuracy-only \
               --model ${MODEL_NAME} \
               --tasks 'lambada_openai' \
               --jit"
else
    EXEC_ARGS=" --device cpu \
                --benchmark \
                --input-tokens=${INPUT_TOKENS} \
                --max-new-tokens ${OUTPUT_TOKENS} \
                --num-iter=${STEPS} \
                --num-warmup=${WARMUP_STEPS} \
                --batch-size=${BATCH_SIZE} \
                -m ${MODEL_NAME} --jit"
    if [ "$OUTPUT_TOKENS" != "1" ]; then
        EXEC_ARGS+=" --token-latency"
    fi
fi
# greedy or beam search
if [ "$GREEDY" == "True" ]; then
    EXEC_ARGS+=" --greedy"
    echo "BEAM: 1"
else
    echo "BEAM: 4" 
fi

# script name
if [ "${MODE}" == "accuracy" ]; then
    EVAL_SCRIPT="run_accuracy.py"
    if [[ "${PRECISION}" == *"int8"* ]]; then
        EXEC_ARGS+=" --dtype int8"
    fi
else
    if [[ "${PRECISION}" == *"int8"* ]]; then
        EVAL_SCRIPT="run_llama_int8.py"
    else
        EVAL_SCRIPT="run_generation.py"
    fi
fi
# precision args
if [[ "${PRECISION}" == *"bfloat16"* ]]; then
    EXEC_ARGS+=" --dtype bfloat16 --ipex"
elif [[ "${PRECISION}" == *"fp32"* ]]; then
    EXEC_ARGS+=" --dtype float32 --ipex"
elif [[ "${PRECISION}" == *"int8"* ]]; then
    EXEC_ARGS+=" --int8-bf16-mixed --quantized-model-path '${TRANSFORMERS_CACHE}/saved_results/best_model.pt'"
else
    echo "Error, not support precision ${PRECISION}"
    exit 1
fi

EVAL_SCRIPT=${WL_PATH}/${EVAL_SCRIPT}

# execute benchmarking script
echo "Start case topology"
start_core=$(( $NUMA_NODES_USE * $CORES_PER_NUMA ))
end_core=$(( $start_core + $CORES_PER_NUMA - 1 ))
NUMA_ARGS="OMP_NUM_THREADS=${CORES_PER_NUMA} numactl -m ${NUMA_NODES_USE} -C ${start_core}-${end_core}"
echo "Run benchmark: ${NUMA_ARGS} python ${EVAL_SCRIPT} ${EXEC_ARGS}"
eval ${NUMA_ARGS} python ${EVAL_SCRIPT} ${EXEC_ARGS}
echo "Finish case topology"

