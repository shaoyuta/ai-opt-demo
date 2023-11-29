#!/bin/bash -e

set -xe

#source activate llm

export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
# Setup environment variables for performance on Xeon
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

SOCKETS=`lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}'`
CORES_PER_NUMA=$(( $SOCKETS * $CORES_PER_SOCKET / $NUMA_NODES ))

if [ "${CORES_PER_INSTANCE}" ] && [ "${CORES_PER_INSTANCE}" -lt "${CORES_PER_NUMA}" ] && [ "${CORES_PER_INSTANCE}" -gt "0" ]; then
    CORES_PER_INSTANCE=${CORES_PER_INSTANCE}
    echo "CORES_PER_INSTANCE: ${CORES_PER_INSTANCE}"
else
    CORES_PER_INSTANCE=${CORES_PER_NUMA}
    echo "CORES_PER_INSTANCE: ${CORES_PER_NUMA}"
fi
end_core=$(( $CORES_PER_INSTANCE - 1 ))

EXEC_CMD="OMP_NUM_THREADS=${CORES_PER_INSTANCE} numactl -m 0 -C 0-${end_core} python "
ARGS="--benchmark --input-tokens=${INPUT_TOKENS} \
    --max-new-tokens ${OUTPUT_TOKENS} \
    --num-iter=${STEPS} \
    --num-warmup=${WARMUP_STEPS} \
    --batch-size=${BATCH_SIZE} \
    -m ${MODEL_NAME} --token-latency "

# bf16 benchmark
if [ $PRECISION == "bfloat16" ]; then
    EXEC_SCRIPT="${WL_PATH}/run_generation.py"
    ARGS+="--benchmark --ipex --deployment-mode --dtype bfloat16"
# weight only quantization int8 benchmark
elif [ $PRECISION == "woq_int8" ]; then
    EXEC_SCRIPT="${WL_PATH}/run_llama_quantization.py"
    ARGS+="--quantized-model-path "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}/best_model.pt" --int8-bf16-mixed"
# weight only quantization int4 benchmark
elif [ $PRECISION == "woq_int4" ]; then
    EXEC_SCRIPT="${WL_PATH}/run_llama_quantization.py"
    ARGS+="--quantized-model-path "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}/best_model.pt" --int8-bf16-mixed"
# static quantization int8 benchmark
elif [ $PRECISION == "static_int8" ]; then
    EXEC_SCRIPT="${WL_PATH}/run_llama_quantization.py"
    ARGS+="--quantized-model-path "${TRANSFORMERS_CACHE}/saved_results_${PRECISION}/best_model.pt" --int8-bf16-mixed"
else
    echo "This precision is not supported, please choose from bfloat16, woq_int8, woq_int4 or static_int8"
    exit 1
fi

if [ ${GREEDY} == "True" ]; then
    echo "BEAM: 1"
    ARGS+=" --greedy"
else
    echo "BEAM: 4"
fi

echo "Start case topology"
echo "Run cmd: ${EXEC_CMD} ${EXEC_SCRIPT} ${ARGS}"
eval ${EXEC_CMD} ${EXEC_SCRIPT} ${ARGS}
echo "Finish case topology"