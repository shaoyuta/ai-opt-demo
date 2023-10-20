#!/bin/bash -e

SOCKETS=`lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}' | tr -d "[:space:]"`
CORES_PER_NUMA=$(( $SOCKETS * $CORES_PER_SOCKET / $NUMA_NODES ))
echo "CORES_PER_INSTANCE: ${CORES_PER_NUMA}"

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

#source /home/taosy/llm/1-c4709ac180e/oneCCL/build/_install/env/setvars.sh   #FIXME
unset KMP_AFFINITY
if [ "${MODE}" == "accuracy" ]; then
    echo "DS_TP: ${NUMA_NODES}"
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
    export LD_LIBRARY_PATH=${ONECCL_DIR}/lib:$LD_LIBRARY_PATH
    EVAL_ARGS="deepspeed  --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank"
    EVAL_SCRIPT="run_accuracy_with_deepspeed.py" 
else
    TOTAL_CORES=$((SOCKETS*CORES_PER_SOCKET))
    CORES_PER_NUMA=$((TOTAL_CORES/NUMA_NODES))
    RANKS_PER_SOCKET=$((CORES_PER_SOCKET/CORES_PER_NUMA))
    if [ "${RANK_USE}" == "0" ] || [ "${RANK_USE}" == "1" ]; then
        echo "DS_TP: ${RANKS_PER_SOCKET}"
        BIND_CORE_LIST=$((RANK_USE*CORES_PER_SOCKET))-$(((RANK_USE+1)*CORES_PER_SOCKET-1))
        EVAL_ARGS="deepspeed --num_accelerators ${RANKS_PER_SOCKET} --bind_cores_to_rank --bind_core_list ${BIND_CORE_LIST}"
    else
        echo "DS_TP: ${NUMA_NODES}"
        EVAL_ARGS="deepspeed --bind_cores_to_rank"
    fi
    EVAL_SCRIPT="run_generation_with_deepspeed.py"
    # kernel inject
    if [ "$USE_KERNEL_INJECT" == "True" ]; then
        EXEC_ARGS+=" --ki"
    fi
fi

EVAL_SCRIPT=${WL_PATH}/${EVAL_SCRIPT}

# execute parameters
if [ "${PRECISION}" == "amx_bfloat16" ]; then
    EXEC_ARGS+=" --dtype bfloat16 --ipex"
elif [ "${PRECISION}" == "amx_int8" ]; then
    EXEC_ARGS+=" --int8-bf16-mixed --ipex --ipex-weight-only-quantization"
else
    echo "Error, accuracy mode with deepspeed only supports precision amx_bfloat16 and amx_int8."
    exit 1
fi

echo "Start case topology"
echo "Run cmd: ${EVAL_ARGS} ${EVAL_SCRIPT} ${EXEC_ARGS}"
eval ${EVAL_ARGS} ${EVAL_SCRIPT} ${EXEC_ARGS}
echo "Finish case topology"

