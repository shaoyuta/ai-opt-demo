#!/bin/bash -e

source activate llm

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

source /oneCCL/build/_install/env/setvars.sh
export WORK_DIR=./
unset KMP_AFFINITY

SOCKETS=`lscpu | grep "Socket(s)" | awk -F ':' '{print $2}'`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}'`
CORES_PER_NUMA=$(( $SOCKETS * $CORES_PER_SOCKET / $NUMA_NODES ))
echo "CORES_PER_INSTANCE: ${CORES_PER_NUMA}"

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
ARGS="--input-tokens=${INPUT_TOKENS} \
    --max-new-tokens ${OUTPUT_TOKENS} \
    --num-iter=${STEPS} \
    --num-warmup=${WARMUP_STEPS} \
    --batch-size=${BATCH_SIZE} \
    -m ${MODEL_NAME} --ipex \
    --benchmark --token-latency --deployment-mode "
if [ ${PRECISION} == "bfloat16" ]; then
    ARGS+="--dtype bfloat16 "
elif [ ${PRECISION} == "woq_int8" ]; then
    ARGS+="--ipex-weight-only-quantization --int8-bf16-mixed "
else
    echo "This precision is not supported for DeepSpeed, please choose bfloat16 or woq_int8".
    exit 1
fi

if [ ${GREEDY} == "True" ]; then
    echo "BEAM: 1"
    ARGS+=" --greedy"
else
    echo "BEAM: 4"
fi

echo "Start case topology"
echo "Run cmd: ${EVAL_ARGS} ${EVAL_SCRIPT} ${ARGS}"
eval ${EVAL_ARGS} ${EVAL_SCRIPT} ${ARGS}
echo "Finish case topology"