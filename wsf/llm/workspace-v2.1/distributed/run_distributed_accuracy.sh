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

ARGS="--batch-size=${BATCH_SIZE} \
    --model ${MODEL_NAME} \
    --ipex \
    --jit \
    --accuracy-only \
    --tasks "lambada_standard" "

echo "DS_TP: ${NUMA_NODES}"
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export LD_LIBRARY_PATH=${ONECCL_DIR}/lib:$LD_LIBRARY_PATH
EVAL_ARGS="deepspeed --num_gpus 2 --master_addr `hostname -I | sed -e 's/\s.*$//'` --bind_cores_to_rank"
EVAL_SCRIPT="run_accuracy_with_deepspeed.py"
if [ ${PRECISION} == "bfloat16" ]; then
    ARGS+=" --dtype bfloat16 "
elif [ ${PRECISION} == "woq_int8" ]; then
    ARGS+="--ipex-weight-only-quantization --int8-bf16-mixed"
else
    echo "This precision is not supported for DeepSpeed accuracy, please choose bfloat16 or woq_int8".
    exit 1
fi
echo "Start case topology"
echo "Run cmd: ${EVAL_ARGS} ${EVAL_SCRIPT} ${ARGS}"
eval ${EVAL_ARGS} ${EVAL_SCRIPT} ${ARGS}
echo "Finish case topology"