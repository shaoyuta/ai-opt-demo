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
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RANK_USE: ${RANK_USE}"
echo "USE_DEEPSPEED": ${USE_DEEPSPEED}
echo "USE_IPEX": ${USE_IPEX}

if [ "$ONEDNN_VERBOSE" == "1" ]; then
    export ONEDNN_VERBOSE=1
fi

RUN_NUM_ITER=${RUN_NUM_ITER:-10}
RUN_NUM_WARMUP=${RUN_NUM_WARMUP:-1}

#pytorch version
python -c "import torch; print(\"torch.version: \"+torch.__version__)"


export HF_HOME="${HOME}/.cache/huggingface/"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=${TRANSFORMERS_CACHE}

#SCRIPT_PATH=${DIR}/../intel-extension-for-pytorch/examples/cpu/inference/python/llm/single_instance
SCRIPT_PATH=${DIR}


# for numa, begin
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
    -m ${MODEL_NAME}"

if [[ "${USE_IPEX}" = "True" ]]; then
    ARGS+=" --ipex"
fi
# for numa, end

#OMP_NUM_THREADS=18 numactl -m 0 -C 0-17 \
#python ${SCRIPT_PATH}/run_generation_m.py  \
#--benchmark  --input-tokens=32  --max-new-tokens 32  --num-iter=${RUN_NUM_ITER}  --num-warmup=${RUN_NUM_WARMUP}  --batch-size=1  \
#-m meta-llama/Llama-2-7b-chat-hf  --dtype bfloat16 --profile 

EXEC_SCRIPT=${SCRIPT_PATH}/run_generation_m.py
echo "Start case topology"
echo "Run cmd: ${EXEC_CMD} ${EXEC_SCRIPT} ${ARGS}"
eval ${EXEC_CMD} ${EXEC_SCRIPT} ${ARGS}
echo "Finish case topology"
