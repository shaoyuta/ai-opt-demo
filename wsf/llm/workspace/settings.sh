#!/usr/bin/bash

#WL Setting
WL_PATH=/home/taosy/llm/1-c4709ac180e/frameworks.ai.pytorch.ipex-cpu/examples/cpu/inference/python/llm

OPTION=${1:-inference_latency_amx_bfloat16_pkm}
PLATFORM=${PLATFORM:-SPR}
WORKLOAD=${WORKLOAD:-llama2_pytorch_dev}
TOPOLOGY="Llama2"
FUNCTION=$(echo ${OPTION}|cut -d_ -f1)
MODE=$(echo ${OPTION}|cut -d_ -f2)
PRECISION=$(echo ${OPTION}|cut -d_ -f3-4)
CASE_TYPE=$(echo ${OPTION}|cut -d_ -f5)
DATA_TYPE="real"
BATCH_SIZE=${BATCH_SIZE:-1}
STEPS=${STEPS:-100}
INPUT_TOKENS=${INPUT_TOKENS:-32}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-32}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}
GREEDY=${GREEDY:-False}
NUMA_NODES_USE=${NUMA_NODES_USE:-0}
USE_DEEPSPEED=${USE_DEEPSPEED:-False}
USE_KERNEL_INJECT=${USE_KERNEL_INJECT:-False}
ONEDNN_VERBOSE=${ONEDNN_VERBOSE:-0}
RANK_USE=${RANK_USE:-0}
BENCHMARKING_TRACE=${BENCHMARKING_TRACE:-True}

TARGET_PLATFORM=${PLATFORM}
WARMUP_STEPS=$(($STEPS/10))


ALL_KEYS="MODE WORKLOAD TOPOLOGY HARDWARE PRECISION NUMA_NODES_USE FUNCTION DATA_TYPE BATCH_SIZE INPUT_TOKENS OUTPUT_TOKENS STEPS GREEDY WARMUP_STEPS MODEL_NAME MODEL_PATH USE_DEEPSPEED USE_KERNEL_INJECT ONEDNN_VERBOSE TARGET_PLATFORM MODEL_SIZE RANK_USE PLATFORM WL_PATH"

function WL_setenv()
{
    export WL_settings="1"
    for v in ${ALL_KEYS};
    do
        export ${v}
    done
}

function WL_unsetenv()
{
    echo "==== Cleanup WL env"
    unset WL_settings
    for v in ${ALL_KEYS};
    do
        unset ${v}
    done
    unset ALL_KEYS
}
