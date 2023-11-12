#!/bin/bash

set -e

# VARs
COMMAND=""
EMON_DATA_PATH="emon.dat"
STR_ROI_START=""
STR_ROI_END=""
PARSE_RESULT=false

trace_started=false

function usage() {
    cat << EOM
Usage: $(basename "$0") [OPTION]...
-f abs pathname of emon.dat, default is "./emon.dat"
-s string of start_roi
-e string of end_roi
-c command
-p parse result
-h usage
EOM
    exit 0
}

function validate_env(){
    if [[ ! $(command -v emon) ]]; then
        echo "emon not found"
        exit 1
    fi
}

function validate_args() {
    [[ -z ${COMMAND} ]] && echo "cmd is None" && exit 1
    [[ -z ${STR_ROI_START} ]] && [[ ! -z ${STR_ROI_END} ]] && echo "con1" && exit 1
    [[ ! -z ${STR_ROI_START} ]] && [[ -z ${STR_ROI_END} ]] && echo "con2" && exit 1
    true
}

function process_args() {
    while getopts "f:s:e:c:p" opt; do
        case $opt in
        f) EMON_DATA_PATH=("$OPTARG");;
        s) STR_ROI_START=("$OPTARG");;
        e) STR_ROI_END=("$OPTARG");;
        c) COMMAND=$OPTARG;;
        p) PARSE_RESULT=true ;;
        h) usage;;
        *) usage;;
        esac
    done
}

function _p(){
    echo cmd : ${COMMAND} 
    echo emon data path: ${EMON_DATA_PATH} 
    echo str_roi_start: ${STR_ROI_START} 
    echo str_roi_end: ${STR_ROI_END}
}

validate_env
process_args "$@"
validate_args

echo start....

if [[ -z ${STR_ROI_START} ]] && [[ -z ${STR_ROI_END} ]]; then
    emon -collect-edp -f ${EMON_DATA_PATH} -w ${COMMAND}
else 

    exec 3< <(eval ${COMMAND})

    while read <&3 line; 
    do 
        echo : ${line}
        if [[ ${trace_started} = false ]] && [[ $(echo ${line} | grep "${STR_ROI_START}") ]]; then
            emon -collect-edp -f ${EMON_DATA_PATH} &
            trace_started=true
        fi
        if [[ ${trace_started} = true ]] && [[ $(echo ${line} | grep "${STR_ROI_END}") ]]; then
            emon -stop
            trace_started=false
        fi
    done

    wait 
    echo "::: finished"

fi

if [[ ${PARSE_RESULT} = "true" ]] && [[ -f ${EMON_DATA_PATH} ]]; then
    python  /usr/local/emon/sep/config/edp/pyedp/edp.py     \
   -m /usr/local/emon/sep/config/edp/sapphirerapids_server_2s_private.xml     \
   -i ${EMON_DATA_PATH}  -o ${EMON_DATA_PATH}.xlsx --socket-view --core-view --thread-view
fi