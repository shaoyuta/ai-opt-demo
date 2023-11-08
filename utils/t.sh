#!/bin/bash

set -e

cmd='/home/taosy/llama2/1-c4709ac180e/workspace/run_test.sh'
str_roi_start='Iteration'
str_roi_end='Summary'
trace_started=false

exec 3< <(eval ${cmd})

while read <&3 line; 
do 
    echo : ${line}
    if [[ ${trace_started} = false ]] && [[ $(echo ${line} | grep ${str_roi_start}) ]]; then
        emon -collect-edp -f emon.dat &
        trace_started=true
    fi
    if [[ ${trace_started} = true ]] && [[ $(echo ${line} | grep ${str_roi_end}) ]]; then
        emon -stop
        trace_started=false
    fi
done

wait 
echo "::: finished"
