#!/bin/bash

set -e

cmd='/home/taosy/llm/1-c4709ac180e/workspace/run_test.sh'
str_roi='Iteration:'


function foo(){
    for i in {1..5};do
        echo ${i}:
        sleep 1
    done
}

#( eval ${cmd} > backlight ) &
#exec 3< backlight


exec 3< <(eval ${cmd})
#str_roi='3:'
#exec 3< <(foo)

while read <&3 line; 
do 
    echo ===${line}
    if [[ $(echo ${line} | grep ${str_roi}) ]]; then
        echo -n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        emon -collect-edp -f emon.dat &
        break
    fi
done



wait 
echo "finished"
emon -stop