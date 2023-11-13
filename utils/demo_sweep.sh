#!/usr/bin/bash

set -e

# <TC>_<CORES>_<Freq>.dat
# ec.sh -p -f ./avx.dat -c '/usr/bin/taskset -c 0-55 /home/taosy/repo/shaoyuta/ai-opt-demo/build/demo1/demo1 -N=56 -p=30 -t=simd_avx'

function run_case(){
    tc_name=$1
    nof_cores=$2
    freq=$3
        
    emon_fn=${tc_name}_${nof_cores}_${freq}
    command="/usr/bin/taskset -c 0-$((${nof_cores}-1)) /home/taosy/repo/shaoyuta/ai-opt-demo/build/demo1/demo1 -N=${nof_cores} -p=20 -t=${tc_name}"
    ec.sh -p -f ./${emon_fn}.dat -c "${command}"

}

FREQUENCYS=("2.4Ghz" "2.6Ghz" "2.8Ghz" "3.0Ghz" "3.2Ghz" "3.4Ghz" "3.6Ghz" "3.8Ghz")
#FREQUENCYS=("2.4Ghz")

TC_NAME=("simd_sse"  "simd_avx"  "simd_avx512")

for freq in ${FREQUENCYS[@]}
do
    echo "Setting frequency-${freq}"
    sudo cpupower frequency-set -u ${freq} >/dev/null ;sudo cpupower frequency-set -d ${freq} >/dev/null
    for tc in ${TC_NAME[@]}
    do
        for cores in {4..56..4}
        #for cores in {4..4..4}
        do
            run_case ${tc} ${cores} ${freq}
        done  
    done
done
