#include "gflags/gflags.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>
#include "tc.h"
#include <string.h>

#define N 0x20
TestCase *tc_set[N];

void init_tc_set(){
    memset((void*)tc_set, 0, sizeof(TestCase *)*N);
}

bool register_tc(TestCase &tcase){
    for( int i=0; i<N; i++){
        if( tc_set[i] != 0 )
            continue;
        else{
            tc_set[i]=&tcase;
            return true;
        }
    }
    return true;
}

void run_tc(string tc_name, unsigned int p, unsigned int nof_thrds, int argc, char** argv){
    for( int i=0; i<N; i++){
        if( tc_set[i]!=0 && tc_name == tc_set[i]->tc_name){
            _pass_args(*tc_set[i], argc, argv);
            run_period_with_n_thrds(p, nof_thrds, *tc_set[i]);
            return ;
        }
    }
}