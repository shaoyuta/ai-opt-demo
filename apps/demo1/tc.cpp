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

TestCase* pTCHeader=nullptr;

bool register_tc(TestCase &tcase){
    TestCase** p=&pTCHeader;
    while( *p )
        p=&(*p)->next;
    *p = &tcase;
    (*p)->next=nullptr;
    return true;
}

void run_tc(string tc_name, unsigned int p, unsigned int nof_thrds, int argc, char** argv){
    TestCase* pTC=pTCHeader;
    while(p){
        if( tc_name == pTC->tc_name){
            _pass_args(*pTC, argc, argv);
            run_period_with_n_thrds(p, nof_thrds, *pTC);
            return ;
        }
    }
}
