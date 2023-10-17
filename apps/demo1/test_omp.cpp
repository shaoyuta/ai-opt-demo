#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <omp.h>
#include "tc.h"

using namespace std;
using namespace std::chrono_literals;

static unsigned long long g_total_size;

void test_omp_main() {
    int i;
    int thread_id;

    #pragma omp parallel
    {
        thread_id = omp_get_thread_num();

        for( int i = 0; i < omp_get_max_threads(); i++){
            if(i == omp_get_thread_num()){
                printf("Hello from process: %d\n", thread_id);
            }
            #pragma omp barrier
        }
    }
}

/*
-t 'omp' 
*/
TestCase omp={
  .tc_name = "omp",
  .run=[](TestCase* pcase){
    test_omp_main();
    return;
  },
};

REGISTER_TC(omp);