#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "tc.h"

using namespace std;
using namespace std::chrono_literals;



unsigned long long getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

void test_mem_main(int sg) {
  int* a;
  unsigned long long size=sg*0x400*0x400;
  size *= 0x400;
  a=(int*)malloc(size);
  if (a == 0)
    return ;
  for( unsigned long long i=0; i<size; i+=sizeof(int)){
     a[i>>2]=0x99;
  }
  free(a);
}

/*
-t 'mem' -- 4 => alloc 4G
*/
TestCase mem={
  .tc_name = "mem",
  .run=[](TestCase* pcase){
    int sg = std::stoi( pcase->argv[0] );
    test_mem_main(sg);
    return;
  },
};

REGISTER_TC(mem);