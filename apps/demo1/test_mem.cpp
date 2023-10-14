#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

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
  for( unsigned long i=0; i<size; i+=sizeof(int)){
     a[i>>2]=0x99;
  }
  free(a);
}