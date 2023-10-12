#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace std;
using namespace std::chrono_literals;


#define SIZE 0x1000000000

unsigned long long getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

void test_mem_main(void) {
  int* a;
  a=(int*)malloc(SIZE);
  if (a == 0)
    return ;
  for( unsigned long i=0; i<SIZE; i+=sizeof(int)){
     a[i>>2]=0x99;
  }
  free(a);
}