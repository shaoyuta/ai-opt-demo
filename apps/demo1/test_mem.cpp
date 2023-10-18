#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include "tc.h"
#include <string.h>


using namespace std;
using namespace std::chrono_literals;

#define DEFAULT_SG 4
#define DEFAUT_PLAN 0

static unsigned long long g_total_size;

static void getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    g_total_size = pages * page_size;
    return;
}

static void test_mem_plan_0(int* p, unsigned long long size){
  memset(p, 0x99, size);
  while(1){
  volatile int v;
    for(unsigned long long i=0; i<size; i+=sizeof(int))
      v=p[i>>2];
    for(unsigned long long i=0; i<size; i+=sizeof(int))
      v=p[i>>2];
    for(unsigned long long i=0; i<size; i+=sizeof(int))
      v=p[i>>2];
  }
  return;
}

static void test_mem_plan_1(int* p, unsigned long long size){
  memset(p, 0x99, size);
  volatile int v;
  for(unsigned long long i=0; i<size; i+=sizeof(int))
    v=p[i>>2];
  for(unsigned long long i=0; i<size; i+=sizeof(int))
    v=p[i>>2];
  for(unsigned long long i=0; i<size; i+=sizeof(int))
    v=p[i>>2];
  return;
}

void test_mem_main(int sg, int plan) {
  int* a;
  unsigned long long size=sg*0x400*0x400;
  size *= 0x400;
  a=(int*)malloc(size);
  if (a == 0)
    return ;
  switch (plan){
    case 0:
      test_mem_plan_0(a,size);
      break;
    case 1:
      test_mem_plan_1(a,size);
      break;
    case 2:
      break;
    default:
      cout<< "tc mem: error plan"<<endl;
      return ;
  }
  free(a);
}

/*
-t 'mem' -- 4 1 => alloc 4G, plan 1 (or 2,3,4)
*/
TestCase mem={
  .tc_name = "mem",
  .prepare_hook=[](TestCase* pcase){
     getTotalSystemMemory();
     return true;
  },
  .run=[](TestCase* pcase){
    int sg, plan;
    switch (pcase->argc){
      case 0:
        sg=DEFAULT_SG;
        plan=DEFAUT_PLAN;
        break;
      case 1:
        sg=stoi( pcase->argv[0] );
        plan=DEFAUT_PLAN;
        break;
      case 2:
        sg=stoi( pcase->argv[0] );
        plan=stoi( pcase->argv[1]);
        break;
      default:
        cout<<"tc mem: param error, exit"<<endl;
        return;
    }
    while(1)
      test_mem_main(sg, plan);
    return;
  },
};

REGISTER_TC(mem);