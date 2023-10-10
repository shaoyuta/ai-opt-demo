#include "gflags/gflags.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>

DEFINE_string(t,"busy","tasks");
DEFINE_uint32(N, 1, "nof threads");
DEFINE_uint32(p, 1, "period (s)");

using namespace std;
using namespace std::chrono_literals;

typedef void (*test_fun)(void);

// extern test func
extern void test_amx_main(void);
extern void test_avx512_main(void);

void run_period_with_n_thrds(int p, int n, test_fun fun) {
  std::mutex m;
  std::condition_variable cv;

  for (int i; i < n; i++) {
    std::thread t([&cv, &fun]() {
      fun();
      cv.notify_one();
    });
    t.detach();
  }

  {
    std::unique_lock<std::mutex> l(m);
    sleep(p);
  }
}

void busyloop() {
  while (1) {
  };
  return;
}

void dummy(){
  sleep(-1);
}

void loop_amx_test(){
  while(1){
    test_amx_main();
  }
}

void loop_avx512_test(){
  while(1){
    test_avx512_main();
  }
}

void cpu_utilization(){
#if 0
  int time_start;
  int fulltime = 100;//总时间
  int runtime = 50;//运行时间
#else
  float ratio=40;
  float r=ratio/100;
  float a=1/r-1;
  unsigned int ut=a*1000;
#endif
   while(1){
#if 0
      time_start = clock();
      while((clock()-time_start)<runtime){}
      usleep(runtime);
#else
      for(unsigned int i=0; i<300000; i++){}
      usleep(ut);
#endif
   }
   return;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

   if (FLAGS_t == "busy")
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, busyloop);
   if (FLAGS_t == "dummy")
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, dummy);
   if (FLAGS_t == "cpu")
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, cpu_utilization);
   if (FLAGS_t == "amx")
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, loop_amx_test);
   if (FLAGS_t == "avx512")
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, loop_avx512_test);

    
   test_avx512_main();
  gflags::ShutDownCommandLineFlags();
  return 0;  
}