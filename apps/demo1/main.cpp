#include "gflags/gflags.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>

DEFINE_string(t,"busy","tasks");
DEFINE_uint32(N, 1, "nof threads");
DEFINE_uint32(p, 1, "period (s)");

using namespace std;
using namespace std::chrono_literals;

typedef void (*test_fun)(void);

// extern test func
extern void test_amx_main(void);
extern void test_avx512_main(void);
extern void test_mem_main(void);

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

typedef struct t_TestCase {
  function<bool(t_TestCase *tcase)> prepare_hook;  // Null: no need to prepare, Func() -> False: return; True: go ahead
  function<void(t_TestCase *tcase)> run;
  function<bool(t_TestCase *tcase)> post_hook;
  int argc;
  char** argv;
}TestCase;

TestCase demo={
  .prepare_hook=[](TestCase* pcase){
     cout<< "demo: in prepare"<<endl;
     return true;
  },
  .run=[](TestCase* pcase){
    for(int i=0; i<pcase->argc; i++)
      cout<< "demo: in run: "<<pcase->argv[i]<<endl;
    return;
  },
  .post_hook=[](TestCase* pcase){
     cout<< "demo: in post"<<endl;
     return true;
  }
};

TestCase busy={
  .run=[](TestCase* pcase){
    while(1){};
    return;
  },
};

TestCase cpu={
  .run=[](TestCase* pcase){
    cpu_utilization();
    return;
  },
};

TestCase amx={
  .run=[](TestCase* pcase){
    test_amx_main();
    return;
  },
};

TestCase mem={
  .run=[](TestCase* pcase){
    test_mem_main();
    return;
  },
};

TestCase avx512={
  .run=[](TestCase* pcase){
    test_avx512_main();
    return;
  },
};

void run_period_with_n_thrds(int p, int n, TestCase tcase) {

  std::mutex m;
  std::condition_variable cv;

  if( tcase.prepare_hook && !tcase.prepare_hook(&tcase) )
    return ;

  for (int i=0; i < n; i++) {
    std::thread t([&]() {
      tcase.run(&tcase);
      cv.notify_one();
    });
    t.detach();
  }

  {
    std::unique_lock<std::mutex> l(m);
    sleep(p);
  }
}

void _pass_args(TestCase& tcase, int argc, char** argv){
  tcase.argc=argc;
  tcase.argv=argv;
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_t == "demo"){
      _pass_args(demo, argc-1, argv+1);
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, demo);
  }
  if (FLAGS_t == "busy"){
      _pass_args(busy, argc-1, argv+1);
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, busy);
  }
  if (FLAGS_t == "cpu"){
      _pass_args(cpu, argc-1, argv+1);
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, cpu);
  }

  if (FLAGS_t == "amx"){
      _pass_args(amx, argc-1, argv+1);
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, amx);
  }

  if (FLAGS_t == "mem"){
      _pass_args(mem, argc-1, argv+1);
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, mem);
  }

  if (FLAGS_t == "avx512"){
      _pass_args(avx512, argc-1, argv+1);
      run_period_with_n_thrds(FLAGS_p, FLAGS_N, avx512);
  }
   
  gflags::ShutDownCommandLineFlags();
  return 0;  
}