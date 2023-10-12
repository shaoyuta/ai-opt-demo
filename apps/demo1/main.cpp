#include "gflags/gflags.h"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>
#include "tc.h"

DEFINE_string(t,"busy","tasks");
DEFINE_uint32(N, 1, "nof threads");
DEFINE_uint32(p, 1, "period (s)");

using namespace std;
using namespace std::chrono_literals;

// extern test func
extern void test_amx_main(void);
extern void test_avx512_main(void);
extern void test_mem_main(void);

void cpu_utilization(int d){
#if 0
  int time_start;
  int fulltime = 100;//总时间
  int runtime = 50;//运行时间
#else
//  cout<<d<<endl;
  float ratio=(float)d;
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



TestCase demo={
  .tc_name = "demo",
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
  .tc_name = "busy",
  .run=[](TestCase* pcase){
    while(1){};
    return;
  },
};

/*
-t "cpu" -- 50
*/
TestCase cpu={
  .tc_name = "cpu",
  .run=[](TestCase* pcase){
    int r = std::stoi( pcase->argv[0] );
    cpu_utilization(r);
    return;
  },
};

TestCase amx={
  .tc_name = "amx",
  .run=[](TestCase* pcase){
    test_amx_main();
    return;
  },
};

TestCase mem={
  .tc_name = "mem",
  .run=[](TestCase* pcase){
    test_mem_main();
    return;
  },
};

TestCase avx512={
  .tc_name = "avx512",
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

void prepare_tcs(){
  init_tc_set();
  register_tc(busy);
  register_tc(cpu);
  register_tc(amx);
  register_tc(avx512);
  register_tc(mem);
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  prepare_tcs();
  run_tc(FLAGS_t, FLAGS_p, FLAGS_N, argc-1, argv+1);
  gflags::ShutDownCommandLineFlags();
  return 0;  
}