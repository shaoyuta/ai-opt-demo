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


REGISTER_TC(busy);

int main(int argc, char *argv[]) { 

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  run_tc(FLAGS_t, FLAGS_p, FLAGS_N, argc-1, argv+1);
  gflags::ShutDownCommandLineFlags();
  return 0;  
}