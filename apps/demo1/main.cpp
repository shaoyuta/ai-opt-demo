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

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_t == "busy")
    run_period_with_n_thrds(FLAGS_p, FLAGS_N, busyloop);
  if (FLAGS_t == "dummy")
    run_period_with_n_thrds(FLAGS_p, FLAGS_N, dummy);

  gflags::ShutDownCommandLineFlags();
  return 0;  
}