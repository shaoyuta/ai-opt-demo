#ifndef _TC_H_
#define _TC_H_
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>
#include <string>

using namespace std;

typedef struct t_TestCase {
  string tc_name;
  function<bool(t_TestCase *tcase)> prepare_hook;  // Null: no need to prepare, Func() -> False: return; True: go ahead
  function<void(t_TestCase *tcase)> run;
  function<bool(t_TestCase *tcase)> post_hook;
  int argc;
  char** argv;
  struct t_TestCase* next;
}TestCase;

extern void prepare_tcs();
extern bool register_tc(TestCase &tcase);
extern void run_tc(string tc_name, unsigned int p, unsigned int nof_thrds, int argc, char** argv);
extern void init_tc_set();
extern void _pass_args(TestCase& tcase, int argc, char** argv);
extern void run_period_with_n_thrds(int p, int n, TestCase tcase);
#endif