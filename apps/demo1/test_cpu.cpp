#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <functional>
#include "tc.h"

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

/*
-t "cpu" -- 50
*/
TestCase cpu={
  .tc_name = "cpu",
  .run=[](TestCase* pcase){
    int r;
    if( pcase->argc >0 ){
      r = std::stoi( pcase->argv[0] );
    }else{
      r=40;
    }
    cpu_utilization(r);
    return;
  },
};

REGISTER_TC(cpu);