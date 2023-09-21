#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>

using namespace std::chrono_literals;

typedef void (*test_fun)(void);

void run_period_with_n_thrds(int p, int n, test_fun fun){
   std::mutex m;
   std::condition_variable cv;

   for(int i;i<n;i++){
      std::thread t([&cv, &fun]() 
      {
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

void f()
{
   while(1);
   return ;
}


int main()
{
   run_period_with_n_thrds(100, 34, f);

   return 0;
}