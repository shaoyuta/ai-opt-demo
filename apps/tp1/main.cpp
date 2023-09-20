#include <iostream>
#include "gflags/gflags.h"
#include <time.h>
#include <pthread.h>
#include <stdio.h>
/* for ETIMEDOUT */
#include <errno.h>
#include <string.h>

DEFINE_string(m, "Hello world!", "Message to print");
DEFINE_uint32(N, 1, "nof threads");
DEFINE_uint32(p, 1, "period (s)");
using namespace std;

pthread_mutex_t calculating = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t done = PTHREAD_COND_INITIALIZER;

void *expensive_call(void *data)
{
        int oldtype;
        pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype);
        for (;;) {}
        pthread_cond_signal(&done);
        return NULL;
}

/* note: this is not thread safe as it uses a global condition/mutex */
int do_or_timeout(struct timespec *max_wait)
{
        struct timespec abs_time;
        pthread_t tid;
        int err;

        pthread_mutex_lock(&calculating);

        clock_gettime(CLOCK_REALTIME, &abs_time);
        abs_time.tv_sec += max_wait->tv_sec;
        abs_time.tv_nsec += max_wait->tv_nsec;

        for(int n=0; n<FLAGS_N; n++){
          pthread_create(&tid, NULL, expensive_call, NULL);
        }

        err = pthread_cond_timedwait(&done, &calculating, &abs_time);

        if (err == ETIMEDOUT)
                fprintf(stderr, "%s: calculation timed out\n", __func__);

        if (!err)
                pthread_mutex_unlock(&calculating);

        return err;
}

int main(int argc, char *argv[])
{
   gflags::ParseCommandLineFlags(&argc, &argv, true);

   struct timespec max_wait;

   memset(&max_wait, 0, sizeof(max_wait));

   /* wait at most 2 seconds */
   max_wait.tv_sec = FLAGS_p;
   do_or_timeout(&max_wait);

   gflags::ShutDownCommandLineFlags();
   return 0;
}