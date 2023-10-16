#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#include "tc.h"

#define N           0xfffffff
#define SEED        0x1234


void test_avx512_main(void) {
    clock_t start;
    double msecs;
    unsigned i;
    float *a, *b, *c;
    a = (float*) _mm_malloc(N*sizeof(float), 32);
    b = (float*) _mm_malloc(N*sizeof(float), 32);
    c = (float*) _mm_malloc(N*sizeof(float), 32);
    
    srand(SEED);
    for(i=0; i<N; i++) {
        a[i] = b[i] = (float)(rand() % N);
    }

    start = clock();

    __m512 A, B, C;

    for(i=0; i<(N & ((~(unsigned)0x0E))); i+=16) {
        A = _mm512_load_ps(&a[i]);
        B = _mm512_load_ps(&b[i]);
        C = _mm512_mul_ps(A, B);
        _mm512_store_ps(&c[i], C);
    }
    for(; i<N; i++) {
        c[i] = a[i] * b[i];
    }
    msecs = (clock()-start)/1000;
    printf("%f, %f, %f, %f\n", c[0], c[1], c[N-2], c[N-1]);
    printf("Execution time = %2.3lf ms\n", msecs);

    free(a);
    free(b);
    free(c);
}

TestCase avx512={
  .tc_name = "avx512",
  .run=[](TestCase* pcase){
    test_avx512_main();
    return;
  },
};
REGISTER_TC(avx512);