#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#include "tc.h"
#include "test_simd_common.h"

void test_simd_sse_main(void) {
    test_x86_sse_fp64();
}

TestCase simd_sse={
  .tc_name = "simd_sse",
  .run=[](TestCase* pcase){
    test_simd_sse_main();
    return;
  },
};
REGISTER_TC(simd_sse);