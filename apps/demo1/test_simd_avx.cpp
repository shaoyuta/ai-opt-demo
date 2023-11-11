#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>
#include "tc.h"
#include "test_simd_common.h"

void test_simd_avx_main(void) {
    test_x86_avx_fp64();
}

TestCase simd_avx={
  .tc_name = "simd_avx",
  .run=[](TestCase* pcase){
    test_simd_avx_main();
    return;
  },
};
REGISTER_TC(simd_avx);