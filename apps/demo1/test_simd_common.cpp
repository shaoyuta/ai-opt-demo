

void test_x86_sse_fp32(void)
{	
    asm(" \
	xorps %%xmm0, %%xmm0; \
	xorps %%xmm1, %%xmm1; \
    xorps %%xmm2, %%xmm2; \
    xorps %%xmm3, %%xmm3; \
    xorps %%xmm4, %%xmm4; \
    xorps %%xmm5, %%xmm5; \
    xorps %%xmm6, %%xmm6; \
    xorps %%xmm7, %%xmm7; \
    xorps %%xmm8, %%xmm8; \
    xorps %%xmm9, %%xmm9; \
    xorps %%xmm10, %%xmm10; \
    xorps %%xmm11, %%xmm11; \
    xorps %%xmm12, %%xmm12; \
    xorps %%xmm13, %%xmm13; \
    xorps %%xmm14, %%xmm14; \
    xorps %%xmm15, %%xmm15; \
.cpufp.x86.sse.fp32.L1: ;\
	mulps %%xmm0, %%xmm0 ;\
	addps %%xmm1, %%xmm1 ;\
    mulps %%xmm2, %%xmm2 ;\
    addps %%xmm3, %%xmm3 ;\
    mulps %%xmm4, %%xmm4 ;\
    addps %%xmm5, %%xmm5 ;\
    mulps %%xmm6, %%xmm6 ;\
    addps %%xmm7, %%xmm7 ;\
    mulps %%xmm8, %%xmm8 ;\
    addps %%xmm9, %%xmm9 ;\
    mulps %%xmm10, %%xmm10 ;\
    addps %%xmm11, %%xmm11 ;\
    mulps %%xmm12, %%xmm12 ;\
    addps %%xmm13, %%xmm13 ;\
    mulps %%xmm14, %%xmm14 ;\
    addps %%xmm15, %%xmm15 ;\
    jmp .cpufp.x86.sse.fp32.L1 ;\
    ret \
	"
	: : :
	);
}

void test_x86_sse_fp64(void)
{	
    asm(" \
	xorpd %%xmm0, %%xmm0; \
	xorpd %%xmm1, %%xmm1; \
    xorpd %%xmm2, %%xmm2; \
    xorpd %%xmm3, %%xmm3; \
    xorpd %%xmm4, %%xmm4; \
    xorpd %%xmm5, %%xmm5; \
    xorpd %%xmm6, %%xmm6; \
    xorpd %%xmm7, %%xmm7; \
    xorpd %%xmm8, %%xmm8; \
    xorpd %%xmm9, %%xmm9; \
    xorpd %%xmm10, %%xmm10; \
    xorpd %%xmm11, %%xmm11; \
    xorpd %%xmm12, %%xmm12; \
    xorpd %%xmm13, %%xmm13; \
    xorpd %%xmm14, %%xmm14; \
    xorpd %%xmm15, %%xmm15; \
.cpufp.x86.sse.fp64.L1: ;\
	mulpd %%xmm0, %%xmm0 ;\
	addpd %%xmm1, %%xmm1 ;\
    mulpd %%xmm2, %%xmm2 ;\
    addpd %%xmm3, %%xmm3 ;\
    mulpd %%xmm4, %%xmm4 ;\
    addpd %%xmm5, %%xmm5 ;\
    mulpd %%xmm6, %%xmm6 ;\
    addpd %%xmm7, %%xmm7 ;\
    mulpd %%xmm8, %%xmm8 ;\
    addpd %%xmm9, %%xmm9 ;\
    mulpd %%xmm10, %%xmm10 ;\
    addpd %%xmm11, %%xmm11 ;\
    mulpd %%xmm12, %%xmm12 ;\
    addpd %%xmm13, %%xmm13 ;\
    mulpd %%xmm14, %%xmm14 ;\
    addpd %%xmm15, %%xmm15 ;\
    jmp .cpufp.x86.sse.fp64.L1 ;\
    ret \
	"
	: : :
	);
}

void test_x86_avx_fp64(void){
        asm(" \
	vxorpd %%ymm0, %%ymm0, %%ymm0; \
	vxorpd %%ymm1, %%ymm1, %%ymm1; \
    vxorpd %%ymm2, %%ymm2, %%ymm3; \
    vxorpd %%ymm0, %%ymm0, %%ymm4; \
    vxorpd %%ymm0, %%ymm0, %%ymm5; \
    vxorpd %%ymm0, %%ymm0, %%ymm6; \
    vxorpd %%ymm0, %%ymm0, %%ymm7; \
    vxorpd %%ymm0, %%ymm0, %%ymm8; \
    vxorpd %%ymm9, %%ymm9, %%ymm9; \
    vxorpd %%ymm10, %%ymm10, %%ymm10; \
    vxorpd %%ymm11, %%ymm11, %%ymm11; \
    vxorpd %%ymm12, %%ymm12, %%ymm12; \
.cpufp.x86.avx.fp64.L1: ;\
	vmulpd %%ymm12, %%ymm12, %%ymm0 ;\
	vaddpd %%ymm12, %%ymm12, %%ymm1 ;\
    vmulpd %%ymm12, %%ymm12, %%ymm2 ;\
    vaddpd %%ymm12, %%ymm12, %%ymm3 ;\
    vmulpd %%ymm12, %%ymm12, %%ymm4 ;\
    vaddpd %%ymm12, %%ymm12, %%ymm5 ;\
    vmulpd %%ymm12, %%ymm12, %%ymm6 ;\
    vaddpd %%ymm12, %%ymm12, %%ymm7 ;\
    vmulpd %%ymm12, %%ymm12, %%ymm8 ;\
    vaddpd %%ymm12, %%ymm12, %%ymm9 ;\
    vmulpd %%ymm12, %%ymm12, %%ymm10 ;\
    vaddpd %%ymm12, %%ymm12, %%ymm11 ;\
    vmulpd %%ymm12, %%ymm12, %%ymm11 ;\
    jmp .cpufp.x86.avx.fp64.L1 ;\
    ret \
	"
	: : :
	);
}


void test_x86_avx512_fp64(void){
        asm(" \
	vpxorq %%zmm0, %%zmm0, %%zmm0; \
	vpxorq %%zmm1, %%zmm1, %%zmm1; \
    vpxorq %%zmm2, %%zmm2, %%zmm3; \
    vpxorq %%zmm0, %%zmm0, %%zmm4; \
    vpxorq %%zmm0, %%zmm0, %%zmm5; \
    vpxorq %%zmm0, %%zmm0, %%zmm6; \
    vpxorq %%zmm0, %%zmm0, %%zmm7; \
    vpxorq %%zmm0, %%zmm0, %%zmm8; \
    vpxorq %%zmm9, %%zmm9, %%zmm9; \
.cpufp.x86.avx512.fp64.L1: ;\
	vfmadd231pd %%zmm12, %%zmm12, %%zmm0 ;\
	vfmadd231pd %%zmm12, %%zmm12, %%zmm1 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm2 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm3 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm4 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm5 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm6 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm7 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm8 ;\
    vfmadd231pd %%zmm12, %%zmm12, %%zmm9 ;\
    jmp .cpufp.x86.avx512.fp64.L1 ;\
    ret \
	"
	: : :
	);
}