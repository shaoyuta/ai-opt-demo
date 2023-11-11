#include <xmmintrin.h>
#include <stdio.h>

static void farpokeb(unsigned short sel, void* off, unsigned char v)
{
    asm ( "push %%fs\n\t"
          "mov  %0, %%fs\n\t"
          "movb %2, %%fs:(%1)\n\t"
          "pop %%fs"
          : : "g"(sel), "r"(off), "r"(v) );
}

static void farpokeb2(unsigned short sel, void* off, unsigned char v)
{
    asm ( "push %%fs; \
          mov  %0, %%fs; \
          movb %2, %%fs:(%1); \
          pop %%fs"
          : : "g"(sel), "r"(off), "r"(v) );
}

#if 1
void cpufp_kernel_x86_sse_fp32(void)
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
#endif


int main()
{
	cpufp_kernel_x86_sse_fp32();
	int src = 1;
	int dst;   

	asm inline (
		"mov %1, %0; "
		"add $1, %0"
		: "=r" (dst) 
		: "r" (src)
		);

	printf("%d\n", dst);

	return 0;
}