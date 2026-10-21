#include "hip/hip_runtime.h"
// Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark stresses DRAM: strided pointer chasing with L1/L2-bypassing loads (sc0 sc1 nt)

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (304 * 1024)
#define WARP_SIZE 32

#define ARRAY_SIZE 67108864 // 2^26
#define STRIDE 163840//2^14 * 10

// GPU error check
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
        if (code != hipSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

#define SCALING_ARRAY_SIZE 16   // words per node, loaded as 4 x dwordx4
static_assert(STRIDE % SCALING_ARRAY_SIZE == 0, "nodes must stay aligned along the chain");

typedef unsigned v4u32 __attribute__((ext_vector_type(4)));   // 128-bit VGPR tuple for inline asm

// 32-bit elements: store the index of the element STRIDE ahead
__global__ void pointers_init(unsigned *posArray){
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ARRAY_SIZE)
    posArray[i] = (unsigned)((i + STRIDE) % ARRAY_SIZE);
}

__global__ void PowerKernal2(unsigned* A, unsigned* B, unsigned long long N)
{
    uint32_t uid = blockDim.x * blockIdx.x + threadIdx.x;
    if (uid < ARRAY_SIZE / SCALING_ARRAY_SIZE) {
    unsigned ia = uid*SCALING_ARRAY_SIZE;   // current node in A
    unsigned ib = uid*SCALING_ARRAY_SIZE;   // current node in B
    unsigned sink = 0;
    v4u32 v0, v1, v2, v3;
	for(uint64_t i=0; i<N/100; ++i) {
#pragma unroll 100
	  for(uint64_t j=0; j<100; ++j) {
      asm volatile (   // 16 consecutive words of the current node as 4 x dwordx4
        "global_load_dwordx4 %0, %4, off sc0 sc1 nt\n\t"   // sc0 sc1 nt: bypass L1 and L2, go to memory
        "global_load_dwordx4 %1, %4, off offset:16 sc0 sc1 nt\n\t"
        "global_load_dwordx4 %2, %4, off offset:32 sc0 sc1 nt\n\t"
        "global_load_dwordx4 %3, %4, off offset:48 sc0 sc1 nt\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3) : "v"(A + ia) : "memory"
      );
      sink += uid + v0.x;
      sink += uid + v0.y;
      sink += uid + v0.z;
      sink += uid + v0.w;
      sink += uid + v1.x;
      sink += uid + v1.y;
      sink += uid + v1.z;
      sink += uid + v1.w;
      sink += uid + v2.x;
      sink += uid + v2.y;
      sink += uid + v2.z;
      sink += uid + v2.w;
      sink += uid + v3.x;
      sink += uid + v3.y;
      sink += uid + v3.z;
      sink += uid + v3.w;
      A[ia + 1] = sink;   // word 1: keeps the store without overwriting the chain in word 0
      ia = v0.x;              // word 0 holds the index of the next node
      asm volatile (   // 16 consecutive words of the current node as 4 x dwordx4
        "global_load_dwordx4 %0, %4, off sc0 sc1 nt\n\t"   // sc0 sc1 nt: bypass L1 and L2, go to memory
        "global_load_dwordx4 %1, %4, off offset:16 sc0 sc1 nt\n\t"
        "global_load_dwordx4 %2, %4, off offset:32 sc0 sc1 nt\n\t"
        "global_load_dwordx4 %3, %4, off offset:48 sc0 sc1 nt\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3) : "v"(B + ib) : "memory"
      );
      sink += uid + v0.x;
      sink += uid + v0.y;
      sink += uid + v0.z;
      sink += uid + v0.w;
      sink += uid + v1.x;
      sink += uid + v1.y;
      sink += uid + v1.z;
      sink += uid + v1.w;
      sink += uid + v2.x;
      sink += uid + v2.y;
      sink += uid + v2.z;
      sink += uid + v2.w;
      sink += uid + v3.x;
      sink += uid + v3.y;
      sink += uid + v3.z;
      sink += uid + v3.w;
      B[ib + 1] = sink;   // word 1: keeps the store without overwriting the chain in word 0
      ib = v0.x;              // word 0 holds the index of the next node
    }
	}
    }
}

int main(int argc, char** argv){
  unsigned long long iterations;
  if (argc != 2){
    fprintf(stderr,"usage: %s #iterations\n",argv[0]);
    exit(1);
  }
  else {
    iterations = atoll(argv[1]);
  }
 printf("Power Microbenchmarks with iterations %llu\n",iterations);

  unsigned *d_A;
  unsigned *d_B;
  checkCudaErrors( hipMalloc(&d_A, (size_t)ARRAY_SIZE*sizeof(unsigned)) );
  checkCudaErrors( hipMalloc(&d_B, (size_t)ARRAY_SIZE*sizeof(unsigned)) );

 hipEvent_t start, stop;
 float elapsedTime = 0;
 checkCudaErrors(hipEventCreate(&start));
 checkCudaErrors(hipEventCreate(&stop));

 int init_blocks = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
 pointers_init<<<init_blocks, THREADS_PER_BLOCK>>>(d_A);
 pointers_init<<<init_blocks, THREADS_PER_BLOCK>>>(d_B);
 checkCudaErrors(hipDeviceSynchronize());

 checkCudaErrors(hipEventRecord(start));
 PowerKernal2<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(d_A, d_B, iterations);
 checkCudaErrors(hipEventRecord(stop));

 checkCudaErrors(hipEventSynchronize(stop));
 checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));
 printf("gpu execution time = %.3f ms\n", elapsedTime);

  checkCudaErrors( hipPeekAtLastError() );
  checkCudaErrors(hipEventDestroy(start));
  checkCudaErrors(hipEventDestroy(stop));
  checkCudaErrors(hipFree(d_A));
  checkCudaErrors(hipFree(d_B));

  return 0;
}
