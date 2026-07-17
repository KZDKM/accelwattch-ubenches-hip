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

//This benchmark stresses the L1 cache

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (304 * 1024)
#define WARP_SIZE 32

// GPU error check
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
        if (code != hipSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void l1_pointers_init(uint64_t *posArray){

  uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid == 0){
    for(uint32_t blk = 0; blk <NUM_OF_BLOCKS; blk++){
      for (uint32_t i=0; i<(THREADS_PER_BLOCK-1); i++){
        posArray[(blk*THREADS_PER_BLOCK)+i] = (uint64_t)(posArray + (blk*THREADS_PER_BLOCK) + i + 1);
      }

      posArray[((blk+1)*THREADS_PER_BLOCK)-1] = (uint64_t)(posArray + (blk*THREADS_PER_BLOCK));
    }
  }
}

__global__ void l1_stress(uint64_t *posArray, uint64_t *dsink, unsigned long long iterations){

  // thread index
  uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;

  if(tid < NUM_OF_BLOCKS*THREADS_PER_BLOCK){
  	// a register to avoid compiler optimization
  	uint64_t *ptr = posArray + tid;
  	uint64_t ptr1, ptr0;

  	// initialize the thread pointer with the start address of the array
  	// use ca modifier to cache the in L1
    ptr1 = __ldg(ptr);
  	// synchronize all threads
  	//asm volatile ("bar.sync 0;");
    __syncthreads();

  	// pointer-chasing iterations times
  	// use ca modifier to cache the load in L1
  	#pragma unroll 100
  	for(unsigned long long i=0; i<iterations; ++i) { 
      ptr0 = __ldg((uint64_t*)ptr1);
  	  ptr1 = ptr0;    //swap the register for the next load
  	}

  	// write data back to memory
  	dsink[tid] = ptr1;
  }
}

int main(int argc, char** argv){
  unsigned long long iterations;
  if (argc != 2){
    fprintf(stderr,"usage: %s #iterations #cores #ActiveThreadsperWarp\n",argv[0]);
    exit(1);
  }
  else {
    iterations = atoll(argv[1]);
  }
  int total_threads = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 printf("Power Microbenchmarks with iterations %llu\n",iterations);

  uint64_t *dsink = (uint64_t*) malloc(total_threads*sizeof(uint64_t));
  

  uint64_t *posArray_g;
  uint64_t *dsink_g;
  

  checkCudaErrors( hipMalloc(&posArray_g, total_threads*sizeof(uint64_t)) );
  checkCudaErrors( hipMalloc(&dsink_g, total_threads*sizeof(uint64_t)) );
 hipEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(hipEventCreate(&start));  
 checkCudaErrors(hipEventCreate(&stop));

    l1_pointers_init<<<1,1>>>(posArray_g);
 checkCudaErrors(hipEventRecord(start));    
  l1_stress<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(posArray_g, dsink_g, iterations);
 checkCudaErrors(hipEventRecord(stop));               
 
 checkCudaErrors(hipEventSynchronize(stop));           
 checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
  
  
  checkCudaErrors( hipPeekAtLastError() );

  checkCudaErrors( hipMemcpy(dsink, dsink_g, total_threads*sizeof(uint64_t), hipMemcpyDeviceToHost) );

  return 0;
} 