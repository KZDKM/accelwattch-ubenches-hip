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

// global_store_byte with lanes 128 b apart: 64 write requests per wave store, kept in l2

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <hip/hip_runtime.h>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
        if (code != hipSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

typedef unsigned v4u32 __attribute__((ext_vector_type(4)));

__global__ void fill_bytes(uint8_t *buf, size_t n){   // hashed, so lanes see different data
  for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x)
    buf[i] = (uint8_t)(((uint32_t)i * 2654435761u) >> 24);
}

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (304 * 1024)
#define FOOTPRINT_BYTES (2u << 20)   // fits one xcd's 4 mb l2
#define LANE_STRIDE_BYTES 128
#define SLOTS (FOOTPRINT_BYTES / LANE_STRIDE_BYTES)
#define KERNEL store_stress

__global__ void store_stress(uint8_t *buf, uint64_t *dsink, unsigned long long iterations){
  uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint8_t *ptr = buf + (size_t)(tid & (SLOTS - 1)) * LANE_STRIDE_BYTES;
  uint32_t v = tid;

  for(unsigned long long i=0; i<iterations/1000; ++i) {   // 0.1 store per iteration: 64 requests each, 1e6 would time out
  #pragma unroll 100
    for(unsigned long long j=0; j<100; ++j) {
    asm volatile (
      "global_store_byte %0, %1, off\n\t"
      "s_waitcnt vmcnt(0)\n\t"
      : : "v"(ptr), "v"(v) : "memory"
    );
    }
  }
  dsink[tid] = tid;
}

int main(int argc, char** argv){
  unsigned long long iterations;
  if (argc != 2){
    fprintf(stderr,"usage: %s #iterations\n",argv[0]);
    exit(1);
  }
  iterations = atoll(argv[1]);
  printf("Power Microbenchmarks with iterations %llu\n",iterations);

  size_t total_threads = (size_t)THREADS_PER_BLOCK*NUM_OF_BLOCKS;
  uint8_t *buf_g;
  uint64_t *dsink_g;
  checkCudaErrors( hipMalloc(&buf_g, FOOTPRINT_BYTES) );
  fill_bytes<<<4096, 256>>>(buf_g, FOOTPRINT_BYTES);
  checkCudaErrors( hipDeviceSynchronize() );
  checkCudaErrors( hipMalloc(&dsink_g, total_threads*sizeof(uint64_t)) );
  uint64_t *dsink = (uint64_t*)malloc(total_threads*sizeof(uint64_t));

  hipEvent_t start, stop;
  float elapsedTime = 0;
  checkCudaErrors(hipEventCreate(&start));
  checkCudaErrors(hipEventCreate(&stop));

  checkCudaErrors(hipEventRecord(start));
  KERNEL<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(buf_g, dsink_g, iterations);
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));
  printf("gpu execution time = %.3f ms\n", elapsedTime);
  checkCudaErrors( hipPeekAtLastError() );

  checkCudaErrors( hipMemcpy(dsink, dsink_g, total_threads*sizeof(uint64_t), hipMemcpyDeviceToHost) );
  free(dsink);
  hipFree(buf_g);
  hipFree(dsink_g);
  return 0;
}
