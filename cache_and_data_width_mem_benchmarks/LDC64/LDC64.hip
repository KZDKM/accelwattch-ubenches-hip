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
// Includes
#include <stdio.h>
#include <stdlib.h>

// includes CUDA
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (304*1024)

__constant__ uint64_t ConstArray1[THREADS_PER_BLOCK];



uint64_t* h_Value;
uint64_t* d_Value;


// Functions
void CleanupResources(void);
void RandomInit(uint64_t*, int);
FILE *fp;


////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(hipError_t err, const char *file, const int line )
{
  if(hipSuccess != err){
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, hipGetErrorString( err ) );
	 exit(-1);
  }
}

// This will output the proper error string when calling hipGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
  hipError_t err = hipGetLastError();
  if (hipSuccess != err){
	fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",file, line, errorMessage, (int)err, hipGetErrorString( err ) );
	exit(-1);
  }
}

// end of CUDA Helper Functions


__global__ void PowerKernal(uint64_t* Value, unsigned long long iterations)
{
	int tid = threadIdx.x;
	int i = blockIdx.x*THREADS_PER_BLOCK + tid;

	uint64_t Value1=0;
    uint64_t sink=0;
	#pragma unroll 100
    for(unsigned long long k=0; k<iterations;k++) {
		asm volatile(
			"global_load_dwordx2 %0, %1, off\n\t"
			"s_waitcnt vmcnt(0)\n\t"
			: "=v"(Value1)
			: "v"(ConstArray1)
		);
        sink += Value1;
	}
		Value[i] = sink;
}



// Host code

int main(int argc, char** argv) 
{
    unsigned long long iterations;
    if (argc != 2){
        fprintf(stderr,"usage: %s #iterations\n",argv[0]);
        exit(1);
    }
    else{
        iterations = atoll(argv[1]);
    }

    printf("Power Microbenchmark with iterations %llu\n",iterations);
	uint64_t array1[THREADS_PER_BLOCK];
	int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
	size_t size = N * sizeof(uint64_t);
	 h_Value = (uint64_t *) malloc(size);

	// Initialize input vectors
	RandomInit(array1, THREADS_PER_BLOCK);

	 hipMemcpyToSymbol(HIP_SYMBOL(ConstArray1), array1, sizeof(uint64_t) * THREADS_PER_BLOCK );
	 
	 checkCudaErrors( hipMalloc((void**)&d_Value, size ));
	 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	 dim3 dimGrid(NUM_OF_BLOCKS,1);
	 dim3 dimBlock(THREADS_PER_BLOCK,1);

	 hipEvent_t start, stop;
	  float elapsedTime = 0;
	  checkCudaErrors(hipEventCreate(&start));
	  checkCudaErrors(hipEventCreate(&stop));

	  checkCudaErrors(hipEventRecord(start));
	  PowerKernal<<<dimGrid,dimBlock>>>(d_Value, iterations);
	  checkCudaErrors(hipEventRecord(stop));
	 
	  checkCudaErrors(hipEventSynchronize(stop));
	  checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));
	  printf("gpu execution time = %.3f ms\n", elapsedTime);
	  getLastCudaError("kernel launch failure");
	  checkCudaErrors( hipMemcpy(h_Value, d_Value, size, hipMemcpyDeviceToHost) );
 	  checkCudaErrors(hipEventDestroy(start));
	  checkCudaErrors(hipEventDestroy(stop));
	  return 0;
}

// Allocates an array with random float entries.
void RandomInit(uint64_t* data, int n)
{
  for (int i = 0; i < n; ++i){
  srand((uint64_t)time(0));  
  data[i] = rand() / RAND_MAX;
  }
}





