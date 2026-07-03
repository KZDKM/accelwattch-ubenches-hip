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

// includes project
#include "../include/repeat.h"

#define MAX_THREADS_PER_BLOCK 256

#define LINE_SIZE 	128
#define SETS		4
#define ASSOC		24
#define SIMD_WIDTH	32

// Variables
int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};


// Functions
void CleanupResources(void);
void RandomInit(int*, int);


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

// This will output the proper error string when calling cudaGetLastError
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

texture<float,1,hipReadModeElementType> texmem1;
texture<float,1,hipReadModeElementType> texmem2;
texture<float,1,hipReadModeElementType> texmem3;


__global__ void tex_bm_kernel( float* out, unsigned size, int iterations)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

	float I1=0;
	float I2=0;
	if(tid < size){
		for(unsigned i=0; i<iterations; ++i){
			out[tid] = tex1Dfetch(texmem1,tid);
			out[tid*2] = tex1Dfetch(texmem2,tid);
			I1=out[tid];
			I2=out[tid*2];
			__asm volatile(
                repeat5("v_add_f32   %0, %1, 1.0\n\t")
                : "=v"(I1) : "v"(I2));
            __asm volatile(
                repeat5("v_mul_f32   %0, %1, 5.0\n\t")
                : "=v"(I2) : "v"(I1));
	    	
			out[tid*3] = tex1Dfetch(texmem1,tid)+I1;
			out[tid*4] = tex1Dfetch(texmem2,tid)+I2;
		}
	}

}



// Host code

int main(int argc, char** argv) 
{
    int iterations;
    if (argc != 2){
        fprintf(stderr,"usage: %s #iterations\n",argv[0]);
        exit(1);
    }
    else{
        iterations = atoi(argv[1]);
    }

    printf("Power Microbenchmark with %d iterations\n",iterations);
	
 int texmem_size = LINE_SIZE*SETS*ASSOC;

 float *host_texture1 = (float*) malloc(texmem_size*sizeof(float));
 for (int i=0; i< texmem_size; i++) {
	host_texture1[i] = i;
 }
 float *device_texture1;
 float *device_texture2;
 float *device_texture3;	
 float *host_out = (float*) malloc(texmem_size*sizeof(float)*10);
 float *device_out;
 
 
 hipMalloc((void**) &device_texture1, texmem_size*sizeof(float));
 hipMalloc((void**) &device_texture2, texmem_size*sizeof(float));
 hipMalloc((void**) &device_texture3, texmem_size*sizeof(float));	

 hipMalloc((void**) &device_out, texmem_size*sizeof(float)*5);

 hipMemcpy(device_texture1, host_texture1, texmem_size*sizeof(float), hipMemcpyHostToDevice);
 hipMemcpy(device_texture2, host_texture1, texmem_size*sizeof(float), hipMemcpyHostToDevice);
 hipMemcpy(device_texture3, host_texture1, texmem_size*sizeof(float), hipMemcpyHostToDevice);
 
 hipBindTexture(0, texmem1, device_texture1, texmem_size*sizeof(float));
 hipBindTexture(0, texmem2, device_texture2, texmem_size*sizeof(float));
 hipBindTexture(0, texmem3, device_texture3, texmem_size*sizeof(float));
 
 unsigned num_blocks = (texmem_size / MAX_THREADS_PER_BLOCK) + 1;
 dim3  grid( num_blocks, 1, 1);
 dim3  threads( MAX_THREADS_PER_BLOCK, 1, 1);

 hipEvent_t start, stop;
 float elapsedTime = 0;
 checkCudaErrors(hipEventCreate(&start));
 checkCudaErrors(hipEventCreate(&stop));

 checkCudaErrors(hipEventRecord(start));
 tex_bm_kernel<<< grid, threads, 0 >>>(device_out, texmem_size, iterations);
 checkCudaErrors(hipEventRecord(stop));

 checkCudaErrors(hipEventSynchronize(stop));
  checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));
  printf("gpu execution time = %.2f s\n", elapsedTime/1000);
  getLastCudaError("kernel launch failure");
  hipDeviceSynchronize();
  checkCudaErrors(hipEventDestroy(start));
  checkCudaErrors(hipEventDestroy(stop));
  return 0;
}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}






