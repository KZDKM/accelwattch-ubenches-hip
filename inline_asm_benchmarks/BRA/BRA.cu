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
#include <stdio.h>
#include <stdlib.h>
//#include <cutil.h>
//#include <mgp.h>
// Includes
//#include <stdio.h>
//#include "../include/ContAcq-IntClk.h"

// includes, project
//#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>
#include <cuda.h> //BT: Needed for uint32_t
#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 640
// Variables
unsigned* h_A;
unsigned* h_B;
unsigned* d_A;
unsigned* d_B;
//bool noprompt = false;
//unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(unsigned*, int);
//void ParseArguments(int, char**);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
  if(cudaSuccess != err){
  fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
   exit(-1);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
  fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
  exit(-1);
  }
}

// end of CUDA Helper Functions



__global__ void PowerKernal2(unsigned* A, unsigned* B, unsigned long long N)
{
    uint32_t uid = blockDim.x * blockIdx.x + threadIdx.x;
    volatile unsigned sink = A[uid];
    unsigned iter = N;
asm volatile (
    ".reg .b32 r1;\n"        // Declare a 32-bit register
    ".reg .pred p;\n"        // Declare a predicate register

    "mov.u32 r1, %0;\n"      // Move the value of 'iterations' into r1

    "loop_alpha:\n"
    "sub.u32 r1, r1, 1;\n"   // Decrement r1 (loop counter)
    "bra loop_1;\n"

    // Begin numbered loop chain (loop_1 to loop_50)
    "loop_1:\n"
    "bra loop_2;\n"
    
    "loop_2:\n"
    "bra loop_3;\n"
    
    "loop_3:\n"
    "bra loop_4;\n"
    
    "loop_4:\n"
    "bra loop_5;\n"
    
    "loop_5:\n"
    "bra loop_6;\n"
    
    "loop_6:\n"
    "bra loop_7;\n"
    
    "loop_7:\n"
    "bra loop_8;\n"
    
    "loop_8:\n"
    "bra loop_9;\n"
    
    "loop_9:\n"
    "bra loop_10;\n"
    
    "loop_10:\n"
    "bra loop_11;\n"
    
    "loop_11:\n"
    "bra loop_12;\n"
    
    "loop_12:\n"
    "bra loop_13;\n"
    
    "loop_13:\n"
    "bra loop_14;\n"
    
    "loop_14:\n"
    "bra loop_15;\n"
    
    "loop_15:\n"
    "bra loop_16;\n"
    
    "loop_16:\n"
    "bra loop_17;\n"
    
    "loop_17:\n"
    "bra loop_18;\n"
    
    "loop_18:\n"
    "bra loop_19;\n"
    
    "loop_19:\n"
    "bra loop_20;\n"
    
    "loop_20:\n"
    "bra loop_21;\n"
    
    "loop_21:\n"
    "bra loop_22;\n"
    
    "loop_22:\n"
    "bra loop_23;\n"
    
    "loop_23:\n"
    "bra loop_24;\n"
    
    "loop_24:\n"
    "bra loop_25;\n"
    
    "loop_25:\n"
    "bra loop_26;\n"
    
    "loop_26:\n"
    "bra loop_27;\n"
    
    "loop_27:\n"
    "bra loop_28;\n"
    
    "loop_28:\n"
    "bra loop_29;\n"
    
    "loop_29:\n"
    "bra loop_30;\n"
    
    "loop_30:\n"
    "bra loop_31;\n"
    
    "loop_31:\n"
    "bra loop_32;\n"
    
    "loop_32:\n"
    "bra loop_33;\n"
    
    "loop_33:\n"
    "bra loop_34;\n"
    
    "loop_34:\n"
    "bra loop_35;\n"
    
    "loop_35:\n"
    "bra loop_36;\n"
    
    "loop_36:\n"
    "bra loop_37;\n"
    
    "loop_37:\n"
    "bra loop_38;\n"
    
    "loop_38:\n"
    "bra loop_39;\n"
    
    "loop_39:\n"
    "bra loop_40;\n"
    
    "loop_40:\n"
    "bra loop_41;\n"
    
    "loop_41:\n"
    "bra loop_42;\n"
    
    "loop_42:\n"
    "bra loop_43;\n"
    
    "loop_43:\n"
    "bra loop_44;\n"
    
    "loop_44:\n"
    "bra loop_45;\n"
    
    "loop_45:\n"
    "bra loop_46;\n"
    
    "loop_46:\n"
    "bra loop_47;\n"
    
    "loop_47:\n"
    "bra loop_48;\n"
    
    "loop_48:\n"
    "bra loop_49;\n"
    
    "loop_49:\n"
    "bra loop_50;\n"

    // End numbered loop chain
    "loop_50:\n"
    "setp.ne.u32 p, r1, 0;\n" // Set predicate if r1 != 0
    "@p bra loop_alpha;\n"     // Branch back to loop_alpha if r1 != 0
    ::
    "r"(iter)                  // Input: iterations
);
    B[uid] = sink;
}


int main(int argc, char** argv)
{
 unsigned long long iterations;
 if(argc!=2) {
   fprintf(stderr,"usage: %s #iterations\n",argv[0]);
   exit(1);
 }
 else {
   iterations = atoll(argv[1]);
 }
 
 printf("Power Microbenchmarks with iterations %lld\n",iterations);
 
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size = N * sizeof(unsigned);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (unsigned*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (unsigned*)malloc(size);
 if (h_B == 0) CleanupResources();


 // Initialize input vectors
 RandomInit(h_A, N);


 // Allocate vectors in device memory
 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
 checkCudaErrors( cudaMalloc((void**)&d_B, size) );


 // Copy vector from host memory to device memory
 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );


 cudaEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(cudaEventCreate(&start));  
 checkCudaErrors(cudaEventCreate(&stop));

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);


 checkCudaErrors(cudaEventRecord(start));              
 PowerKernal2<<<dimGrid,dimBlock>>>(d_A, d_B,iterations);  
 checkCudaErrors(cudaEventRecord(stop));               
 
 checkCudaErrors(cudaEventSynchronize(stop));           
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
 getLastCudaError("kernel launch failure");              

 // Copy result from device memory to host memory
 // h_B contains the result in host memory
 checkCudaErrors( cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost) );
  checkCudaErrors(cudaEventDestroy(start));
 checkCudaErrors(cudaEventDestroy(stop));
 CleanupResources();

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
  cudaFree(d_A);
  if (d_B)
  cudaFree(d_B);

  // Free host memory
  if (h_A)
  free(h_A);
  if (h_B)
  free(h_B);

}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
  srand((unsigned)time(0));  
  data[i] = rand() / RAND_MAX;
  }
}
