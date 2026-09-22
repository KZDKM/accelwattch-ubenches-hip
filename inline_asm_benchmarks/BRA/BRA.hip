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
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h> //BT: Needed for uint32_t
#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (304*1024)
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



__global__ void PowerKernal2(unsigned* A, unsigned* B, unsigned long long N)
{
    uint32_t uid = blockDim.x * blockIdx.x + threadIdx.x;
    volatile unsigned sink = A[uid];
    unsigned iter = N;
    unsigned out;
asm volatile (
    //".reg .b32 r1;\n"        // Declare a 32-bit register
    //".reg .pred p;\n"        // Declare a predicate register
    // pred corresponds to vcc in AMD which does not need declaration here?
    
    // %0 corresponds to r1, writable SGPR that stores output
    // %1 corresponds to %0 before

    "s_mov_b32 %0, %1\n\t"      // Move the value of 'iterations' into r1

    "loop_alpha:\n\t"
    "s_sub_u32 %0, %0, 1\n\t"   // Decrement r1 (loop counter)
    "s_branch loop_1\n\t"

    // Begin numbered loop chain (loop_1 to loop_50)
    // just let LLM formatted this branch chain part cuz I aint rewriting all that
    "loop_1:\n\t"  "s_branch loop_2\n\t"
    "loop_2:\n\t"  "s_branch loop_3\n\t"
    "loop_3:\n\t"  "s_branch loop_4\n\t"
    "loop_4:\n\t"  "s_branch loop_5\n\t"
    "loop_5:\n\t"  "s_branch loop_6\n\t"
    "loop_6:\n\t"  "s_branch loop_7\n\t"
    "loop_7:\n\t"  "s_branch loop_8\n\t"
    "loop_8:\n\t"  "s_branch loop_9\n\t"
    "loop_9:\n\t"  "s_branch loop_10\n\t"
    "loop_10:\n\t" "s_branch loop_11\n\t"
    "loop_11:\n\t" "s_branch loop_12\n\t"
    "loop_12:\n\t" "s_branch loop_13\n\t"
    "loop_13:\n\t" "s_branch loop_14\n\t"
    "loop_14:\n\t" "s_branch loop_15\n\t"
    "loop_15:\n\t" "s_branch loop_16\n\t"
    "loop_16:\n\t" "s_branch loop_17\n\t"
    "loop_17:\n\t" "s_branch loop_18\n\t"
    "loop_18:\n\t" "s_branch loop_19\n\t"
    "loop_19:\n\t" "s_branch loop_20\n\t"
    "loop_20:\n\t" "s_branch loop_21\n\t"
    "loop_21:\n\t" "s_branch loop_22\n\t"
    "loop_22:\n\t" "s_branch loop_23\n\t"
    "loop_23:\n\t" "s_branch loop_24\n\t"
    "loop_24:\n\t" "s_branch loop_25\n\t"
    "loop_25:\n\t" "s_branch loop_26\n\t"
    "loop_26:\n\t" "s_branch loop_27\n\t"
    "loop_27:\n\t" "s_branch loop_28\n\t"
    "loop_28:\n\t" "s_branch loop_29\n\t"
    "loop_29:\n\t" "s_branch loop_30\n\t"
    "loop_30:\n\t" "s_branch loop_31\n\t"
    "loop_31:\n\t" "s_branch loop_32\n\t"
    "loop_32:\n\t" "s_branch loop_33\n\t"
    "loop_33:\n\t" "s_branch loop_34\n\t"
    "loop_34:\n\t" "s_branch loop_35\n\t"
    "loop_35:\n\t" "s_branch loop_36\n\t"
    "loop_36:\n\t" "s_branch loop_37\n\t"
    "loop_37:\n\t" "s_branch loop_38\n\t"
    "loop_38:\n\t" "s_branch loop_39\n\t"
    "loop_39:\n\t" "s_branch loop_40\n\t"
    "loop_40:\n\t" "s_branch loop_41\n\t"
    "loop_41:\n\t" "s_branch loop_42\n\t"
    "loop_42:\n\t" "s_branch loop_43\n\t"
    "loop_43:\n\t" "s_branch loop_44\n\t"
    "loop_44:\n\t" "s_branch loop_45\n\t"
    "loop_45:\n\t" "s_branch loop_46\n\t"
    "loop_46:\n\t" "s_branch loop_47\n\t"
    "loop_47:\n\t" "s_branch loop_48\n\t"
    "loop_48:\n\t" "s_branch loop_49\n\t"
    "loop_49:\n\t" "s_branch loop_50\n\t"

    // End numbered loop chain
    "loop_50:\n\t"
    //"setp.ne.u32 p, r1, 0;\n" // Set predicate if r1 != 0
    "s_cmp_lg_u32 %0, 0\n\t"
    //"@p bra loop_alpha;\n"     // Branch back to loop_alpha if r1 != 0
    "s_cbranch_scc1 loop_alpha\n\t"
    : "=r"(out)
    : "r"(iter)                  // Input: iterations
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
 checkCudaErrors( hipMalloc((void**)&d_A, size) );
 checkCudaErrors( hipMalloc((void**)&d_B, size) );


 // Copy vector from host memory to device memory
 checkCudaErrors( hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice) );


 hipEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(hipEventCreate(&start));  
 checkCudaErrors(hipEventCreate(&stop));

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);


 checkCudaErrors(hipEventRecord(start));              
 PowerKernal2<<<dimGrid,dimBlock>>>(d_A, d_B,iterations);  
 checkCudaErrors(hipEventRecord(stop));               
 
 checkCudaErrors(hipEventSynchronize(stop));           
 checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
 getLastCudaError("kernel launch failure");              

 // Copy result from device memory to host memory
 // h_B contains the result in host memory
 checkCudaErrors( hipMemcpy(h_B, d_B, size, hipMemcpyDeviceToHost) );
  checkCudaErrors(hipEventDestroy(start));
 checkCudaErrors(hipEventDestroy(stop));
 CleanupResources();

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
  hipFree(d_A);
  if (d_B)
  hipFree(d_B);

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
