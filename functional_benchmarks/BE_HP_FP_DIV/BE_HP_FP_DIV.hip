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
// Includes
//#include <stdio.h>

// includes, project
//#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <hip/hip_runtime.h>

//Get Half precision in scope
#include <rocwmma/rocwmma.hpp>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (608 * 1024)
//#define ITERATIONS 40
//#include "../include/ContAcq-IntClk.h"

// Variables
float* h_A;
float* h_B;
half* h_C;
float* d_A_fp32;
float* d_B_fp32;
half* d_A;
half* d_B;
half* d_C;
//bool noprompt = false;
//unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(float*, int);
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

//half precision written for device
__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}
// end of CUDA Helper Functions




// Device code
__global__ void PowerKernal1(const half* A, const half* B, half* C, unsigned long long iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    half Value1=0;
    half Value2=999;
    half Value3=0;
    half Value=0;
    half I1=A[i];
    half I2=B[i];
    // Excessive Addition access
    for(unsigned long long k=0; k<iterations/100;k++) {
#pragma unroll 100
      for(unsigned long long j=0; j<100;j++) {
  Value1=I1/I2;
  Value3=I1/I2;
  Value1/=Value2;
  Value1/=Value2;
  Value2=Value3/Value1;
  Value1=Value2/Value3;

    }
    }

    __syncthreads();
    Value=Value1;
    C[i]=Value/Value2;

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
 size_t size = N * sizeof(float);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (float*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (float*)malloc(size);
 if (h_B == 0) CleanupResources();
 h_C = (half*)malloc(size);
 if (h_C == 0) CleanupResources();

 // Initialize input vectors
 RandomInit(h_A, N);
 RandomInit(h_B, N);

 // Allocate vectors in device memory
printf("before\n");
 checkCudaErrors( hipMalloc((void**)&d_A, N * sizeof(half)) );
 checkCudaErrors( hipMalloc((void**)&d_B, N * sizeof(half)) );
 checkCudaErrors( hipMalloc((void**)&d_C, N * sizeof(half)) );
 checkCudaErrors( hipMalloc((void**)&d_A_fp32, size) );
 checkCudaErrors( hipMalloc((void**)&d_B_fp32, size) );
printf("after\n");

 hipEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(hipEventCreate(&start));  
 checkCudaErrors(hipEventCreate(&stop));
 
 // Copy vectors from host memory to device memory
 checkCudaErrors( hipMemcpy(d_A_fp32, h_A, size, hipMemcpyHostToDevice) );
 checkCudaErrors( hipMemcpy(d_B_fp32, h_B, size, hipMemcpyHostToDevice) );
 
 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);
 dim3 dimGrid2(1,1);
 dim3 dimBlock2(1,1);

 //Convert data from float to half
 convertFp32ToFp16<<<dimGrid,dimBlock>>>(d_A,d_A_fp32,N);
 convertFp32ToFp16<<<dimGrid,dimBlock>>>(d_B,d_B_fp32,N);

 checkCudaErrors(hipEventRecord(start));              
 PowerKernal1<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, iterations);  
 checkCudaErrors(hipEventRecord(stop));               
 
 checkCudaErrors(hipEventSynchronize(stop));           
 checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
 getLastCudaError("kernel launch failure");              
 hipDeviceSynchronize();

/*CUT_SAFE_CALL(cutCreateTimer(&my_timer)); 
TaskHandle taskhandle = LaunchDAQ();
CUT_SAFE_CALL(cutStartTimer(my_timer)); 
printf("execution time = %f\n", cutGetTimerValue(my_timer));



PowerKernal1<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N);
CUDA_SAFE_CALL( hipDeviceSynchronize() );
printf("execution time = %f\n", cutGetTimerValue(my_timer));


getLastCudaError("kernel launch failure");
CUDA_SAFE_CALL( hipDeviceSynchronize() );
CUT_SAFE_CALL(cutStopTimer(my_timer));
TurnOffDAQ(taskhandle, cutGetTimerValue(my_timer));
printf("execution time = %f\n", cutGetTimerValue(my_timer));
CUT_SAFE_CALL(cutDeleteTimer(my_timer)); 

#ifdef _DEBUG
 checkCudaErrors( hipDeviceSynchronize() );
#endif*/

 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 // checkCudaErrors( hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost) );
 
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
  if (d_C)
	hipFree(d_C);
  if (d_A_fp32)
	hipFree(d_A_fp32);
  if (d_B_fp32)
	hipFree(d_B_fp32);

  // Free host memory
  if (h_A)
	free(h_A);
  if (h_B)
	free(h_B);
  if (h_C)
	free(h_C);

}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
  for (int i = 0; i < n; ++i){ 
	data[i] = rand() / RAND_MAX;
  }
}






