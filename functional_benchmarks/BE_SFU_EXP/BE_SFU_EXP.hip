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
#include <math.h>

// includes CUDA
#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS (608 * 1024)

// Variables
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;

// Functions
void CleanupResources(void);
void RandomInit(float*, int);

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




// Device code
__global__ void PowerKernal1(const float* A, const float* B, float* C, int iterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    //Do Some Computation
    float Value1=A[i];
    float Value2=0;
    float Value3=0;
    float Value=0;
    float I1=A[i];  
    float I2=B[i];   
    // exponential function
    for(unsigned k=0; k<iterations/100;k++) {
    #pragma unroll 100
      for(unsigned j=0; j<100;j++) {
      Value2=__expf(Value1);
      Value3=__expf(Value2);
      Value1=__expf(Value3);
      Value2=__expf(Value1);
    }
    }

  

   Value=Value3-Value2;		

    C[i]=Value;
    __syncthreads();

}

// Host code

int main(int argc, char** argv)
{
 int iterations;
 if(argc!=2) {
   fprintf(stderr,"usage: %s #iterations\n",argv[0]);
   exit(1);
 }
 else {
   iterations = atoi(argv[1]);
 }
 
 printf("Power Microbenchmarks with iterations %d\n",iterations);
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size = N * sizeof(float);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (float*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (float*)malloc(size);
 if (h_B == 0) CleanupResources();
 h_C = (float*)malloc(size);
 if (h_C == 0) CleanupResources();

 // Initialize input vectors
 RandomInit(h_A, N);
 RandomInit(h_B, N);

 // Allocate vectors in device memory
 printf("before malloc in GPU0\n");
 checkCudaErrors( hipMalloc((void**)&d_A, size) );
 printf("before malloc in GPU1\n");
 checkCudaErrors( hipMalloc((void**)&d_B, size) );
 printf("before malloc in GPU2\n");
 checkCudaErrors( hipMalloc((void**)&d_C, size) );
 printf("after malloc in GPU\n");

 hipEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(hipEventCreate(&start));  
 checkCudaErrors(hipEventCreate(&stop));   

 // Copy vectors from host memory to device memory
 checkCudaErrors( hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice) );
 checkCudaErrors( hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice) );

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);
 dim3 dimGrid2(1,1);
 dim3 dimBlock2(1,1);

 checkCudaErrors(hipEventRecord(start));              
 PowerKernal1<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, iterations);  
 checkCudaErrors(hipEventRecord(stop));               
 
 checkCudaErrors(hipEventSynchronize(stop));           
 checkCudaErrors(hipEventElapsedTime(&elapsedTime, start, stop));  
 printf("execution time = %.2f s\n", elapsedTime/1000);  
 getLastCudaError("kernel launch failure");              
 hipDeviceSynchronize();                                


 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 checkCudaErrors( hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost) );

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
  for (int i = 0; i < n; ++i)
	data[i] = rand() / (float)RAND_MAX;
}






