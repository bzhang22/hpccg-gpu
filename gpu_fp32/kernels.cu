#include "kernels.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

__global__ void waxpby_kernel(int n, float alpha, const float *x,
                               float beta, const float *y, float *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = alpha * x[i] + beta * y[i];
    }
}

__global__ void waxpby_kernel_a1(int n, const float *x,
                                  float beta, const float *y, float *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = x[i] + beta * y[i];
    }
}

void gpu_waxpby(int n, float alpha, const float *d_x,
                float beta, const float *d_y, float *d_w) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    if (alpha == 1.0)
        waxpby_kernel_a1<<<gridSize, blockSize>>>(n, d_x, beta, d_y, d_w);
    else
        waxpby_kernel<<<gridSize, blockSize>>>(n, alpha, d_x, beta, d_y, d_w);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in waxpby_kernel: " << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void ddot_kernel(int n, const float *x, const float *y, float *result) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float temp = 0.0;
    while (i < n) {
        temp += x[i] * y[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = temp;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

float gpu_ddot(int n, const float *d_x, const float *d_y) {
    float *d_result;
    float h_result = 0.0;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));
    
    int blockSize = 256;
    int gridSize = std::min((n + blockSize - 1) / blockSize, 1024);
    ddot_kernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_result);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in ddot_kernel: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__global__ void spmv_kernel(int nrow, const int *row_ptr, const int *col_idx,
                             const float *values, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrow) {
        float sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

void gpu_spmv(GPU_Data *gpu, const float *d_x, float *d_y) {
    int blockSize = 256;
    int gridSize = (gpu->nrow + blockSize - 1) / blockSize;
    spmv_kernel<<<gridSize, blockSize>>>(
        gpu->nrow, gpu->d_row_ptr, gpu->d_col_idx, 
        gpu->d_values, d_x, d_y);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in spmv_kernel: " << cudaGetErrorString(err) << std::endl;
    }
}
