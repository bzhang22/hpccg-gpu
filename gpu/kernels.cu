#include "kernels.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

__global__ void waxpby_kernel(int n, double alpha, const double *x,
                               double beta, const double *y, double *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = alpha * x[i] + beta * y[i];
    }
}

__global__ void waxpby_kernel_a1(int n, const double *x,
                                  double beta, const double *y, double *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] = x[i] + beta * y[i];
    }
}

void gpu_waxpby(int n, double alpha, const double *d_x,
                double beta, const double *d_y, double *d_w) {
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

__global__ void ddot_kernel(int n, const double *x, const double *y, double *result) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double temp = 0.0;
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
        volatile double *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    
    if (tid == 0) atomicAdd(result, sdata[0]);
}

double gpu_ddot(int n, const double *d_x, const double *d_y) {
    double *d_result;
    double h_result = 0.0;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemset(d_result, 0, sizeof(double));
    
    int blockSize = 256;
    int gridSize = std::min((n + blockSize - 1) / blockSize, 1024);
    ddot_kernel<<<gridSize, blockSize>>>(n, d_x, d_y, d_result);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in ddot_kernel: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}

__global__ void spmv_kernel(int nrow, const int *row_ptr, const int *col_idx,
                             const double *values, const double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrow) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

void gpu_spmv(GPU_Data *gpu, const double *d_x, double *d_y) {
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
