#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>

#ifdef USING_MPI
#include <mpi.h>
#endif

void convert_to_csr(HPC_Sparse_Matrix *A, CSR_Matrix *csr) {
    csr->nrow = A->local_nrow;
    csr->row_ptr = new int[csr->nrow + 1];
    
    // Build row_ptr
    csr->row_ptr[0] = 0;
    for (int i = 0; i < csr->nrow; i++) {
        csr->row_ptr[i+1] = csr->row_ptr[i] + A->nnz_in_row[i];
    }
    csr->nnz = csr->row_ptr[csr->nrow];
    
    // list_of_vals and list_of_inds are already contiguous
    csr->values = A->list_of_vals;
    csr->col_idx = A->list_of_inds;
}

void gpu_init(GPU_Data *gpu, CSR_Matrix *csr, float *b, float *x, int ncol) {
#ifdef USING_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);
#else
    cudaSetDevice(0);
#endif

    int nrow = csr->nrow;
    int nnz = csr->nnz;
    
    cudaMalloc(&gpu->d_row_ptr, (nrow+1) * sizeof(int));
    cudaMalloc(&gpu->d_col_idx, nnz * sizeof(int));
    cudaMalloc(&gpu->d_values, nnz * sizeof(float));
    cudaMemcpy(gpu->d_row_ptr, csr->row_ptr, (nrow+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_col_idx, csr->col_idx, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_values, csr->values, nnz*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&gpu->d_x, nrow * sizeof(float));
    cudaMalloc(&gpu->d_r, nrow * sizeof(float));
    cudaMalloc(&gpu->d_p, ncol * sizeof(float));
    cudaMalloc(&gpu->d_Ap, nrow * sizeof(float));
    cudaMalloc(&gpu->d_b, nrow * sizeof(float)); // allocate b to initialize r
    
    cudaMemcpy(gpu->d_x, x, nrow*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu->d_b, b, nrow*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in gpu_init: " << cudaGetErrorString(err) << std::endl;
    }
    
    gpu->nrow = nrow;
    gpu->nnz = nnz;
}

void gpu_cleanup(GPU_Data *gpu) {
    cudaFree(gpu->d_row_ptr);
    cudaFree(gpu->d_col_idx);
    cudaFree(gpu->d_values);
    cudaFree(gpu->d_x);
    cudaFree(gpu->d_r);
    cudaFree(gpu->d_p);
    cudaFree(gpu->d_Ap);
    cudaFree(gpu->d_b);
}
