#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "HPC_Sparse_Matrix.hpp"

struct CSR_Matrix {
    int nrow;
    int nnz;
    int *row_ptr;      // length nrow+1
    int *col_idx;      // length nnz
    float *values;    // length nnz
};

struct GPU_Data {
    // Matrix (transferred once)
    int *d_row_ptr;
    int *d_col_idx;
    float *d_values;
    int nrow;
    int nnz;
    
    // Vectors
    float *d_x;
    float *d_r;
    float *d_p;
    float *d_Ap;
    float *d_b;
};

void convert_to_csr(HPC_Sparse_Matrix *A, CSR_Matrix *csr);
void gpu_init(GPU_Data *gpu, CSR_Matrix *csr, float *b, float *x, int ncol);
void gpu_cleanup(GPU_Data *gpu);

#endif // CUDA_UTILS_H
