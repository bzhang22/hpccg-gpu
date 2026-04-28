#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_utils.h"

void gpu_waxpby(int n, float alpha, const float *d_x, float beta, const float *d_y, float *d_w);
float gpu_ddot(int n, const float *d_x, const float *d_y);
void gpu_spmv(GPU_Data *gpu, const float *d_x, float *d_y);

#endif // KERNELS_H
