#ifndef KERNELS_H
#define KERNELS_H

#include "cuda_utils.h"

void gpu_waxpby(int n, double alpha, const double *d_x, double beta, const double *d_y, double *d_w);
double gpu_ddot(int n, const double *d_x, const double *d_y);
void gpu_spmv(GPU_Data *gpu, const double *d_x, double *d_y);

#endif // KERNELS_H
