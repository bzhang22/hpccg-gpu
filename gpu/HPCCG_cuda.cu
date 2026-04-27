#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cmath>
#include "mytimer.hpp"
#include "HPCCG.hpp"
#include "cuda_utils.h"
#include "kernels.h"
#include <cuda_runtime.h>

#define TICK()  cudaDeviceSynchronize(); t0 = mytimer()
#define TOCK(t) cudaDeviceSynchronize(); t += mytimer() - t0

int HPCCG(HPC_Sparse_Matrix * A,
	  const double * const b, double * const x,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times)

{
  double t_begin = mytimer();  // Start timing right away

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
  int nrow = A->local_nrow;
  int ncol = A->local_ncol;

  normr = 0.0;
  double rtrans = 0.0;
  double oldrtrans = 0.0;

  int print_freq = max_iter/10; 
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;

  // === GPU Initialization ===
  CSR_Matrix csr;
  convert_to_csr(A, &csr);
  GPU_Data gpu;
  gpu_init(&gpu, &csr, (double*)b, x, ncol);

  // Initialize: p = x (on GPU)
  TICK(); gpu_waxpby(nrow, 1.0, gpu.d_x, 0.0, gpu.d_x, gpu.d_p); TOCK(t2);
  
  // Ap = A * p (on GPU)
  TICK(); gpu_spmv(&gpu, gpu.d_p, gpu.d_Ap); TOCK(t3);
  
  // r = b - Ap (on GPU, using d_b pre-loaded in gpu_init)
  TICK(); gpu_waxpby(nrow, 1.0, gpu.d_b, -1.0, gpu.d_Ap, gpu.d_r); TOCK(t2);
  
  // rtrans = r^T * r (results copied back to host)
  TICK(); rtrans = gpu_ddot(nrow, gpu.d_r, gpu.d_r); TOCK(t1);
  normr = sqrt(rtrans);

  cout << "Initial Residual = "<< normr << endl;

  for(int k=1; k<max_iter && normr > tolerance; k++ )
    {
      if (k == 1)
	{
	  TICK(); gpu_waxpby(nrow, 1.0, gpu.d_r, 0.0, gpu.d_r, gpu.d_p); TOCK(t2);
	}
      else
	{
	  oldrtrans = rtrans;
	  TICK(); rtrans = gpu_ddot (nrow, gpu.d_r, gpu.d_r); TOCK(t1);
	  double beta = rtrans/oldrtrans;
	  TICK(); gpu_waxpby (nrow, 1.0, gpu.d_r, beta, gpu.d_p, gpu.d_p);  TOCK(t2);
	}
      normr = sqrt(rtrans);
      if (k%print_freq == 0 || k+1 == max_iter)
        cout << "Iteration = "<< k << "   Residual = "<< normr << endl;
     
      TICK(); gpu_spmv(&gpu, gpu.d_p, gpu.d_Ap); TOCK(t3); 
      
      double alpha = 0.0;
      TICK(); alpha = gpu_ddot(nrow, gpu.d_p, gpu.d_Ap); TOCK(t1); 
      alpha = rtrans/alpha;
      
      TICK(); 
      gpu_waxpby(nrow, 1.0, gpu.d_x, alpha, gpu.d_p, gpu.d_x);
      gpu_waxpby(nrow, 1.0, gpu.d_r, -alpha, gpu.d_Ap, gpu.d_r);  
      TOCK(t2);
      
      niters = k;
    }

  // === Results copied back to host ===
  cudaMemcpy(x, gpu.d_x, nrow * sizeof(double), cudaMemcpyDeviceToHost);
  
  gpu_cleanup(&gpu);
  delete[] csr.row_ptr;

  // Store times
  times[1] = t1; // ddot time
  times[2] = t2; // waxpby time
  times[3] = t3; // sparsemv time
  times[4] = t4; // AllReduce time
  times[0] = mytimer() - t_begin;  // Total time. All done...
  return(0);
}
