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

#ifdef USING_MPI
#include <mpi.h>
void gpu_exchange_externals(HPC_Sparse_Matrix * A, GPU_Data * gpu, float * host_p) {
  int num_neighbors = A->num_send_neighbors;
  int * recv_length = A->recv_length;
  int * send_length = A->send_length;
  int * neighbors = A->neighbors;
  float * send_buffer = A->send_buffer;
  int total_to_be_sent = A->total_to_be_sent;
  int * elements_to_send = A->elements_to_send;
  int local_nrow = A->local_nrow;

  // 1. Copy local p from GPU to host
  cudaMemcpy(host_p, gpu->d_p, local_nrow * sizeof(float), cudaMemcpyDeviceToHost);

  // 2. Post receives
  int MPI_MY_TAG = 99;  
  MPI_Request * request = new MPI_Request[num_neighbors];
  float *x_external = host_p + local_nrow;
  for (int i = 0; i < num_neighbors; i++) {
      int n_recv = recv_length[i];
      MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+i);
      x_external += n_recv;
  }

  // 3. Pack send buffer and send
  for (int i=0; i<total_to_be_sent; i++) send_buffer[i] = host_p[elements_to_send[i]];
  for (int i = 0; i < num_neighbors; i++) {
      int n_send = send_length[i];
      MPI_Send(send_buffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
      send_buffer += n_send;
  }

  // 4. Wait for receives
  MPI_Status status;
  for (int i = 0; i < num_neighbors; i++) {
      MPI_Wait(request+i, &status);
  }
  delete [] request;

  // 5. Copy received external elements back to GPU
  int total_external = A->local_ncol - A->local_nrow;
  if (total_external > 0) {
      cudaMemcpy(gpu->d_p + local_nrow, host_p + local_nrow, total_external * sizeof(float), cudaMemcpyHostToDevice);
  }
}
#endif

#define TICK()  cudaDeviceSynchronize(); t0 = mytimer()
#define TOCK(t) cudaDeviceSynchronize(); t += mytimer() - t0

int HPCCG(HPC_Sparse_Matrix * A,
	  const float * const b, float * const x,
	  const int max_iter, const float tolerance, int &niters, float & normr,
	  float * times)

{
  float t_begin = mytimer();  // Start timing right away

  float t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
#ifdef USING_MPI
  float t5 = 0.0;
#endif
  int nrow = A->local_nrow;
  int ncol = A->local_ncol;

  normr = 0.0;
  float rtrans = 0.0;
  float oldrtrans = 0.0;

  int print_freq = max_iter/10; 
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;

  // === GPU Initialization ===
  CSR_Matrix csr;
  convert_to_csr(A, &csr);
  GPU_Data gpu;
  gpu_init(&gpu, &csr, (float*)b, x, ncol);

  float *host_p = new float[ncol];

  // Initialize: p = x (on GPU)
  TICK(); gpu_waxpby(nrow, 1.0, gpu.d_x, 0.0, gpu.d_x, gpu.d_p); TOCK(t2);
  
#ifdef USING_MPI
  TICK(); gpu_exchange_externals(A, &gpu, host_p); TOCK(t5);
#endif
  // Ap = A * p (on GPU)
  TICK(); gpu_spmv(&gpu, gpu.d_p, gpu.d_Ap); TOCK(t3);
  
  // r = b - Ap (on GPU, using d_b pre-loaded in gpu_init)
  TICK(); gpu_waxpby(nrow, 1.0, gpu.d_b, -1.0, gpu.d_Ap, gpu.d_r); TOCK(t2);
  
  // rtrans = r^T * r (results copied back to host)
  TICK(); rtrans = gpu_ddot(nrow, gpu.d_r, gpu.d_r); TOCK(t1);
#ifdef USING_MPI
  float local_rtrans1 = rtrans;
  TICK(); MPI_Allreduce(&local_rtrans1, &rtrans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); TOCK(t4);
#endif
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
#ifdef USING_MPI
      float local_rtrans2 = rtrans;
      TICK(); MPI_Allreduce(&local_rtrans2, &rtrans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); TOCK(t4);
#endif
	  float beta = rtrans/oldrtrans;
	  TICK(); gpu_waxpby (nrow, 1.0, gpu.d_r, beta, gpu.d_p, gpu.d_p);  TOCK(t2);
	}
      normr = sqrt(rtrans);
      int rank = 0;
#ifdef USING_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
      if (rank == 0 && (k%print_freq == 0 || k+1 == max_iter))
        cout << "Iteration = "<< k << "   Residual = "<< normr << endl;
     
#ifdef USING_MPI
      TICK(); gpu_exchange_externals(A, &gpu, host_p); TOCK(t5);
#endif
      TICK(); gpu_spmv(&gpu, gpu.d_p, gpu.d_Ap); TOCK(t3); 
      
      float alpha = 0.0;
      TICK(); alpha = gpu_ddot(nrow, gpu.d_p, gpu.d_Ap); TOCK(t1); 
#ifdef USING_MPI
      float local_alpha = alpha;
      TICK(); MPI_Allreduce(&local_alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); TOCK(t4);
#endif
      alpha = rtrans/alpha;
      
      TICK(); 
      gpu_waxpby(nrow, 1.0, gpu.d_x, alpha, gpu.d_p, gpu.d_x);
      gpu_waxpby(nrow, 1.0, gpu.d_r, -alpha, gpu.d_Ap, gpu.d_r);  
      TOCK(t2);
      
      niters = k;
    }

  // === Results copied back to host ===
  cudaMemcpy(x, gpu.d_x, nrow * sizeof(float), cudaMemcpyDeviceToHost);
  
  gpu_cleanup(&gpu);
  delete[] csr.row_ptr;
  delete[] host_p;

  // Store times
  times[1] = t1; // ddot time
  times[2] = t2; // waxpby time
  times[3] = t3; // sparsemv time
  times[4] = t4; // AllReduce time
#ifdef USING_MPI
  times[5] = t5; // exchange boundary time
#endif
  times[0] = mytimer() - t_begin;  // Total time. All done...
  return(0);
}
