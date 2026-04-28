# HPCCG Supplementary Experiments

This document contains the exact commands used to run the supplementary benchmark experiments (P0, P1, P2) for our analysis.

## P0: Serial CPU 256³ Baseline (Pure Serial, No OpenMP)
To establish a baseline without multi-threading overhead, OpenMP was disabled in `baseline/Makefile`:
```bash
cd baseline/
make clean && make
./test_HPCCG 256 256 256
```

## P1: B200 FP64 64³ Test
To collect small-grid dual-precision baseline data on the Blackwell architecture:
```bash
cd gpu/
./test_HPCCG_cuda 64 64 64
```

## P1: L4 FP32 128³ Test
To capture mid-grid single-precision (FP32) performance on the Ada Lovelace L4 architecture:
```bash
cd gpu_fp32/
./test_HPCCG_cuda 128 128 128
```

## P1: Correctness Validation (MAE FP64 vs FP64)
To scientifically prove that the GPU double-precision (FP64) execution is mathematically identical to the CPU implementation, we dumped the solution vector `x` for a $64^3$ grid on both platforms and computed the Mean Absolute Error (MAE):
```bash
python3 compute_mae.py baseline/x_cpu.txt gpu/x_gpu.txt
# Result: Mean Absolute Error (MAE): 0.000000e+00 (Bit-for-bit identical)
```

## P2: Nsight Compute Profiling
To conduct an extremely deep micro-architectural analysis of the Conjugate Gradient solver on a $256^3$ grid, capturing Memory Throughput, SM Occupancy, and Warp stalls:
```bash
cd gpu/
ncu --set full ./test_HPCCG_cuda 256 256 256 > ncu_full_report.txt
```
