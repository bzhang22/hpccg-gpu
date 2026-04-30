#!/bin/bash
#SBATCH --job-name=hpccg_demo
#SBATCH --output=demo_output_%j.out
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gpus=b200:1

echo "================================================="
echo "  HPCCG GPU Acceleration & Mixed Precision DEMO  "
echo "================================================="
echo ""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "Detected GPU: ${GPU_NAME}"
echo ""

# Setup
HPCCG_DIR=/home/bohanzhang1/hpccg-gpu
module purge
module load cuda/12.4.1 gcc/12.2.0

echo "Step 1: Compiling Codebases..."
echo "Compiling CPU Baseline..."
cd $HPCCG_DIR/baseline && make clean > /dev/null 2>&1 && make > /dev/null 2>&1

echo "Compiling GPU FP64 (Double Precision)..."
cd $HPCCG_DIR/gpu && make -f Makefile.b200 clean > /dev/null 2>&1 && make -f Makefile.b200 > /dev/null 2>&1

echo "Compiling GPU FP32 (Single Precision)..."
cd $HPCCG_DIR/gpu_fp32 && make -f Makefile.b200 clean > /dev/null 2>&1 && make -f Makefile.b200 > /dev/null 2>&1
echo "Compilation Successful!"
echo ""

echo "Step 2: Running CPU Baseline (Serial) - 128x128x128..."
cd $HPCCG_DIR/baseline
./test_HPCCG 128 128 128 > cpu_out.txt
CPU_TIME=$(grep "SPARSEMV:" cpu_out.txt | awk '{print $2}')
echo "CPU SPARSEMV Time: ${CPU_TIME} seconds"
echo ""

echo "Step 3: Running GPU FP64 (Double Precision) - 128x128x128..."
cd $HPCCG_DIR/gpu
./test_HPCCG_cuda 128 128 128 > gpu_fp64_out.txt
FP64_TIME=$(grep "SPARSEMV:" gpu_fp64_out.txt | awk '{print $2}')
echo "GPU FP64 SPARSEMV Time: ${FP64_TIME} seconds"
SPEEDUP=$(echo "scale=2; $CPU_TIME / $FP64_TIME" | bc)
echo "=> Speedup vs CPU: ${SPEEDUP}x"
echo ""

echo "Step 4: Running GPU FP32 (Single Precision) - 128x128x128..."
cd $HPCCG_DIR/gpu_fp32
./test_HPCCG_cuda 128 128 128 > gpu_fp32_out.txt
FP32_TIME=$(grep "SPARSEMV:" gpu_fp32_out.txt | awk '{print $2}')
echo "GPU FP32 SPARSEMV Time: ${FP32_TIME} seconds"
SPEEDUP_FP32=$(echo "scale=2; $CPU_TIME / $FP32_TIME" | bc)
echo "=> Speedup vs CPU: ${SPEEDUP_FP32}x"
SPEEDUP_FP64=$(echo "scale=2; $FP64_TIME / $FP32_TIME" | bc)
echo "=> Relative Speedup (FP32 vs FP64): ${SPEEDUP_FP64}x (Theoretical Memory-Bound limit is ~1.5x!)"
echo ""

echo "Step 5: Verifying FP64 Mathematical Correctness (64x64x64)..."
echo "Running CPU 64^3 to dump x_cpu.txt..."
cd $HPCCG_DIR/baseline && ./test_HPCCG 64 64 64 > /dev/null
echo "Running GPU 64^3 to dump x_gpu.txt..."
cd $HPCCG_DIR/gpu && ./test_HPCCG_cuda 64 64 64 > /dev/null
echo "Calculating Mean Absolute Error (MAE)..."
cd $HPCCG_DIR
python3 compute_mae.py baseline/x_cpu.txt gpu/x_gpu.txt

echo ""
echo "================================================="
echo "                 DEMO COMPLETED!                 "
echo "================================================="
