#!/bin/bash
# HPCCG Interactive Live Demo Script
# Ensure this is run on a GPU compute node!

# Terminal Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}=================================================${NC}"
echo -e "${CYAN}  HPCCG GPU Acceleration & Mixed Precision DEMO  ${NC}"
echo -e "${CYAN}=================================================${NC}"
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: No GPU detected! You are likely on a login node.${NC}"
    echo -e "Please run the following command to get an interactive GPU node first:"
    echo -e "${YELLOW}srun --partition=hpg-b200 --gpus=b200:1 --pty bash -i${NC}"
    echo -e "or for L4: ${YELLOW}srun --partition=hpg-default --gpus=1 --pty bash -i${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo -e "${GREEN}Detected GPU: ${GPU_NAME}${NC}"
echo ""

# Setup
HPCCG_DIR=$(realpath "$(dirname "$0")/..")
module purge
module load cuda/12.4.1 gcc/12.2.0

echo -e "${YELLOW}Step 1: Compiling Codebases...${NC}"
echo "Compiling CPU Baseline..."
cd $HPCCG_DIR/baseline && make clean > /dev/null 2>&1 && make > /dev/null 2>&1

echo "Compiling GPU FP64 (Double Precision)..."
cd $HPCCG_DIR/gpu && make -f Makefile.b200 clean > /dev/null 2>&1 && make -f Makefile.b200 > /dev/null 2>&1

echo "Compiling GPU FP32 (Single Precision)..."
cd $HPCCG_DIR/gpu_fp32 && make -f Makefile.b200 clean > /dev/null 2>&1 && make -f Makefile.b200 > /dev/null 2>&1
echo -e "${GREEN}Compilation Successful!${NC}"
echo ""

read -p "Press [Enter] to run the CPU Baseline (128^3 Grid)..."

echo -e "\n${YELLOW}Step 2: Running CPU Baseline (Serial) - 128x128x128${NC}"
cd $HPCCG_DIR/baseline
./test_HPCCG 128 128 128 > cpu_out.txt
CPU_TIME=$(awk '/SPARSEMV:/ {print $2; exit}' cpu_out.txt)
echo -e "CPU SPARSEMV Time: ${RED}${CPU_TIME} seconds${NC}"

read -p "Press [Enter] to run the GPU FP64 Version..."

echo -e "\n${YELLOW}Step 3: Running GPU FP64 (Double Precision) - 128x128x128${NC}"
cd $HPCCG_DIR/gpu
./test_HPCCG_cuda 128 128 128 > gpu_fp64_out.txt
FP64_TIME=$(awk '/SPARSEMV:/ {print $2; exit}' gpu_fp64_out.txt)
echo -e "GPU FP64 SPARSEMV Time: ${GREEN}${FP64_TIME} seconds${NC}"
SPEEDUP=$(echo "scale=2; $CPU_TIME / $FP64_TIME" | bc)
echo -e "=> Speedup vs CPU: ${GREEN}${SPEEDUP}x${NC}"

read -p "Press [Enter] to run the GPU FP32 Version (Mixed Precision Ablation)..."

echo -e "\n${YELLOW}Step 4: Running GPU FP32 (Single Precision) - 128x128x128${NC}"
cd $HPCCG_DIR/gpu_fp32
./test_HPCCG_cuda 128 128 128 > gpu_fp32_out.txt
FP32_TIME=$(awk '/SPARSEMV:/ {print $2; exit}' gpu_fp32_out.txt)
echo -e "GPU FP32 SPARSEMV Time: ${CYAN}${FP32_TIME} seconds${NC}"
SPEEDUP_FP32=$(echo "scale=2; $CPU_TIME / $FP32_TIME" | bc)
echo -e "=> Speedup vs CPU: ${CYAN}${SPEEDUP_FP32}x${NC}"
SPEEDUP_FP64=$(echo "scale=2; $FP64_TIME / $FP32_TIME" | bc)
echo -e "=> Relative Speedup (FP32 vs FP64): ${CYAN}${SPEEDUP_FP64}x${NC} (Theoretical Memory-Bound limit is ~1.5x!)"

read -p "Press [Enter] to run Mathematical Correctness Validation (MAE)..."

echo -e "\n${YELLOW}Step 5: Verifying FP64 Mathematical Correctness (64x64x64)${NC}"
echo "Running CPU 64^3 to dump x_cpu.txt..."
cd $HPCCG_DIR/baseline && ./test_HPCCG 64 64 64 > /dev/null
echo "Running GPU 64^3 to dump x_gpu.txt..."
cd $HPCCG_DIR/gpu && ./test_HPCCG_cuda 64 64 64 > /dev/null
echo "Calculating Mean Absolute Error (MAE)..."
cd $HPCCG_DIR
python3 compute_mae.py baseline/x_cpu.txt gpu/x_gpu.txt

echo -e "\n${GREEN}=================================================${NC}"
echo -e "${GREEN}                 DEMO COMPLETED!                 ${NC}"
echo -e "${GREEN}=================================================${NC}"
