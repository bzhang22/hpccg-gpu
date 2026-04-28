#!/bin/bash
#SBATCH --job-name=hpccg_b200_fp32
#SBATCH --output=hpccg_b200_fp32_%j.out
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gpus=b200:1

module purge
module load cuda/12.4.1 gcc/12.2.0

cd ~/hpccg-gpu/gpu_fp32

echo "Compiling for Blackwell (B200) in Single Precision (FP32)..."
make -f Makefile.b200 clean
make -f Makefile.b200

echo "Running HPCCG in FP32 on NVIDIA B200 at 256^3 Grid..."
./test_HPCCG_cuda 256 256 256

echo "Running HPCCG in FP32 on NVIDIA B200 at 384^3 Grid..."
./test_HPCCG_cuda 384 384 384
