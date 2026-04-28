#!/bin/bash
#SBATCH --job-name=hpccg_b200
#SBATCH --output=hpccg_b200_%j.out
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --gpus=b200:1

module purge
module load cuda/12.4.1 gcc/12.2.0

cd ~/hpccg-gpu/gpu

echo "Compiling for Blackwell (B200)..."
make -f Makefile.b200 clean
make -f Makefile.b200

nvidia-smi

echo "Running HPCCG on NVIDIA B200 at 256^3 Grid..."
./test_HPCCG_cuda 256 256 256

echo "Running HPCCG on NVIDIA B200 at 384^3 Grid..."
./test_HPCCG_cuda 384 384 384
