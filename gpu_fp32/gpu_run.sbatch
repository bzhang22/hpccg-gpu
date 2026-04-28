#!/bin/bash
#SBATCH --job-name=hpccg-gpu
#SBATCH --output=hpccg_gpu_%j.out
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=00:05:00
#SBATCH --gpus=1

module load cuda/12.4.1 gcc/12.2.0
cd ~/hpccg-gpu/gpu

nvidia-smi

echo "Compiling on compute node..."
nvcc --version
make -f Makefile.cuda clean
make -f Makefile.cuda

./test_HPCCG_cuda 64 64 64
