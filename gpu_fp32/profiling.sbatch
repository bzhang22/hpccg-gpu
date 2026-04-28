#!/bin/bash
#SBATCH --job-name=hpccg-profile
#SBATCH --output=hpccg_profile_%j.out
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

nsys profile --output=hpccg_profile ./test_HPCCG_cuda 128 128 128

ncu --kernel-name spmv_kernel --launch-skip 5 --launch-count 3 \
    ./test_HPCCG_cuda 128 128 128
