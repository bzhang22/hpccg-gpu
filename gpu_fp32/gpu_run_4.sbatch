#!/bin/bash
#SBATCH --job-name=hpccg_gpu_4
#SBATCH --output=hpccg_gpu_4_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --gpus-per-node=2

module purge
module load cuda/12.4.1 gcc/12.2.0 openmpi

mpirun -np 4 ./test_HPCCG_cuda 256 256 256
