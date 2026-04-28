#!/bin/bash
#SBATCH --job-name=hpccg_gpu_2
#SBATCH --output=hpccg_gpu_2_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --gpus=2

module purge
module load cuda/12.4.1 gcc/12.2.0 openmpi

make -f Makefile.cuda clean
make -f Makefile.cuda

mpirun -np 2 ./test_HPCCG_cuda 256 256 256
