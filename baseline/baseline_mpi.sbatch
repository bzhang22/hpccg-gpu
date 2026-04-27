#!/bin/bash
#SBATCH --job-name=hpccg-mpi
#SBATCH --output=hpccg_mpi_%j.out
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=00:10:00

module load gcc openmpi
cd ~/hpccg-gpu/baseline

mpirun -np 1 ./test_HPCCG 64 64 64
mpirun -np 4 ./test_HPCCG 64 64 64
