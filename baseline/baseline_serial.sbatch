#!/bin/bash
#SBATCH --job-name=hpccg-omp
#SBATCH --output=hpccg_omp_%j.out
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=00:30:00

module load gcc
cd ~/hpccg-gpu/baseline

make clean
make

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running with $OMP_NUM_THREADS threads"

./test_HPCCG 192 192 192
./test_HPCCG 256 256 256
