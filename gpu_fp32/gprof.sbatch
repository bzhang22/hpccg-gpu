#!/bin/bash
#SBATCH --job-name=hpccg_gprof
#SBATCH --output=hpccg_gprof_%j.out
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gpus=1

module purge
module load cuda/12.4.1 gcc/12.2.0

cd ~/hpccg-gpu/gpu

echo "Compiling with gprof flags..."
make -f Makefile.gprof clean
make -f Makefile.gprof

echo "Running HPCCG to generate gmon.out..."
./test_HPCCG_cuda 64 64 64

echo "Generating gprof report..."
gprof ./test_HPCCG_cuda gmon.out > gprof_report.txt

echo "Done."
