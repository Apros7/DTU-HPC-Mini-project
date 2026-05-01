#!/bin/bash
#BSUB -J task7
#BSUB -q hpc
#BSUB -W 60
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -o task7_%J.out
#BSUB -e task7_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Time for parallelized code with numba.njit jacobi. Processes:
time python simulate_task7.py 64 1
time python simulate_task7.py 64 2
time python simulate_task7.py 64 4
time python simulate_task7.py 64 8
time python simulate_task7.py 64 16
time python simulate_task7.py 64 32
