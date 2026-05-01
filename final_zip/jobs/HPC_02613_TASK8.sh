#!/bin/bash
#BSUB -J task8
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 60
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o task8_%J.out
#BSUB -e task8_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# numba.cuda Jacobi solver, one floorplan at a time on the GPU.
# Second argument is ignored by the script but kept for CLI compatibility
# with the CPU benchmarks (and run_all_tasks.py).
time python simulate_task8.py 64 1
