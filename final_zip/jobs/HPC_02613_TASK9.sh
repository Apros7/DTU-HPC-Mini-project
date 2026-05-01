#!/bin/bash
#BSUB -J task9
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 60
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o task9_%J.out
#BSUB -e task9_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Vectorised CuPy Jacobi (no fusion). One floorplan at a time on the GPU.
time python simulate_task9.py 64 1
