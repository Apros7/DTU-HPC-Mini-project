#!/bin/bash
#BSUB -J task10
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 60
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o task10_%J.out
#BSUB -e task10_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Fused-stencil CuPy Jacobi (@cp.fuse). One floorplan at a time on the GPU.
# Second argument is ignored by the script but kept for CLI compatibility.
time python simulate_task10.py 64 1
