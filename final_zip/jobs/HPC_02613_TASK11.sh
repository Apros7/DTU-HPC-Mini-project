#!/bin/bash
#BSUB -J task11-8
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 60
#BSUB -R "rusage[mem=40GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o task11-8_%J.out
#BSUB -e task11-8_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Batched CuPy Jacobi: custom ElementwiseKernel + ping-pong buffers + fp32
# state. First run: small benchmark (matches the comparison plot at N=64).
time python simulate_task11-8.py 64 1

# Second run: full population of 4571 floorplans. Produces
# results/task11-8_stats.csv used by analyze_task11_8_stats.py for the
# histograms and summary numbers in exercise 12.
time python simulate_task11-8.py 4571 1
