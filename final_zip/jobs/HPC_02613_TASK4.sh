#!/bin/bash
#BSUB -J task4
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=300GB]"
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -o task4_%J.out
#BSUB -e task4_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Python script, for the first 4 buildings
kernprof -l simulate.py 15