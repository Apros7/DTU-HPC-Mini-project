#!/bin/bash
#BSUB -J task2
#BSUB -q hpc
#BSUB -W 15
#BSUB -R "rusage[mem=300GB]"
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -o task2_%J.out
#BSUB -e task2_%J.err

# Initializing Python Environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Python script, for the first 15 buildings
python simulate.py 15