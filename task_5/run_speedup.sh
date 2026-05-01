#!/bin/bash
#BSUB -J wallheat_par
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -o logs/wallheat_par_%J.out
#BSUB -e logs/wallheat_par_%J.err

# Static-scheduling parallel timing experiment.
# Runs the parallel simulator for a fixed N (<= 100) with several
# worker counts and records timings to results/timings.csv.

set -eo pipefail

N=${N:-100}
WORKERS=(1 2 4 8 16 24 32)

mkdir -p results logs out

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

set -u

# Pin numpy/BLAS to a single thread per process so that worker count
# is the only source of parallelism we measure.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CSV=results/timings.csv
echo "workers,total_time,compute_time" > "$CSV"

echo "================================================================"
echo "Parallel speed-up experiment"
echo "  N (floorplans)      : $N"
echo "  Worker counts       : ${WORKERS[*]}"
echo "  Repeats per setting : 1"
echo "  Host                : $(hostname)"
echo "  Started             : $(date)"
echo "================================================================"

JOB_START=$SECONDS

for w in "${WORKERS[@]}"; do
    STEP_START=$SECONDS
    echo ""
    echo "[$(date +%H:%M:%S)] >>> workers=$w  (chunksize ~ $(( (N + w - 1) / w )))"
    OUT="out/run_w${w}.csv"
    ERR="out/run_w${w}.err"
    python -u simulate_parallel.py "$N" "$w" > "$OUT" 2> "$ERR"

    line=$(grep '^TIMING' "$ERR")
    total=$(echo "$line" | sed -n 's/.*total_time=\([0-9.]*\).*/\1/p')
    compute=$(echo "$line" | sed -n 's/.*compute_time=\([0-9.]*\).*/\1/p')
    echo "${w},${total},${compute}" >> "$CSV"

    STEP_DURATION=$(( SECONDS - STEP_START ))
    TOTAL_DURATION=$(( SECONDS - JOB_START ))
    echo "[$(date +%H:%M:%S)] <<< workers=$w  total=${total}s  compute=${compute}s" \
         "(step ${STEP_DURATION}s, elapsed ${TOTAL_DURATION}s)"
done

echo ""
echo "================================================================"
echo "Done at $(date). Total elapsed: $(( SECONDS - JOB_START )) s"
echo "Results in $CSV"
echo "================================================================"
cat "$CSV"
