#!/usr/bin/env bash
# Local (non-HPC) version of run_speedup.sh.
#
# Runs the static-scheduling parallel timing experiment on whichever
# machine you're on (Mac, laptop, etc.). Uses `uv run` so you don't
# need to manage a Python environment manually - uv builds an
# ephemeral venv from the PEP 723 metadata in simulate_parallel.py.
#
# Requirements:
#   - uv installed         (https://docs.astral.sh/uv/)
#   - Floorplan data       Set LOAD_DIR to a directory containing the
#                          {bid}_domain.npy / {bid}_interior.npy files
#                          and a building_ids.txt index file.
#
# Configuration via env vars:
#   N         number of floorplans to time (default: 100, max: 100)
#   WORKERS   space-separated worker counts to test
#             (default chosen to match local core count)
#   LOAD_DIR  path to the floorplan data directory
#
# Example:
#   LOAD_DIR=./data N=50 WORKERS="1 2 4 8" ./run_speedup_local.sh

set -eo pipefail

cd "$(dirname "$0")"

N=${N:-100}

# Pick a sensible default WORKERS list based on local core count if the
# user didn't override it.
if [[ -z "${WORKERS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        NCORES=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        NCORES=$(sysctl -n hw.ncpu)
    else
        NCORES=8
    fi
    # Build a 1,2,4,...,NCORES list, then add NCORES if not a power of 2.
    WORKERS_LIST=()
    p=1
    while (( p < NCORES )); do
        WORKERS_LIST+=("$p")
        p=$(( p * 2 ))
    done
    WORKERS_LIST+=("$NCORES")
    WORKERS="${WORKERS_LIST[*]}"
fi
# shellcheck disable=SC2206
WORKERS_ARR=($WORKERS)

if [[ -z "${LOAD_DIR:-}" ]]; then
    if [[ -d /dtu/projects/02613_2025/data/modified_swiss_dwellings ]]; then
        LOAD_DIR=/dtu/projects/02613_2025/data/modified_swiss_dwellings/
    else
        echo "ERROR: LOAD_DIR not set and the default DTU HPC path is not"
        echo "       available. Point LOAD_DIR at your local copy of the"
        echo "       floorplan data, e.g.:"
        echo "         LOAD_DIR=./data $0"
        exit 1
    fi
fi
export LOAD_DIR

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found in PATH. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

mkdir -p results out

# Pin numpy/BLAS to a single thread per process so worker count is the
# only source of parallelism we measure.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CSV=results/timings.csv
echo "workers,total_time,compute_time" > "$CSV"

echo "================================================================"
echo "Parallel speed-up experiment (LOCAL)"
echo "  N (floorplans)      : $N"
echo "  Worker counts       : ${WORKERS_ARR[*]}"
echo "  LOAD_DIR            : $LOAD_DIR"
echo "  Host                : $(hostname)"
echo "  Started             : $(date)"
echo "================================================================"

JOB_START=$SECONDS
for w in "${WORKERS_ARR[@]}"; do
    STEP_START=$SECONDS
    echo ""
    echo "[$(date +%H:%M:%S)] >>> workers=$w  (chunksize ~ $(( (N + w - 1) / w )))"
    OUT="out/run_w${w}.csv"
    ERR="out/run_w${w}.err"
    # tqdm progress bar goes to stderr; tee duplicates it to the
    # terminal (>&2) AND to the per-run .err file we grep below.
    uv run --quiet simulate_parallel.py "$N" "$w" > "$OUT" \
        2> >(tee "$ERR" >&2)

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

echo ""
echo "Now run the analysis:"
echo "    uv run analyze_speedup.py"
