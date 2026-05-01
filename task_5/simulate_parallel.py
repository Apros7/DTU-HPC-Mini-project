# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Parallel version of simulate.py using static scheduling.

Each worker is assigned an equal-sized contiguous chunk of floorplans
(static scheduling) by using `multiprocessing.Pool.map` with
`chunksize = ceil(N / num_workers)`. With this chunksize, the pool
hands every worker exactly one chunk, so the work distribution is
fixed up front and does not depend on runtime per-task durations.

Usage:
    python simulate_parallel.py <N> <num_workers>          # HPC / conda env
    uv run simulate_parallel.py <N> <num_workers>          # anywhere (local)

Data location:
    The floorplan files are read from $LOAD_DIR if set, otherwise from
    the default DTU HPC path:
        /dtu/projects/02613_2025/data/modified_swiss_dwellings/

    To run locally, point LOAD_DIR at a directory containing the same
    files (e.g. an rsync'd subset):
        LOAD_DIR=./data uv run simulate_parallel.py 20 4

Prints a single line to stderr with timing info:
    TIMING,N=...,workers=...,total_time=...,compute_time=...

The CSV results are printed to stdout (same format as the reference).
"""

import os
import sys
import time
from math import ceil
from multiprocessing import Pool
from os.path import join

import numpy as np


LOAD_DIR = os.environ.get(
    'LOAD_DIR',
    '/dtu/projects/02613_2025/data/modified_swiss_dwellings/',
)
MAX_ITER = 20_000
ABS_TOL = 1e-4


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


def process_one(bid):
    """Worker function: load, simulate, summarize a single floorplan."""
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, interior_mask)
    return bid, stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python simulate_parallel.py <N> <num_workers>",
              file=sys.stderr)
        sys.exit(1)

    N = int(sys.argv[1])
    num_workers = int(sys.argv[2])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    building_ids = building_ids[:N]

    # Static scheduling: give every worker a single, equal-size chunk.
    # With Pool.map(chunksize=ceil(N/P)), tasks are partitioned ahead of
    # time and each worker processes the same number of floorplans.
    chunksize = max(1, ceil(N / num_workers))

    t0 = time.perf_counter()
    with Pool(processes=num_workers) as pool:
        t_pool_ready = time.perf_counter()
        results = pool.map(process_one, building_ids, chunksize=chunksize)
    t1 = time.perf_counter()

    total_time = t1 - t0
    compute_time = t1 - t_pool_ready

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ", ".join(str(stats[k]) for k in stat_keys))

    print(
        f"TIMING,N={N},workers={num_workers},chunksize={chunksize},"
        f"total_time={total_time:.4f},compute_time={compute_time:.4f}",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
