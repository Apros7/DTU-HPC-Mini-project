# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tqdm"]
# ///
"""Parallel version of simulate.py using static scheduling.

The work is statically partitioned into `num_workers` equal-size
contiguous chunks of floorplans. Each chunk is dispatched to exactly
one worker via `pool.apply_async` before any work starts, so the
assignment is fixed (no work-stealing, no dynamic redistribution).
A small `multiprocessing.Manager().Queue` carries one tick per
completed floorplan back to the main process so we can drive a tqdm
progress bar without affecting the scheduling.

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
from multiprocessing import Manager, Pool
from os.path import join

import numpy as np
from tqdm import tqdm


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
    """Process a single floorplan: load + simulate + summarize."""
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, interior_mask)
    return bid, stats


def process_chunk(args):
    """Worker entry point: process a static chunk of floorplans.

    `progress_q` is a multiprocessing Queue used to send a single
    sentinel per completed floorplan so the main process can drive a
    per-floorplan progress bar without breaking the static scheduling.
    """
    chunk, progress_q = args
    out = []
    for bid in chunk:
        out.append(process_one(bid))
        if progress_q is not None:
            progress_q.put(1)
    return out


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

    # Static scheduling: split building_ids into num_workers equal-size
    # contiguous chunks up front. Each worker is dispatched exactly one
    # chunk via apply_async, so the assignment is fixed before any
    # work starts (no work-stealing, no dynamic redistribution).
    chunksize = max(1, ceil(N / num_workers))
    chunks = [
        building_ids[i:i + chunksize]
        for i in range(0, N, chunksize)
    ]

    # tqdm goes to stderr so it does not pollute the CSV on stdout.
    # We keep the bar enabled even when stderr is a pipe/file, but
    # throttle updates so log files stay readable. Set NO_PROGRESS=1
    # to disable entirely.
    show_progress = os.environ.get("NO_PROGRESS", "") == ""

    t0 = time.perf_counter()
    with Manager() as manager, Pool(processes=num_workers) as pool:
        progress_q = manager.Queue() if show_progress else None
        t_pool_ready = time.perf_counter()
        async_results = [
            pool.apply_async(process_chunk, ((chunk, progress_q),))
            for chunk in chunks
        ]
        # Drain per-floorplan progress ticks from the workers.
        if show_progress:
            with tqdm(total=N, desc=f"P={num_workers}", unit="fp",
                      mininterval=1.0, file=sys.stderr) as pbar:
                done = 0
                while done < N:
                    progress_q.get()
                    done += 1
                    pbar.update(1)
        results = [r for ar in async_results for r in ar.get()]
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
