from os.path import join
import sys

import numpy as np
import cupy as cp


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi_gpu(u, interior_mask, max_iter, atol=1e-6, check_every=100):
    """Vectorised Jacobi smoother running entirely on the GPU.

    `delta` is computed every step but only copied to host every
    `check_every` iterations to avoid a per-iteration sync.
    """
    u = cp.asarray(u, dtype=cp.float64).copy()
    mask = cp.asarray(interior_mask)
    for it in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:]
                        + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[mask]
        delta = cp.abs(u[1:-1, 1:-1][mask] - u_new_interior).max()
        u[1:-1, 1:-1][mask] = u_new_interior
        if it % check_every == 0 and float(delta) < atol:
            break
    return u


def summary_stats(u, interior_mask):
    mask = cp.asarray(interior_mask)
    u_int = u[1:-1, 1:-1][mask]
    return {
        'mean_temp':    float(u_int.mean()),
        'std_temp':     float(u_int.std()),
        'pct_above_18': float((u_int > 18).sum() / u_int.size * 100),
        'pct_below_15': float((u_int < 15).sum() / u_int.size * 100),
    }


def process_one(bid):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi_gpu(u0, interior_mask, MAX_ITER, ABS_TOL)
    return bid, summary_stats(u, interior_mask)


if __name__ == '__main__':
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    # Second arg (worker count) is accepted for CLI compatibility with the
    # CPU scripts but ignored: one host process drives the GPU.
    if len(sys.argv) > 2:
        _ = int(sys.argv[2])

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    STAT_KEYS = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    print('building_id, ' + ', '.join(STAT_KEYS))
    for bid in building_ids:
        bid, stats = process_one(bid)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in STAT_KEYS))
