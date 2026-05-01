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


@cp.fuse()
def _jacobi_step(u_left, u_right, u_up, u_down, u_int, mask):
    """Fused per-cell update: 4 adds + 1 multiply + 1 where in one kernel."""
    u_new = 0.25 * (u_left + u_right + u_up + u_down)
    return cp.where(mask, u_new, u_int)


def jacobi_gpu(u, interior_mask, max_iter, atol=1e-6, check_every=500):
    """Vectorised Jacobi smoother running entirely on the GPU.

    The hot loop used to issue 6 kernel launches per iteration (3 adds, a
    multiply, a where, and a slice-copy). Profiling showed launch overhead
    dominating - 2.5M launches at ~2 us each ~= 5 s of pure overhead.
    `@cp.fuse` collapses the stencil + where into a single kernel, leaving
    the slice-assign as the only other launch per iteration (2 total).
    """
    u = cp.asarray(u, dtype=cp.float64).copy()
    mask = cp.asarray(interior_mask)
    for it in range(max_iter):
        u_int = u[1:-1, 1:-1]
        u_int_new = _jacobi_step(u[1:-1, :-2], u[1:-1, 2:],
                                 u[:-2, 1:-1], u[2:, 1:-1],
                                 u_int, mask)
        if it % check_every == 0:
            # u_int_new == u_int outside the mask, so the diff there is 0;
            # no need for an extra cp.where to zero non-interior cells.
            delta = cp.abs(u_int_new - u_int).max()
            if float(delta) < atol:
                u[1:-1, 1:-1] = u_int_new
                break
        u[1:-1, 1:-1] = u_int_new
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


STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


def process_one(bid):
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi_gpu(u0, interior_mask, MAX_ITER, ABS_TOL)
    return bid, summary_stats(u, interior_mask)


if __name__ == '__main__':
    LOAD_DIR = "/home/easysort/DTU-HPC-Mini-project/data/"
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

    print("building_id, " + ", ".join(STAT_KEYS), flush=True)
    for bid in building_ids:
        bid, stats = process_one(bid)
        print(
            f"[task10] building_id={bid}  mean_temp={stats['mean_temp']:.6f}  "
            f"std_temp={stats['std_temp']:.6f}  "
            f"pct_above_18={stats['pct_above_18']:.4f}  "
            f"pct_below_15={stats['pct_below_15']:.4f}",
            flush=True,
        )
        print(
            f"{bid}, " + ", ".join(str(stats[k]) for k in STAT_KEYS),
            flush=True,
        )
