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


def load_batch(load_dir, building_ids):
    """Stack all buildings into (B, H, W) and (B, h, w) host arrays."""
    us, masks = [], []
    for bid in building_ids:
        u, m = load_data(load_dir, bid)
        us.append(u)
        masks.append(m)
    return np.stack(us), np.stack(masks)


def jacobi_gpu_batched(us, masks, max_iter, atol=1e-6, check_every=500):
    """Batched Jacobi: one kernel launch per step updates all B buildings.

    Cuts launch overhead by ~B compared to per-building loops, since the
    same stencil is fused across the leading batch dim. `cp.where` avoids
    boolean-mask indexing (data-dependent output size -> host sync).
    """
    us = cp.asarray(us, dtype=cp.float64).copy()
    masks = cp.asarray(masks)
    for it in range(max_iter):
        u_new = 0.25 * (us[:, 1:-1, :-2] + us[:, 1:-1, 2:]
                        + us[:, :-2, 1:-1] + us[:, 2:, 1:-1])
        u_int = us[:, 1:-1, 1:-1]
        us[:, 1:-1, 1:-1] = cp.where(masks, u_new, u_int)
        if it % check_every == 0:
            delta = cp.abs(cp.where(masks, u_int - u_new, 0.0)).max()
            if float(delta) < atol:
                break
    return us


def summary_stats_batched(us, masks):
    """Per-building stats computed entirely on the GPU.

    Returns a host (B, 4) array: mean, std, pct>18, pct<15.
    Uses arithmetic on the boolean mask instead of boolean indexing so
    nothing in here triggers a per-slice host sync.
    """
    u_int = us[:, 1:-1, 1:-1]
    m = masks.astype(cp.float64)
    n = m.sum(axis=(1, 2))
    mean = (u_int * m).sum(axis=(1, 2)) / n
    var = (((u_int - mean[:, None, None]) ** 2) * m).sum(axis=(1, 2)) / n
    std = cp.sqrt(var)
    above = ((u_int > 18) & masks).sum(axis=(1, 2)) / n * 100
    below = ((u_int < 15) & masks).sum(axis=(1, 2)) / n * 100
    return cp.asnumpy(cp.stack([mean, std, above, below], axis=1))


if __name__ == '__main__':
    LOAD_DIR = "/home/easysort/DTU-HPC-Mini-project/data/"
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]
    # Second arg accepted for CLI compatibility with run_all_tasks.py but
    # ignored: one host process drives the GPU.
    if len(sys.argv) > 2:
        _ = int(sys.argv[2])

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    STAT_KEYS = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    # Memory: 2 * B * H * W * 8 bytes; for B=64, H=W=514 -> ~270 MB.
    # If B grows past a few hundred, chunk this loop.
    us_h, masks_h = load_batch(LOAD_DIR, building_ids)
    us = jacobi_gpu_batched(us_h, masks_h, MAX_ITER, ABS_TOL)
    stats = summary_stats_batched(us, cp.asarray(masks_h))

    print('building_id, ' + ', '.join(STAT_KEYS))
    for bid, row in zip(building_ids, stats):
        print(f"{bid}, " + ", ".join(str(v) for v in row))
