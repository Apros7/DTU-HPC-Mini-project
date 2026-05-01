from os.path import join
from pathlib import Path
import csv
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


_jacobi_step_kernel = cp.ElementwiseKernel(
    'T u_left, T u_right, T u_up, T u_down, T u_int, bool mask',
    'T u_int_new',
    'u_int_new = mask ? T(0.25) * (u_left + u_right + u_up + u_down) : u_int',
    'jacobi_step_batched',
)


def jacobi_gpu_batched(us, masks, max_iter, atol=1e-4, check_every=500,
                      dtype=cp.float32):
    """Batched Jacobi: ping-pong buffers + custom in-place stencil kernel + float32."""
    a = cp.ascontiguousarray(cp.asarray(us, dtype=dtype))
    b = a.copy()
    masks = cp.asarray(masks)
    for it in range(max_iter):
        u_int_old = a[:, 1:-1, 1:-1]
        u_int_new = b[:, 1:-1, 1:-1]
        _jacobi_step_kernel(a[:, 1:-1, :-2], a[:, 1:-1, 2:],
                            a[:, :-2, 1:-1], a[:, 2:, 1:-1],
                            u_int_old, masks, u_int_new)
        if it % check_every == 0:
            delta = cp.abs(u_int_new - u_int_old).max()
            if float(delta) < atol:
                a, b = b, a
                break
        a, b = b, a
    return a


def summary_stats_batched(us, masks):
    """Per-building stats computed entirely on the GPU.

    Returns a host (B, 4) array: mean, std, pct>18, pct<15. Promotes the
    interior to float64 so aggregates match the float64 reference even
    when the solver runs in float32.
    """
    u_int = us[:, 1:-1, 1:-1].astype(cp.float64, copy=False)
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
    print("Total length of building_ids: ", len(building_ids))
    building_ids = building_ids[:N]
    print("Building IDs length: ", len(building_ids))
    # Second arg accepted for CLI compatibility with run_all_tasks.py but
    # ignored: one host process drives the GPU.
    if len(sys.argv) > 2:
        _ = int(sys.argv[2])

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    STAT_KEYS = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    # Process buildings in chunks so the working set on the GPU stays
    # bounded. With float32 state, each chunk uses ~ BATCH * 514^2 * 4
    # bytes per buffer (`a`, `b`), so BATCH=256 -> ~270 MB per buffer
    # (~540 MB combined), comfortably below a 32 GB GPU. Could be raised
    # to 512-1024 if you want fewer chunks.
    BATCH = 256
    pool = cp.get_default_memory_pool()
    chunks = [building_ids[i:i + BATCH]
              for i in range(0, len(building_ids), BATCH)]

    all_stats = []
    for c, chunk in enumerate(chunks, 1):
        print(f"  chunk {c}/{len(chunks)}: {len(chunk)} buildings", flush=True)
        us_h, masks_h = load_batch(LOAD_DIR, chunk)
        us = jacobi_gpu_batched(us_h, masks_h, MAX_ITER, ABS_TOL)
        stats = summary_stats_batched(us, cp.asarray(masks_h))
        all_stats.append(stats)
        # Release device buffers so the pool doesn't grow unboundedly.
        del us
        pool.free_all_blocks()
    stats = np.concatenate(all_stats, axis=0)

    out_path = Path(__file__).parent / "results" / "task11-8_stats.csv"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["building_id", *STAT_KEYS])
        for bid, row in zip(building_ids, stats):
            w.writerow([bid, *row])
    print(f"saved {len(building_ids)} rows to {out_path}")
