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


# Custom elementwise kernel: writes the new interior value directly into a
# pre-allocated `out` buffer instead of allocating a temp + slice-copying it
# back, which was ~49 % of runtime in the previous fused-step version.
# Using `T` as the dtype param so we can run the same kernel on float32
# (faster) or float64 (reference) without recompiling logic.
_jacobi_step_kernel = cp.ElementwiseKernel(
    'T u_left, T u_right, T u_up, T u_down, T u_int, bool mask',
    'T u_int_new',
    'u_int_new = mask ? T(0.25) * (u_left + u_right + u_up + u_down) : u_int',
    'jacobi_step_batched',
)


def jacobi_gpu_batched(us, masks, max_iter, atol=1e-4, check_every=500,
                      dtype=cp.float32):
    """Batched Jacobi with ping-pong buffers + in-place stencil kernel.

    Optimisations vs. the previous fused-step version:
    - **No slice copy.** The custom `ElementwiseKernel` writes directly
      into the *other* buffer's interior view, so the per-iter
      `us[:, 1:-1, 1:-1] = u_int_new` (~49 % of GPU time) is gone.
    - **Float32 state.** Halves HBM traffic per iter; tolerances and
      values (room temperatures, atol >= 1e-4) sit comfortably inside
      float32 precision.
    - **Boundaries copied once.** Both buffers carry the BCs in their
      outer ring; the kernel only ever touches the interior, so we can
      swap buffers without rewriting the boundary every step.
    """
    a = cp.ascontiguousarray(cp.asarray(us, dtype=dtype))
    b = a.copy()  # holds same boundaries; interior will be overwritten
    masks = cp.asarray(masks)
    for it in range(max_iter):
        u_int_old = a[:, 1:-1, 1:-1]
        u_int_new = b[:, 1:-1, 1:-1]
        _jacobi_step_kernel(
            a[:, 1:-1, :-2], a[:, 1:-1, 2:],
            a[:, :-2, 1:-1], a[:, 2:, 1:-1],
            u_int_old, masks,
            u_int_new,
        )
        if it % check_every == 0:
            delta = cp.abs(u_int_new - u_int_old).max()
            if float(delta) < atol:
                a, b = b, a
                break
        a, b = b, a
    return a


def summary_stats_batched(us, masks):
    """Per-building stats computed entirely on the GPU.

    Returns a host (B, 4) array: mean, std, pct>18, pct<15.
    Uses arithmetic on the boolean mask instead of boolean indexing so
    nothing in here triggers a per-slice host sync. Computes in float64
    even when the solver runs in float32, so the aggregated CSV values
    match the reference solver to ~6 decimals.
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
    # LOAD_DIR = "/home/easysort/DTU-HPC-Mini-project/modified-swiss-dwellings/"
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("Total length of building_ids: ", len(building_ids))
    building_ids = building_ids[:N]
    # Second arg accepted for CLI compatibility with run_all_tasks.py but
    # ignored: one host process drives the GPU.
    if len(sys.argv) > 2:
        _ = int(sys.argv[2])

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    STAT_KEYS = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    # Memory: 2 * B * H * W * 4 bytes (float32 state); for B=64, H=W=514
    # -> ~135 MB. If B grows past ~1000, switch to the chunked variant in
    # simulate_task11-8_run_all.py to bound the working set.
    us_h, masks_h = load_batch(LOAD_DIR, building_ids)
    us = jacobi_gpu_batched(us_h, masks_h, MAX_ITER, ABS_TOL)
    stats = summary_stats_batched(us, cp.asarray(masks_h))

    out_path = Path(__file__).parent / "results" / "task11-8_stats.csv"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["building_id", *STAT_KEYS])
        for bid, row in zip(building_ids, stats):
            w.writerow([bid, *row])

    print('building_id, ' + ', '.join(STAT_KEYS))
    for bid, row in zip(building_ids, stats):
        print(f"{bid}, " + ", ".join(str(v) for v in row))
    print(f"saved {len(building_ids)} rows to {out_path}")
