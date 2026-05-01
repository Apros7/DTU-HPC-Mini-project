from os.path import join
import math
import sys

import numpy as np
from numba import cuda


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_step(u, u_new, mask):
    """One Jacobi sweep: write u_new from u.

    The mask is shape (H, W); u/u_new are shape (H+2, W+2) with a one-cell
    halo. Threads index the interior coordinate system (i, j) -> u[i+1, j+1].
    Non-interior cells are copied through so the buffer swap is safe.
    """
    i, j = cuda.grid(2)
    h, w = mask.shape
    if i < h and j < w:
        if mask[i, j]:
            u_new[i + 1, j + 1] = 0.25 * (
                u[i + 1, j] + u[i + 1, j + 2]
                + u[i, j + 1] + u[i + 2, j + 1]
            )
        else:
            u_new[i + 1, j + 1] = u[i + 1, j + 1]


def jacobi_cuda(u, interior_mask, max_iter, atol=1e-6, check_every=200):
    """Run Jacobi iterations on the GPU with periodic convergence checks."""
    d_u = cuda.to_device(u)
    d_u_new = cuda.to_device(u.copy())
    d_mask = cuda.to_device(interior_mask)

    tpb = (16, 16)
    bpg = (
        math.ceil(interior_mask.shape[0] / tpb[0]),
        math.ceil(interior_mask.shape[1] / tpb[1]),
    )

    for it in range(max_iter):
        jacobi_step[bpg, tpb](d_u, d_u_new, d_mask)
        d_u, d_u_new = d_u_new, d_u
        if (it + 1) % check_every == 0:
            # Pull both buffers and check max diff. ~2 MB transfer per check.
            delta = float(np.abs(d_u.copy_to_host()
                                 - d_u_new.copy_to_host()).max())
            if delta < atol:
                break

    return d_u.copy_to_host()


def summary_stats(u, interior_mask):
    u_int = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp':    u_int.mean(),
        'std_temp':     u_int.std(),
        'pct_above_18': (u_int > 18).sum() / u_int.size * 100,
        'pct_below_15': (u_int < 15).sum() / u_int.size * 100,
    }


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

    print('building_id, ' + ', '.join(STAT_KEYS))
    for bid in building_ids:
        u0, mask = load_data(LOAD_DIR, bid)
        u = jacobi_cuda(u0, mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u, mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in STAT_KEYS))
