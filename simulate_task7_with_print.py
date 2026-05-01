from os.path import join
from multiprocessing.pool import Pool
import sys

import multiprocessing as mp
mp.set_start_method("fork")

import numpy as np
import matplotlib.pyplot # For saving the plots (task 3)

from numba import njit

# ------------------ DEFINING FUNCTIONS -----------------------

# Files reader and processor: creation of the IC (u0) and the binary mask
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))        # BC for a certain building ID bid is searched in the HPC directory:
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))      # binary mask for a certain building ID bid is searched in the HPC directory:
    return u, interior_mask

# Jacobi algorithm: for interior_mask nodes with a 1, updates the data in u till a certain tolerance atol is achieved
def jacobi(u, interior_mask, max_iter, atol=1e-6):
   u = np.copy(u)
   for i in range(max_iter):                                                    # Till the maximum number of iterations is achieved
       # Compute average of left, right, up and down neighbors, see eq. (1)
       u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])  # Update calculation for all grid points
       u_new_interior = u_new[interior_mask]                                     # Saves only the points belonging to the interior grid points (where interior_mask take values 1)
       delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()       # Max-norm update in interior points calculation
       u[1:-1, 1:-1][interior_mask] = u_new_interior                             # Update values of u in the interior grid point (although u contains values for all grid points)

       if delta < atol:                                                          # Or till a certain tolerance is achieved
           break
   return u

@njit
def jacobi_jit(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    n,m = u.shape

    for it in range(max_iter):
        delta = 0.0
        u_new = np.copy(u)
        
        for i in range(1, n-1):
            for j in range(1, m-1):
                if interior_mask[i-1,j-1]:
                    new_val = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1,j] + u[i+1,j])
                    diff = abs(u[i,j] - new_val)
                    
                    if diff > delta:
                        delta = diff
                    
                    u_new[i, j] = new_val
        u = u_new
        if delta < atol: break
    return u

# STATS indicators: mean, standard deviation, under 18, under 15
def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

# Function for parallelizing the code
    # Left-hand side indexes elimination: not need of 3-order tensors use due to different memory in each process
STAT_KEYS = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]


def f_for_multiprocessing(bid):  # Each floorplan in a worker process
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi_jit(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, interior_mask)
    # Live line (mean temperature easy to spot while workers finish)
    print(
        f"[task7] building_id={bid}  mean_temp={stats['mean_temp']:.6f}  "
        f"std_temp={stats['std_temp']:.6f}  "
        f"pct_above_18={stats['pct_above_18']:.4f}  "
        f"pct_below_15={stats['pct_below_15']:.4f}",
        flush=True,
    )
    # Same CSV row as before (one header printed once in __main__)
    print(f"{bid}, " + ", ".join(str(stats[k]) for k in STAT_KEYS), flush=True)


# -------------------- CODE -------------------------------

if __name__ == '__main__':
    LOAD_DIR = "/home/easysort/DTU-HPC-Mini-project/data/"   # Load data from the course directory in the HPC
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:                                                           # If the number of buildings is provided as the first argument in the batch job
        N = int(sys.argv[1])
    building_ids = building_ids[:N]                                 # Then calcule the T distribution only the first N buildings

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # ----------------------------- MULTIPROCESSING ---------------------------
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print("building_id, " + ", ".join(STAT_KEYS), flush=True)
    with Pool(workers) as pool:
        for _ in pool.imap(f_for_multiprocessing, building_ids):
            pass