from os.path import join
import sys

import numpy as np
import matplotlib.pyplot # For saving the plots (task 3)

import time
t = time.time()          # For measuring time (task 2)

# ------------------ DEFINING FUNCTIONS -----------------------

# Files reader and processor: creation of the IC (u0) and the binary mask
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))        # BC for a certain building ID bid is searched in the HPC directory:
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))      # binary mask for a certain building ID bid is searched in the HPC directory:
    return u, interior_mask

# Jacobi algorithm: for interior_mask nodes with a 1, updates the data in u till a certain tolerance atol is achieved
# @profile # For kernprof profiling (task 4)
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

# -------------------- CODE -------------------------------

if __name__ == '__main__':
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"   # Load data from the course directory in the HPC
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:                                                           # If the number of buildings is provided as the first argument in the batch job
        N = int(sys.argv[1])
    building_ids = building_ids[:N]                                 # Then calcule the T distribution only the first N buildings

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))                                # Tensor for containing in the first subindex the IC
    all_interior_mask = np.empty((N, 512, 512), dtype='bool') # Tensor for containing in the first subindex the interior masks
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)                # For each floor: Matrix containing IC and Matrix containing the binary mask
        all_u0[i] = u0                                              # And insert this matrix in each subindex of the tensor all_u0
        all_interior_mask[i] = interior_mask                        # And insert this matrix in each subindex of the tensor all_interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):   # Each loop u0 takes the values from all_u0[i] and interior_mask from all_interior_mask[i]
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)                       # JACOBI ITERATOR LOOPS TILL THE SOLUTION CONVERGES
        all_u[i] = u                                                           # Save solutions in each subindex of the tensor all_u

    # Print summary statistics in CSV format (and plots of the result)
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)                                # Analyisis function: mean, standard deviation, below 18, below 15
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))          # In that order, print the result for each building

        # FOR EACH OF THE BUILDINGS (ID: bid) SAVING u PLOTS IN A .png (task 3)
        # matplotlib.pyplot.figure()
        # matplotlib.pyplot.imshow(u)
        # matplotlib.pyplot.colorbar()
        # matplotlib.pyplot.savefig(f"{bid}_Final_Temp.png")
        # matplotlib.pyplot.close()

print(time.time()-t)