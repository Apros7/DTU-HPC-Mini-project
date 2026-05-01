# TASK 1 ASSIGNMENT 02613
import matplotlib.pyplot
import numpy as np

# ------------------------ PLOT INITIAL CONDITIONS ---------------------------
# List containing the files for the Binary masks with the interior points
files = ['470_domain.npy','730_domain.npy','1980_domain.npy','2571_domain.npy']
subplot1, subplot2 = matplotlib.pyplot.subplots(1, 4, figsize=(9, 9))

n=0
for f in files:
    input_data = np.load(f)
    s=subplot2[n].imshow(input_data)
    subplot2[n].set_title(f"ID: {f.split('_')[0]}")
    n+=1
subplot1.colorbar(s, ax=subplot2,shrink=0.4)

# ----------------- PLOT BINARY MASKS WITH INTERIOR POINTS --------------------
files = ['470_interior.npy','730_interior.npy','1980_interior.npy','2571_interior.npy']
subplot3, subplot4 = matplotlib.pyplot.subplots(1, 4)
n=0
for f in files:
    input_data = np.load(f)
    subplot4[n].imshow(input_data, cmap='gray')
    subplot4[n].set_title(f"ID: {f.split('_')[0]}")
    n+=1
matplotlib.pyplot.show()