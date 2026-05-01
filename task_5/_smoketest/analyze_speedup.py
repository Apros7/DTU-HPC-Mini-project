# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Analyze parallel speed-up results and fit Amdahl's law.

Reads results/timings.csv (columns: workers, total_time, compute_time;
optional `repeat` column is averaged automatically) and produces:
  - results/speedup.csv  - per-worker mean times, speed-up, efficiency
  - results/amdahl.txt   - estimated parallel fraction & predictions
  - results/speedup.png  - speed-up plot with Amdahl's law fit

This script declares its dependencies inline (PEP 723) so it can be
executed anywhere with uv - no HPC, no manual venv:

    uv run analyze_speedup.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


HERE = Path(__file__).parent
TIMINGS = HERE / "results" / "timings.csv"
OUT_CSV = HERE / "results" / "speedup.csv"
OUT_TXT = HERE / "results" / "amdahl.txt"
OUT_PNG = HERE / "results" / "speedup.png"

TOTAL_FLOORPLANS = 4571  # full dataset size
N_TIMED = 100            # floorplans used per timing run


def amdahl(p, f):
    """Amdahl's law speed-up given parallel fraction f and p workers."""
    return 1.0 / ((1.0 - f) + f / p)


def main():
    df = pd.read_csv(TIMINGS).sort_values("workers").reset_index(drop=True)
    # If multiple repeats are present, average them; otherwise use as-is.
    agg = (
        df.groupby("workers", as_index=False)
          .agg(total_mean=("total_time", "mean"),
               compute_mean=("compute_time", "mean"))
          .sort_values("workers")
    )

    t1 = float(agg.loc[agg.workers == agg.workers.min(), "compute_mean"].iloc[0])
    agg["speedup"] = t1 / agg["compute_mean"]
    agg["efficiency"] = agg["speedup"] / agg["workers"]
    agg.to_csv(OUT_CSV, index=False)

    workers = agg["workers"].to_numpy(dtype=float)
    speedups = agg["speedup"].to_numpy(dtype=float)

    # Fit Amdahl's law to estimate parallel fraction f.
    popt, _ = curve_fit(amdahl, workers, speedups,
                        p0=[0.95], bounds=(0.0, 1.0))
    f_hat = float(popt[0])
    s_max = 1.0 / (1.0 - f_hat) if f_hat < 1.0 else float("inf")

    # Best measured speed-up and the worker count that achieved it.
    best_idx = int(np.argmax(speedups))
    best_workers = int(workers[best_idx])
    best_speedup = float(speedups[best_idx])
    best_compute = float(agg["compute_mean"].iloc[best_idx])

    # Estimated wall-clock to process all floorplans with the fastest
    # parallel solution, scaling linearly from N_TIMED -> TOTAL_FLOORPLANS.
    est_total_seconds = best_compute * (TOTAL_FLOORPLANS / N_TIMED)

    lines = []
    lines.append("Amdahl's-law analysis of parallel simulator")
    lines.append("===========================================")
    lines.append(f"Floorplans timed per run (N): {N_TIMED}")
    lines.append(f"Repeats per worker count:     {int(df.groupby('workers').size().min())}")
    lines.append("")
    lines.append("Per-worker results (compute_time):")
    for _, row in agg.iterrows():
        lines.append(
            f"  P={int(row.workers):2d}: t={row.compute_mean:7.2f}s  "
            f"speedup={row.speedup:5.2f}  efficiency={row.efficiency:5.2f}"
        )
    lines.append("")
    lines.append(f"Estimated parallel fraction f  = {f_hat:.4f}  "
                 f"({f_hat*100:.2f}% of work is parallelisable)")
    lines.append(f"Theoretical max speed-up S_inf = 1/(1-f) = {s_max:.2f}x")
    lines.append(f"Best measured speed-up         = {best_speedup:.2f}x "
                 f"with P={best_workers} workers")
    lines.append(f"Fraction of S_inf achieved     = "
                 f"{best_speedup / s_max * 100:.1f}%")
    lines.append("")
    lines.append(
        f"Estimated time for all {TOTAL_FLOORPLANS} floorplans with "
        f"P={best_workers}: {est_total_seconds:.1f} s "
        f"({est_total_seconds/60:.1f} min)"
    )

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    p_dense = np.linspace(workers.min(), workers.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(workers, speedups, "o-", label="Measured speed-up", color="C0")
    ax.plot(p_dense, amdahl(p_dense, f_hat), "--",
            label=f"Amdahl fit, f={f_hat:.3f}", color="C1")
    ax.plot(p_dense, p_dense, ":", color="gray", label="Ideal (linear)")
    ax.axhline(s_max, color="C3", linestyle=":",
               label=f"S_inf = {s_max:.2f}x")
    ax.set_xlabel("Number of workers P")
    ax.set_ylabel("Speed-up  S(P) = T(1) / T(P)")
    ax.set_title(f"Static-scheduling speed-up (N={N_TIMED} floorplans)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
