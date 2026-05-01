# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Render results/speedup_all.png from results/timings_all.csv only.

Same plot that ``run_all_tasks.py`` produces at the end of a benchmark
sweep, but without rerunning the benchmarks - useful for tweaking the
figure (labels, axis ticks, baseline) against an existing CSV.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
TIMINGS = ROOT / "results" / "timings_all.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Tasks for which only P=1 was measured (GPU implementations); these are
# drawn as horizontal dashed lines across the CPU worker range.
SINGLE = {"task8", "task9", "task10", "task11-8"}
WORKERS_CPU = [1, 2, 4, 8, 16]


def amdahl(p, f):
    """Amdahl's law speed-up: parallel fraction `f`, `p` workers."""
    return 1.0 / ((1.0 - f) + f / p)


# Load: task -> sorted [(P, T), ...]
rows = []
with TIMINGS.open() as f:
    for r in csv.DictReader(f):
        rows.append((r["task"], int(r["workers"]), float(r["total_time"])))

tasks = sorted({n for n, _, _ in rows})
N = "?"  # for the title; not stored in the CSV, leave unknown

# Speed-up baseline: the slowest T(P=1) in the file. Falls back to first P=1.
ref_match = next(((n, t) for n, p, t in rows if n == "task5" and p == 1), None)
if ref_match is None:
    ref_match = next((n, t) for n, p, t in rows if p == 1)
ref_name, T_ref = ref_match

fig, ax = plt.subplots(figsize=(7, 5))
for idx, name in enumerate(tasks):
    pts = sorted([(p, t) for n, p, t in rows if n == name])
    P = np.array([p for p, _ in pts], float)
    T = np.array([t for _, t in pts], float)
    S_abs = T_ref / T
    c = f"C{idx}"
    if name in SINGLE or len(P) == 1:
        ax.axhline(float(S_abs[0]), color=c,
                   label=f"{name} ({T[0]:.2f}s, {float(S_abs[0]):.1f}x)")
    else:
        ax.plot(P, S_abs, "o-", color=c,
                label=f"{name} (T1={T[0]:.1f}s)")
        if len(P) >= 2:
            S_self = T[0] / T
            (f_hat,), _ = curve_fit(amdahl, P, S_self,
                                    p0=[0.9], bounds=(0, 1))
            pp = np.linspace(P.min(), P.max(), 200)
            ax.plot(pp, S_abs[0] * amdahl(pp, f_hat), "--", color=c,
                    label=f"{name} Amdahl f={f_hat:.3f}")

ax.plot([1, max(WORKERS_CPU)], [1, max(WORKERS_CPU)], ":", color="gray",
        label=f"ideal vs {ref_name} P=1")
ax.set_xlabel("workers P")
ax.set_ylabel(f"speed-up vs {ref_name} P=1")
ax.set_title(f"Speed-up (from {TIMINGS.name})")
ax.set_xticks(WORKERS_CPU)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "speedup_all.png", dpi=150)
plt.close(fig)
print("saved", OUT / "speedup_all.png")
