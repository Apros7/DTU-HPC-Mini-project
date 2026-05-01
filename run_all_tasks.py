# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy"]
# ///
"""Benchmark CPU tasks (5/6/7) over [1,2,4,8] workers and GPU tasks
(8/9/10/11-8) once each at P=1 on 64 floorplans, then plot speed-up
relative to task5 P=1."""
import csv, subprocess, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
TASKS = {
    "task6":    ROOT / "simulate_task6.py",
    "task7":    ROOT / "simulate_task7.py",
    "task8":    ROOT / "simulate_task8.py",
    "task9":    ROOT / "simulate_task9.py",
    "task10":   ROOT / "simulate_task10.py",
    "task11-8": ROOT / "simulate_task11-8.py",
}
SINGLE = {"task8", "task9", "task10", "task11-8"}  # ignore P, run once
WORKERS_CPU = [1, 2, 4, 8, 16]
N = 64
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def Ps(name):
    return [1] if name in SINGLE else WORKERS_CPU


plan = [(name, p) for name in TASKS for p in Ps(name)]
total = len(plan)

rows = []
bench_start = time.perf_counter()
for i, (name, p) in enumerate(plan, 1):
    tag = f"[{i:2d}/{total}] {name} P={p:2d}"
    print(f"{tag} ... starting", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.Popen(["uv", "run", str(TASKS[name]), str(N), str(p)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT)
    while proc.poll() is None:
        print(f"\r{tag} ... running {time.perf_counter()-t0:6.1f}s",
              end="", flush=True)
        time.sleep(1)
    t = time.perf_counter() - t0
    if proc.returncode:
        print(); raise SystemExit(f"{tag} FAILED (exit {proc.returncode})")
    rows.append((name, p, t))
    eta = (time.perf_counter() - bench_start) / i * (total - i)
    print(f"\r{tag} done in {t:7.2f}s  | ETA {eta/60:5.1f} min",
          flush=True)

with (OUT / "timings_all.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(("task", "workers", "total_time"))
    w.writerows(rows)

amdahl = lambda p, f: 1 / ((1 - f) + f / p)

# Speed-up is reported against a single common baseline (task5 P=1) so GPU
# and CPU points share an axis. Falls back to the first available P=1.
ref_match = next(((n, t) for n, p, t in rows if n == "task5" and p == 1),
                 None) or next((n, t) for n, p, t in rows if p == 1)
ref_name, T_ref = ref_match

fig, ax = plt.subplots(figsize=(7, 5))
for idx, name in enumerate(TASKS):
    pts = sorted([(p, t) for n, p, t in rows if n == name])
    P = np.array([p for p, _ in pts], float)
    T = np.array([t for _, t in pts], float)
    S_abs = T_ref / T
    c = f"C{idx}"
    if name in SINGLE:
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
ax.set_title(f"Speed-up on N={N} floorplans")
ax.set_xticks(WORKERS_CPU)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "speedup_all.png", dpi=150)
print("saved", OUT / "speedup_all.png")
