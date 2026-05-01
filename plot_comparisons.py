# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///
"""Generate the per-pair / per-triple comparison plots from timings_all.csv.

Speed-up is computed against a common baseline per plot: the slowest T(1)
of the tasks in that comparison. That way GPU tasks (which only have a
single P=1 measurement) and CPU tasks (which sweep workers) sit on the
same axis - the slowest task passes through S=1 at P=1, faster tasks sit
above. Single-point (GPU) tasks are drawn as a horizontal dashed line
across the worker range of the comparison so they're visually comparable.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def amdahl(p, f):
    """Amdahl's law speed-up: parallel fraction `f`, `p` workers."""
    return 1.0 / ((1.0 - f) + f / p)

HERE = Path(__file__).resolve().parent
TIMINGS = HERE / "results" / "timings_all.csv"
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# task -> sorted [(P, T), ...]
data: dict[str, list[tuple[int, float]]] = {}
with TIMINGS.open() as f:
    for r in csv.DictReader(f):
        data.setdefault(r["task"], []).append(
            (int(r["workers"]), float(r["total_time"]))
        )
for k in data:
    data[k].sort()


def compare(names, fname, title):
    base = max(dict(data[n])[1] for n in names)  # slowest T(1) in the group
    all_P = sorted({p for n in names for p, _ in data[n]})
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(names):
        P = [p for p, _ in data[name]]
        T = [t for _, t in data[name]]
        S = [base / t for t in T]
        c = f"C{i}"
        if len(P) == 1:
            # Single point - draw a horizontal dashed line across the
            # comparison's full worker range so the magnitude is visible.
            xs = [min(all_P), max(all_P)] if len(all_P) > 1 else [1]
            if len(xs) == 1:
                ax.plot(xs, S, "o", color=c,
                        label=f"{name} ({T[0]:.2f}s, {S[0]:.2f}x)")
            else:
                ax.plot(xs, [S[0]] * 2, "--", color=c,
                        label=f"{name} ({T[0]:.2f}s, {S[0]:.2f}x)")
        else:
            ax.plot(P, S, "o-", color=c,
                    label=f"{name} (T1={T[0]:.2f}s)")
            # Amdahl fit on self-speedup, then scale to the shared axis.
            S_self = np.array([T[0] / t for t in T])
            (f_hat,), _ = curve_fit(amdahl, P, S_self,
                                    p0=[0.95], bounds=(0.0, 1.0))
            # Extend past the measured range so the curve visibly flattens
            # toward its asymptote instead of looking quasi-linear.
            pp = np.linspace(min(P), max(max(P) * 4, 64), 400)
            ax.plot(pp, S[0] * amdahl(pp, f_hat), "--", color=c, alpha=0.7,
                    label=f"{name} Amdahl f={f_hat:.3f}")
            # Asymptotic speed-up S_inf = 1/(1-f), scaled onto shared axis.
            if f_hat < 0.999:
                s_inf = S[0] / (1.0 - f_hat)
                ax.axhline(s_inf, color=c, linestyle=":", alpha=0.5,
                           label=f"{name} S∞={s_inf:.1f}x")

    ax.set_xlabel("workers P")
    ax.set_ylabel("speed-up S(P)")
    ax.set_title(title)
    if max(all_P) > 1:
        ax.set_xscale("log", base=2)
        # Show ticks across the full plotted range (the Amdahl fit is
        # extended past the measured points so the asymptote is visible).
        p_max = max(max(all_P) * 4, 64)
        ticks = [p for p in (1, 2, 4, 8, 16, 32, 64, 128, 256) if p <= p_max]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(p) for p in ticks])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"saved {fname}")


COMPARISONS = [
    (["task6", "task7"],                       "compare_6_vs_7.png",   "task6 vs task7"),
    (["task7", "task8"],                       "compare_7_vs_8.png",   "task7 vs task8"),
    (["task8", "task9"],                       "compare_8_vs_9.png",   "task8 vs task9"),
    (["task8", "task9", "task10"],             "compare_8_9_10.png",   "task8 vs task9 vs task10"),
    (["task8", "task10", "task11-8"],          "compare_8_10_11.png",  "task8 vs task10 vs task11-8"),
]
for names, fname, title in COMPARISONS:
    missing = [n for n in names if n not in data]
    if missing:
        print(f"skipping {fname}: missing {missing}")
        continue
    compare(names, fname, title)
