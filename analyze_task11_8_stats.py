# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Summarize per-building statistics from a task stats CSV (task 10 / 11-8).

Reads columns: building_id, mean_temp, std_temp, pct_above_18, pct_below_15.

Default input is ``results/task11-8_stats.csv``. For task 10 runs::

    uv run python analyze_task11_8_stats.py \\
        --csv results/task10_stats.csv \\
        --figure results/task10_mean_temp_histograms.png \\
        --title \"Task 10\"
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Histograms and aggregates from stats CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=HERE / "results" / "task11-8_stats.csv",
        help="Path to stats CSV (default: results/task11-8_stats.csv)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=HERE / "results" / "task11-8_mean_temp_histograms.png",
        help="Output PNG path for histograms",
    )
    parser.add_argument(
        "--title",
        default="Task 11-8",
        help="Short label for figure title (default: Task 11-8)",
    )
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else HERE / args.csv
    out_fig = args.figure if args.figure.is_absolute() else HERE / args.figure

    # Columns: building_id, mean_temp, std_temp, pct_above_18, pct_below_15
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    mean_temp = raw[:, 1]
    std_temp = raw[:, 2]
    pct_above_18 = raw[:, 3]
    pct_below_15 = raw[:, 4]

    n = len(mean_temp)
    avg_mean = float(np.mean(mean_temp))
    avg_std = float(np.mean(std_temp))
    n_hot = int(np.sum(pct_above_18 >= 50.0))
    n_cold = int(np.sum(pct_below_15 >= 50.0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(mean_temp, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Mean temperature")
    axes[0].set_ylabel("Number of buildings")
    axes[0].set_title("Distribution of mean temperatures (40 bins)")
    axes[0].axvline(avg_mean, color="crimson", linestyle="--", linewidth=1.5, label=f"mean = {avg_mean:.4g}")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].hist(mean_temp, bins=80, color="teal", edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[1].set_xlabel("Mean temperature")
    axes[1].set_ylabel("Number of buildings")
    axes[1].set_title("Distribution of mean temperatures (80 bins)")
    axes[1].axvline(avg_mean, color="crimson", linestyle="--", linewidth=1.5)

    fig.suptitle(f"{args.title}: mean temperature across {n} buildings", fontsize=12)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    print(f"Loaded {n} buildings from {csv_path}")
    print()
    print("Average mean temperature:", avg_mean)
    print("Average of per-building std_temp:", avg_std)
    print()
    print("Buildings with at least 50% of area above 18°C:", n_hot)
    print("Buildings with at least 50% of area below 15°C:", n_cold)
    print()
    print(f"Figure saved: {out_fig}")


if __name__ == "__main__":
    main()
