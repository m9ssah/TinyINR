"""
Produces two graphs:
1. runtime comparison
2. speedup comparison
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import ticker


def read_csv(path: str) -> list[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def plot_kernel(kernel_name: str, rows: list[dict], output_dir: str):
    by_f = defaultdict(list)
    for row in rows:
        f_val = int(row["F"])
        by_f[f_val].append(row)

    f_values = sorted(by_f.keys())

    # plot 1: runtime comparison
    fig, axes = plt.subplots(
        1, len(f_values), figsize=(6 * len(f_values), 5), sharey=True
    )
    if len(f_values) == 1:
        axes = [axes]

    fig.suptitle(f"{kernel_name} — Runtime (ms)", fontsize=14, fontweight="bold")

    for ax, f_val in zip(axes, f_values):
        f_rows = sorted(by_f[f_val], key=lambda r: int(r["N"]))
        ns = [int(r["N"]) for r in f_rows]
        cpu_ms = [float(r["cpu_ms"]) for r in f_rows]
        kernel_ms = [float(r["kernel_ms"]) for r in f_rows]
        e2e_ms = [float(r["e2e_gpu_ms"]) for r in f_rows]

        ax.plot(ns, cpu_ms, "o-", label="CPU", color="#e74c3c", linewidth=2)
        ax.plot(ns, e2e_ms, "s-", label="GPU (E2E)", color="#3498db", linewidth=2)
        ax.plot(ns, kernel_ms, "^-", label="GPU (kernel)", color="#2ecc71", linewidth=2)

        ax.set_xlabel("Coordinates (N)")
        ax.set_title(f"F = {f_val}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    axes[0].set_ylabel("Time (ms)")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{kernel_name}_runtime.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

    # plot 2: speedup comparison
    fig, axes = plt.subplots(
        1, len(f_values), figsize=(6 * len(f_values), 5), sharey=True
    )
    if len(f_values) == 1:
        axes = [axes]

    fig.suptitle(f"{kernel_name} — Speedup vs CPU", fontsize=14, fontweight="bold")

    for ax, f_val in zip(axes, f_values):
        f_rows = sorted(by_f[f_val], key=lambda r: int(r["N"]))
        ns = [int(r["N"]) for r in f_rows]
        kernel_speedup = [float(r["kernel_speedup"]) for r in f_rows]
        e2e_speedup = [float(r["e2e_speedup"]) for r in f_rows]

        ax.plot(
            ns,
            kernel_speedup,
            "^-",
            label="Kernel speedup",
            color="#2ecc71",
            linewidth=2,
        )
        ax.plot(
            ns, e2e_speedup, "s-", label="E2E speedup", color="#3498db", linewidth=2
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Breakeven")

        ax.set_xlabel("Coordinates (N)")
        ax.set_title(f"F = {f_val}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    axes[0].set_ylabel("Speedup (CPU time / GPU time)")
    fig.tight_layout()
    path = os.path.join(output_dir, f"{kernel_name}_speedup.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def main():
    results_dir = "benchmarks/results"
    output_dir = results_dir

    if len(sys.argv) > 1:
        csv_files = [sys.argv[1]]
    else:
        csv_files = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv")
        ]

    if not csv_files:
        print(f"No CSV files found in {results_dir}/")
        return

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        rows = read_csv(csv_path)
        if not rows:
            print("  (empty file, skipping)")
            continue

        by_kernel = defaultdict(list)
        for row in rows:
            by_kernel[row["kernel"]].append(row)

        for kernel_name, kernel_rows in by_kernel.items():
            print(f"  Plotting: {kernel_name}")
            plot_kernel(kernel_name, kernel_rows, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
