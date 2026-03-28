"""
compare_results.py
==================
Reads results/timing.csv and prints a formatted ablation table.
Also saves a matplotlib bar chart to results/timing_comparison.png.

Run after completing Phase 1 and Phase 2:
    python compare_results.py
"""

import os
import sys
import csv
import math

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "timing.csv")


def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def print_table(rows):
    headers = ["dataset", "model", "reordering", "k",
               "mean_ms_per_epoch", "std_ms_per_epoch", "val_acc"]
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows))
                  for h in headers}

    def fmt_row(r):
        return "  ".join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers)

    sep = "  ".join("-" * col_widths[h] for h in headers)
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)

    print("\n" + "=" * len(sep))
    print("  ABLATION TABLE")
    print("=" * len(sep))
    print(header_line)
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("=" * len(sep) + "\n")


def plot_results(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not available; skipping plot.")
        return

    datasets = sorted(set(r["dataset"] for r in rows))
    models = sorted(set(r["model"] for r in rows))

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                             squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        ds_rows = [r for r in rows if r["dataset"] == ds]
        reorderings = sorted(set(r["reordering"] for r in ds_rows))

        x = range(len(reorderings))
        width = 0.35
        offsets = [-width / 2, width / 2] if len(models) == 2 else [0]

        for m_idx, model in enumerate(models):
            means, stds = [], []
            for ro in reorderings:
                row = next((r for r in ds_rows
                            if r["model"] == model and r["reordering"] == ro), None)
                means.append(float(row["mean_ms_per_epoch"]) if row else 0)
                stds.append(float(row["std_ms_per_epoch"]) if row else 0)

            bars = ax.bar(
                [xi + offsets[m_idx % len(offsets)] for xi in x],
                means,
                width,
                yerr=stds,
                label=model,
                capsize=4,
            )

        ax.set_title(ds)
        ax.set_ylabel("ms / epoch")
        ax.set_xticks(list(x))
        ax.set_xticklabels(reorderings, rotation=30, ha="right")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "timing_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"[compare] Plot saved → {out_path}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[compare] No results file found at {CSV_PATH}")
        print("  Run run_phase1.py and run_phase2.py first.")
        sys.exit(1)

    rows = load_results(CSV_PATH)
    print(f"[compare] Loaded {len(rows)} result rows from {CSV_PATH}")
    print_table(rows)
    plot_results(rows)


if __name__ == "__main__":
    main()
