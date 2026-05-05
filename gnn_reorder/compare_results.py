"""
compare_results.py
==================
Reads results from Phase 1/2 (timing.csv) and Phase 3 (phase3_results.csv)
and prints a combined ablation table.

Also generates:
  • results/timing_comparison.png  — bar chart of ms/epoch per configuration
  • results/figures/pareto.png     — TRR vs ms/epoch Pareto scatter plot

Run after completing any combination of Phase 1–3:
    python compare_results.py
"""

import math
import os
import sys
import csv

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR    = os.path.join(RESULTS_DIR, "figures")
TIMING_CSV     = os.path.join(RESULTS_DIR, "timing.csv")
PHASE3_CSV     = os.path.join(RESULTS_DIR, "phase3_results.csv")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a CSV file into a list of dicts; return [] if file absent."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merge_rows(timing_rows, phase3_rows):
    """
    Combine Phase 1/2 timing rows with Phase 3 rows.

    Phase 3 rows may contain richer fields (TRR, ACC, k_star).
    Phase 1/2 rows are kept as-is; fields absent in those rows are left blank.
    Duplicate (dataset, model, reordering) entries from Phase 3 override Phase 1/2.
    """
    # Index Phase 3 rows by (dataset, model, reordering) for dedup
    p3_index = {}
    for r in phase3_rows:
        key = (r.get("dataset"), r.get("model"), r.get("reordering"))
        p3_index[key] = r

    merged = []
    seen = set()
    for r in timing_rows:
        key = (r.get("dataset"), r.get("model"), r.get("reordering"))
        if key in p3_index:
            # Phase 3 row has more fields — prefer it
            merged.append(p3_index[key])
        else:
            merged.append(r)
        seen.add(key)

    # Append Phase 3 configs not present in timing.csv
    for r in phase3_rows:
        key = (r.get("dataset"), r.get("model"), r.get("reordering"))
        if key not in seen:
            merged.append(r)
            seen.add(key)

    return merged


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def print_table(rows):
    # Prefer phase3 columns when available, fall back to timing.csv columns
    preferred = [
        "dataset", "model", "reordering", "k", "k_star",
        "mean_ms_per_epoch", "std_ms_per_epoch",
        "val_acc", "cache_coverage_pct", "temporal_reuse_ratio",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    headers = [h for h in preferred if h in all_keys]

    if not headers:
        print("[compare] No recognised columns found in results.")
        return

    col_widths = {
        h: max(len(h), max(len(str(r.get(h, ""))) for r in rows))
        for h in headers
    }

    sep = "  ".join("-" * col_widths[h] for h in headers)
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)

    print("\n" + "=" * len(sep))
    print("  COMBINED ABLATION TABLE  (Phase 1 / 2 / 3)")
    print("=" * len(sep))
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))
    print("=" * len(sep) + "\n")


# ---------------------------------------------------------------------------
# Bar chart (ms/epoch per configuration)
# ---------------------------------------------------------------------------

def plot_bar_chart(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not available; skipping bar chart.")
        return

    datasets = sorted(set(r["dataset"] for r in rows))
    models   = sorted(set(r["model"]   for r in rows))

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(7 * len(datasets), 5), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        ds_rows    = [r for r in rows if r["dataset"] == ds]
        reorderings = sorted(set(r["reordering"] for r in ds_rows))

        x      = range(len(reorderings))
        width  = 0.35
        n_mod  = len(models)
        offsets = [width * (i - (n_mod - 1) / 2) for i in range(n_mod)]

        for m_idx, model in enumerate(models):
            means, stds = [], []
            for ro in reorderings:
                row = next(
                    (r for r in ds_rows
                     if r["model"] == model and r["reordering"] == ro),
                    None,
                )
                try:
                    means.append(float(row["mean_ms_per_epoch"]) if row else 0)
                    stds.append(float(row["std_ms_per_epoch"]) if row else 0)
                except (TypeError, ValueError):
                    means.append(0)
                    stds.append(0)

            ax.bar(
                [xi + offsets[m_idx] for xi in x],
                means, width,
                yerr=stds, label=model, capsize=4,
            )

        ax.set_title(ds)
        ax.set_ylabel("ms / epoch")
        ax.set_xticks(list(x))
        ax.set_xticklabels(reorderings, rotation=35, ha="right", fontsize=8)
        ax.legend()

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "timing_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Bar chart saved → {out_path}")


# ---------------------------------------------------------------------------
# Pareto plot (TRR vs ms/epoch)
# ---------------------------------------------------------------------------

def plot_pareto(rows):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[compare] matplotlib not available; skipping Pareto plot.")
        return

    # Filter rows that have both timing and TRR
    valid = []
    for r in rows:
        try:
            ms  = float(r.get("mean_ms_per_epoch", "nan"))
            trr = float(r.get("temporal_reuse_ratio", "nan"))
            if math.isnan(ms) or math.isnan(trr) or ms <= 0:
                continue
        except (TypeError, ValueError):
            continue
        valid.append({**r, "_ms": ms, "_trr": trr})

    if not valid:
        print("[compare] No rows with both timing and TRR; skipping Pareto plot.")
        return

    from collections import defaultdict
    groups = defaultdict(list)
    for r in valid:
        key = (r.get("model", "?"), r.get("reordering", "?"))
        groups[key].append((r["_ms"], r["_trr"]))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, ((model, reorder), vals) in enumerate(sorted(groups.items())):
        ms_mean  = float(np.mean([v[0] for v in vals]))
        trr_mean = float(np.mean([v[1] for v in vals]))
        ax.scatter(
            ms_mean, trr_mean,
            marker=markers[idx % len(markers)],
            color=colours[idx % len(colours)],
            s=130, zorder=3,
            label=f"{model} / {reorder}",
        )
        ax.annotate(
            f"{model}\n{reorder}",
            xy=(ms_mean, trr_mean),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Mean epoch time (ms)  — lower is better", fontsize=10)
    ax.set_ylabel("Temporal Reuse Ratio (%)  — higher is better", fontsize=10)
    ax.set_title("Cache-Efficiency / Speed Pareto Frontier", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "pareto.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] Pareto plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timing_rows = load_csv(TIMING_CSV)
    phase3_rows = load_csv(PHASE3_CSV)

    if not timing_rows and not phase3_rows:
        print(f"[compare] No results found.")
        print(f"  Expected: {TIMING_CSV}")
        print(f"         or {PHASE3_CSV}")
        print("  Run run_phase1.py, run_phase2.py, or run_phase3.py first.")
        sys.exit(1)

    if timing_rows:
        print(f"[compare] Loaded {len(timing_rows)} rows from timing.csv")
    if phase3_rows:
        print(f"[compare] Loaded {len(phase3_rows)} rows from phase3_results.csv")

    rows = merge_rows(timing_rows, phase3_rows)
    print_table(rows)
    plot_bar_chart(rows)
    plot_pareto(rows)


if __name__ == "__main__":
    main()
