"""
run_phase3.py
=============
Phase 3: Novel Methods — Cache-Aware k* and Two-Level Hybrid Reordering.

This script implements and evaluates the two novel contributions proposed in
the project:

  (1)  Two-level hybrid permutation:
           pi_hybrid = pi_RCM_local ∘ pi_METIS
       METIS partitions into k blocks; RCM is then applied inside each block
       to reduce the local bandwidth of each diagonal tile.

  (2)  Cache-size-aware partition count:
           k* = ceil(alpha * n * d * b / C)
       k is derived analytically from the GPU's L2 cache capacity rather
       than swept as a hyperparameter.

Configurations run by this script (one dataset at a time):
  • baseline         — no reordering (reference)
  • rcm              — RCM-only
  • metis_k{k}       — METIS with swept k (optional, via --sweep-k)
  • metis_kstar      — METIS with cache-aware k*  (alpha=1.0)
  • hybrid_kstar     — Two-level hybrid with k*   (alpha=1.0)
  • hybrid_kstar_a{a}— Hybrid with k* scaled by headroom factor alpha

For each configuration the script records:
  • ms/epoch  (mean ± std over --epochs measured epochs)
  • TRR       (temporal reuse ratio — cache-efficiency proxy)
  • ACC       (analytical cache coverage — theoretical upper bound)
  • Val accuracy (to confirm lossless transformation)

Output files written to results/:
  • phase3_results.csv  — full ablation table
  • figures/sparsity_*.png — adjacency density plots before/after reordering
  • figures/pareto.png     — TRR vs ms/epoch Pareto scatter plot

Usage examples:
    # Quick run on arxiv (full-batch, GraphSAGE only, 20 measured epochs)
    python run_phase3.py --dataset ogbn-arxiv --models GraphSAGE --epochs 20

    # Full ablation with both models, METIS sweep, alpha calibration
    python run_phase3.py --dataset ogbn-arxiv \\
        --models GraphSAGE GAT \\
        --epochs 50 --warmup 10 \\
        --sweep-k 4 8 16 32 \\
        --alpha 1.0 1.25 1.5

    # Skip training (cache metrics only)
    python run_phase3.py --dataset ogbn-arxiv --no-train
"""

import argparse
import csv
import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

# PyTorch 2.6 safe-globals shim for OGB cached tensors
try:
    import torch_geometric.data.data as _pyg_data
    torch.serialization.add_safe_globals([
        _pyg_data.DataEdgeAttr,
        _pyg_data.DataTensorAttr,
        _pyg_data.GlobalStorage,
    ])
except (ImportError, AttributeError):
    pass

sys.path.insert(0, os.path.dirname(__file__))

from models.graphsage import GraphSAGE
from models.gat import GAT
from profiling.timer import EpochTimer
from profiling.cache_proxy import (
    analytical_cache_coverage,
    temporal_reuse_ratio,
    A100_L2_BYTES,
)
from profiling.sparsity_plot import plot_before_after
from reordering.rcm import rcm_reorder
from reordering.metis_reorder import metis_reorder
from reordering.hybrid_reorder import hybrid_reorder, cache_aware_k_star
from run_phase1 import (
    GPUData,
    train_fullbatch,
    eval_fullbatch,
    train_minibatch,
    eval_minibatch,
    make_loaders,
    FULLBATCH_DATASETS,
)

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets")
PHASE3_CSV   = os.path.join(RESULTS_DIR, "phase3_results.csv")

# Mini-batch defaults (ogbn-products)
DEFAULT_NEIGHBOR_SIZES = [15, 10]
DEFAULT_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Cache metric helpers
# ---------------------------------------------------------------------------

def compute_cache_metrics(data, split_idx, feat_dim, label=""):
    """Return a dict of TRR and ACC metrics for the given graph."""
    n = data.num_nodes
    acc_info = analytical_cache_coverage(n, feat_dim)

    train_idx = split_idx["train"]
    trr_info = temporal_reuse_ratio(
        data.edge_index, train_idx, sample_size=50_000
    )

    print(f"  [{label}]  ACC={100*acc_info['cache_coverage']:.1f}%  "
          f"TRR={100*trr_info['temporal_reuse_ratio']:.2f}%  "
          f"k*={acc_info['k_star']}")

    return {
        "cache_coverage":      acc_info["cache_coverage"],
        "estimated_miss_rate": acc_info["estimated_miss_rate"],
        "k_star":              acc_info["k_star"],
        "temporal_reuse_ratio": trr_info["temporal_reuse_ratio"],
        "mean_train_degree":   trr_info["mean_train_degree"],
    }


# ---------------------------------------------------------------------------
# Training helpers (full-batch and mini-batch)
# ---------------------------------------------------------------------------

def run_training(
    model_name: str,
    data,
    split_idx,
    device,
    warmup: int,
    epochs: int,
    dataset_name: str,
    run_eval: bool = True,
    neighbor_sizes=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Train a single model on data for the specified number of epochs.
    Returns (timer_summary_dict, accuracy_dict).
    """
    if neighbor_sizes is None:
        neighbor_sizes = DEFAULT_NEIGHBOR_SIZES

    use_fullbatch = dataset_name in FULLBATCH_DATASETS
    in_channels = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1

    if model_name == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_channels=256,
                          out_channels=num_classes).to(device)
    elif model_name == "GAT":
        model = GAT(in_channels, hidden_channels=256,
                    out_channels=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    timer = EpochTimer(warmup=warmup)

    if use_fullbatch:
        gdata = GPUData(data, split_idx, device)
        for epoch in range(1, warmup + epochs + 1):
            timer.start()
            train_fullbatch(model, gdata, optimizer)
            timer.stop()
        acc = eval_fullbatch(model, gdata) if run_eval else {}
    else:
        train_loader, val_loader = make_loaders(
            data, split_idx, neighbor_sizes, batch_size
        )
        for epoch in range(1, warmup + epochs + 1):
            timer.start()
            train_minibatch(model, train_loader, optimizer, device)
            timer.stop()
        val_acc = eval_minibatch(model, val_loader, device) if run_eval else float("nan")
        acc = {"train": float("nan"), "valid": val_acc, "test": float("nan")}

    return timer.summary(), acc


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def _append_csv(row: dict):
    """Append a result row to the phase3 CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(PHASE3_CSV)
    with open(PHASE3_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()),
                                extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[phase3] Row saved → {PHASE3_CSV}")


def _save_config_results(
    dataset: str,
    model: str,
    reordering_tag: str,
    k_value,
    k_star,
    alpha,
    timer_summary: dict,
    acc: dict,
    cache_metrics: dict,
):
    """Persist a single configuration result to phase3_results.csv."""
    row = {
        "dataset":               dataset,
        "model":                 model,
        "reordering":            reordering_tag,
        "k":                     k_value,
        "k_star":                k_star,
        "alpha":                 alpha,
        "mean_ms_per_epoch":     timer_summary.get("mean_ms_per_epoch", float("nan")),
        "std_ms_per_epoch":      timer_summary.get("std_ms_per_epoch", float("nan")),
        "epochs_measured":       timer_summary.get("epochs_measured", 0),
        "val_acc":               round(acc.get("valid", float("nan")), 4),
        "train_acc":             round(acc.get("train", float("nan")), 4),
        "cache_coverage_pct":    round(100 * cache_metrics.get("cache_coverage", 0), 2),
        "temporal_reuse_ratio":  round(100 * cache_metrics.get("temporal_reuse_ratio", 0), 4),
        "estimated_miss_rate":   round(100 * cache_metrics.get("estimated_miss_rate", 0), 2),
        "mean_train_degree":     cache_metrics.get("mean_train_degree", float("nan")),
    }
    _append_csv(row)


# ---------------------------------------------------------------------------
# Per-configuration runner
# ---------------------------------------------------------------------------

def run_config(
    tag: str,
    data_perm,
    split_idx,
    device,
    dataset_name: str,
    warmup: int,
    epochs: int,
    model_names: list,
    k_value,
    k_star: int,
    alpha: float,
    run_eval: bool,
    no_train: bool,
    neighbor_sizes,
    batch_size: int,
):
    """Train all requested models on the given permuted graph and save results."""
    feat_dim = data_perm.x.shape[1]

    print(f"\n{'='*60}")
    print(f"  Config: {tag}  |  n={data_perm.num_nodes:,}  d={feat_dim}")
    print(f"{'='*60}")

    cache_metrics = compute_cache_metrics(data_perm, split_idx, feat_dim, label=tag)

    if no_train:
        # Save a metrics-only row (no timing)
        for model_name in model_names:
            _save_config_results(
                dataset=dataset_name, model=model_name,
                reordering_tag=tag, k_value=k_value, k_star=k_star, alpha=alpha,
                timer_summary={}, acc={}, cache_metrics=cache_metrics,
            )
        return

    for model_name in model_names:
        print(f"\n  ---- {model_name} ----")
        timer_summary, acc = run_training(
            model_name=model_name,
            data=data_perm,
            split_idx=split_idx,
            device=device,
            warmup=warmup,
            epochs=epochs,
            dataset_name=dataset_name,
            run_eval=run_eval,
            neighbor_sizes=neighbor_sizes,
            batch_size=batch_size,
        )
        ms = timer_summary.get("mean_ms_per_epoch", float("nan"))
        std = timer_summary.get("std_ms_per_epoch", float("nan"))
        val = acc.get("valid", float("nan"))
        print(f"  [{model_name}]  {ms:.2f} ± {std:.2f} ms/epoch  "
              f"val_acc={100*val:.2f}%")

        _save_config_results(
            dataset=dataset_name, model=model_name,
            reordering_tag=tag, k_value=k_value, k_star=k_star, alpha=alpha,
            timer_summary=timer_summary, acc=acc, cache_metrics=cache_metrics,
        )


# ---------------------------------------------------------------------------
# Pareto plot
# ---------------------------------------------------------------------------

def plot_pareto(csv_path: str, out_path: str):
    """
    Scatter plot of TRR (y) vs ms/epoch (x) across all configurations in
    phase3_results.csv.  Configurations closer to the top-left corner
    (fast AND high reuse) are preferable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[pareto] matplotlib not available; skipping.")
        return

    if not os.path.isfile(csv_path):
        print(f"[pareto] Results CSV not found at {csv_path}; skipping.")
        return

    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ms  = float(row["mean_ms_per_epoch"])
                trr = float(row["temporal_reuse_ratio"])
                if math.isnan(ms) or math.isnan(trr):
                    continue
            except (KeyError, ValueError):
                continue
            rows.append(row)

    if not rows:
        print("[pareto] No valid rows for Pareto plot.")
        return

    # Aggregate: for each (model, reordering) pair take the mean across datasets
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["model"], r["reordering"])
        groups[key].append((float(r["mean_ms_per_epoch"]),
                            float(r["temporal_reuse_ratio"])))

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, ((model, reorder), vals) in enumerate(sorted(groups.items())):
        ms_mean  = float(np.mean([v[0] for v in vals]))
        trr_mean = float(np.mean([v[1] for v in vals]))
        ax.scatter(
            ms_mean, trr_mean,
            marker=markers[idx % len(markers)],
            color=colours[idx % len(colours)],
            s=120, zorder=3,
            label=f"{model} / {reorder}",
        )
        ax.annotate(
            f"{model}\n{reorder}",
            xy=(ms_mean, trr_mean),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Mean epoch time (ms)  — lower is better →", fontsize=10)
    ax.set_ylabel("Temporal Reuse Ratio (%)  — higher is better ↑", fontsize=10)
    ax.set_title("Cache-Efficiency / Speed Pareto Frontier", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[pareto] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Ablation table printer
# ---------------------------------------------------------------------------

def print_ablation_table(csv_path: str):
    """Print a formatted ablation table from phase3_results.csv."""
    if not os.path.isfile(csv_path):
        print(f"[table] No results found at {csv_path}")
        return

    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[table] Results file is empty.")
        return

    cols = ["dataset", "model", "reordering", "k", "k_star",
            "mean_ms_per_epoch", "std_ms_per_epoch",
            "val_acc", "cache_coverage_pct", "temporal_reuse_ratio"]
    cols = [c for c in cols if c in rows[0]]

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(c.ljust(widths[c]) for c in cols)

    print("\n" + "=" * len(sep))
    print("  PHASE 3 ABLATION TABLE")
    print("=" * len(sep))
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))
    print("=" * len(sep) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Cache-Aware k* and Hybrid Reordering"
    )
    parser.add_argument("--dataset", default="ogbn-arxiv",
                        choices=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--epochs", type=int, default=50,
                        help="Measured epochs (after warmup).")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup epochs excluded from statistics.")
    parser.add_argument("--models", nargs="+", default=["GraphSAGE", "GAT"],
                        choices=["GraphSAGE", "GAT"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--neighbor-sizes", type=int, nargs="+",
                        default=DEFAULT_NEIGHBOR_SIZES)
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 1.25],
                        help="Headroom factor(s) for k* calibration. "
                             "A value of 1.0 gives the lower-bound k*; "
                             "values >1.0 add cache headroom for inter-partition spill.")
    parser.add_argument("--sweep-k", type=int, nargs="+", default=[],
                        help="Additional METIS k values to sweep (optional).")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the baseline (no reordering) run.")
    parser.add_argument("--no-rcm", action="store_true",
                        help="Skip the RCM-only run.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip accuracy evaluation (faster).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip edge-set correctness check.")
    parser.add_argument("--no-train", action="store_true",
                        help="Compute cache metrics only; skip training.")
    parser.add_argument("--figures", action="store_true", default=True,
                        help="Generate sparsity and Pareto figures (default: on).")
    parser.add_argument("--no-figures", action="store_false", dest="figures")
    args = parser.parse_args()

    # ---- Device ----------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f"[main] GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("[main] WARNING: No CUDA GPU found — running on CPU.")
    print(f"[main] Device: {device}")

    # ---- Dataset ---------------------------------------------------------------
    print(f"\n[main] Loading {args.dataset} ...")
    dataset   = PygNodePropPredDataset(name=args.dataset, root=DATASET_ROOT)
    data      = dataset[0]
    split_idx = dataset.get_idx_split()

    n          = data.num_nodes
    feat_dim   = data.x.shape[1]
    verify     = not args.no_verify
    run_eval   = not args.no_eval
    no_train   = args.no_train

    print(f"[main] n={n:,}  feat_dim={feat_dim}  "
          f"edges={data.edge_index.shape[1]:,}")

    # ---- Cache-aware k* --------------------------------------------------------
    k_star_base = cache_aware_k_star(n, feat_dim, l2_bytes=A100_L2_BYTES, alpha=1.0)
    print(f"\n[main] Cache-aware k*  =  {k_star_base}  "
          f"(formula: ceil(n*d*b / C)  "
          f"= ceil({n}*{feat_dim}*4 / {A100_L2_BYTES}))")

    # ---- Common run_config kwargs ----------------------------------------------
    rc_kwargs = dict(
        split_idx=split_idx,
        device=device,
        dataset_name=args.dataset,
        warmup=args.warmup,
        epochs=args.epochs,
        model_names=args.models,
        run_eval=run_eval,
        no_train=no_train,
        neighbor_sizes=args.neighbor_sizes,
        batch_size=args.batch_size,
    )

    # Alias for the original graph (used in before/after sparsity plots)
    data_original = data

    # ===========================================================================
    # (1) Baseline
    # ===========================================================================
    if not args.no_baseline:
        run_config(
            tag="baseline", data_perm=data, k_value="N/A",
            k_star=k_star_base, alpha="N/A", **rc_kwargs,
        )

    # ===========================================================================
    # (2) RCM-only
    # ===========================================================================
    if not args.no_rcm:
        print("\n[main] Applying RCM reordering ...")
        data_rcm, _ = rcm_reorder(data, verify=verify)

        if args.figures:
            plot_before_after(
                data_original, data_rcm,
                label_before="Baseline (original)",
                label_after="RCM reordered",
                out_dir=FIGURES_DIR,
                stem=f"{args.dataset}_rcm",
            )

        run_config(
            tag="rcm", data_perm=data_rcm, k_value="N/A",
            k_star=k_star_base, alpha="N/A", **rc_kwargs,
        )

    # ===========================================================================
    # (3) METIS with swept k (optional)
    # ===========================================================================
    for k in args.sweep_k:
        print(f"\n[main] METIS sweep: k={k}")
        data_metis, _, _ = metis_reorder(data, k=k, verify=verify)

        if args.figures:
            plot_before_after(
                data_original, data_metis,
                label_before="Baseline (original)",
                label_after=f"METIS k={k}",
                out_dir=FIGURES_DIR,
                stem=f"{args.dataset}_metis_k{k}",
            )

        run_config(
            tag=f"metis_k{k}", data_perm=data_metis, k_value=k,
            k_star=k_star_base, alpha="N/A", **rc_kwargs,
        )

    # ===========================================================================
    # (4 & 5) Cache-aware METIS-only and Hybrid for each alpha
    # ===========================================================================
    for alpha in args.alpha:
        k_star = cache_aware_k_star(n, feat_dim, l2_bytes=A100_L2_BYTES, alpha=alpha)
        a_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
        print(f"\n[main] alpha={alpha}  →  k*={k_star}")

        # (4) METIS with cache-aware k*
        print(f"[main] METIS-only with k*={k_star} (alpha={alpha}) ...")
        tag_metis = "metis_kstar" if alpha == 1.0 else f"metis_kstar_a{a_str}"
        data_metis_k, _, _ = metis_reorder(data, k=k_star, verify=verify)

        if args.figures:
            plot_before_after(
                data_original, data_metis_k,
                label_before="Baseline (original)",
                label_after=f"METIS k*={k_star} (α={alpha})",
                out_dir=FIGURES_DIR,
                stem=f"{args.dataset}_metis_kstar_a{a_str}",
            )

        run_config(
            tag=tag_metis, data_perm=data_metis_k, k_value=k_star,
            k_star=k_star, alpha=alpha, **rc_kwargs,
        )

        # (5) Two-level hybrid with cache-aware k*
        print(f"[main] Hybrid (METIS k*={k_star} + intra-RCM, alpha={alpha}) ...")
        tag_hybrid = "hybrid_kstar" if alpha == 1.0 else f"hybrid_kstar_a{a_str}"
        data_hybrid, _ = hybrid_reorder(data, k=k_star, verify=verify)

        if args.figures:
            plot_before_after(
                data_original, data_hybrid,
                label_before="Baseline (original)",
                label_after=f"Hybrid k*={k_star} (α={alpha})",
                out_dir=FIGURES_DIR,
                stem=f"{args.dataset}_hybrid_kstar_a{a_str}",
            )

        run_config(
            tag=tag_hybrid, data_perm=data_hybrid, k_value=k_star,
            k_star=k_star, alpha=alpha, **rc_kwargs,
        )

    # ===========================================================================
    # Output
    # ===========================================================================
    print_ablation_table(PHASE3_CSV)

    if args.figures:
        plot_pareto(PHASE3_CSV, os.path.join(FIGURES_DIR, "pareto.png"))

    print(f"\n[main] Phase 3 complete.")
    print(f"  Results  → {PHASE3_CSV}")
    print(f"  Figures  → {FIGURES_DIR}/")
    print(f"\n  Run  python compare_results.py  to see the combined ablation table.")


if __name__ == "__main__":
    main()
