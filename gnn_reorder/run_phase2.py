"""
run_phase2.py
=============
Phase 2: Baseline Reordering

Steps performed:
  1. Load dataset (same as Phase 1).
  2. Apply the chosen reordering method (RCM or METIS).
  3. Validate correctness: verify that A' = P A P^T preserves the edge set.
  4. Train GraphSAGE and GAT on the reordered graph, record timing.
  5. Append results to results/timing.csv for comparison with the baseline.

Usage:
    # RCM reordering
    python run_phase2.py --dataset ogbn-arxiv --method rcm

    # METIS with a single k
    python run_phase2.py --dataset ogbn-arxiv --method metis --k 8

    # METIS with a sweep over several k values
    python run_phase2.py --dataset ogbn-arxiv --method metis --k 4 8 16 32

    # Skip accuracy eval and run faster (useful during sweep)
    python run_phase2.py --dataset ogbn-arxiv --method rcm --no-eval
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

sys.path.insert(0, os.path.dirname(__file__))

from models.graphsage import GraphSAGE
from models.gat import GAT
from profiling.timer import EpochTimer
from reordering.rcm import rcm_reorder
from reordering.metis_reorder import metis_reorder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets")


# ---------------------------------------------------------------------------
# Shared training/eval (identical to Phase 1 — no data-loader needed here
# for the full-batch setting on ogbn-arxiv)
# ---------------------------------------------------------------------------

def train_one_epoch(model, data, optimizer, train_idx, device):
    model.train()
    optimizer.zero_grad()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze(1).to(device)
    out = model(x, edge_index)
    loss = F.cross_entropy(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, device):
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze(1).to(device)
    out = model(x, edge_index)
    pred = out.argmax(dim=-1)
    results = {}
    for split, idx in split_idx.items():
        idx = idx.to(device)
        results[split] = pred[idx].eq(y[idx]).sum().item() / idx.numel()
    return results


# ---------------------------------------------------------------------------
# Per-model runner (reused for both RCM and METIS)
# ---------------------------------------------------------------------------

def run_model(
    model_name,
    data_perm,
    split_idx,
    device,
    warmup,
    epochs,
    dataset_name,
    reordering_tag,  # e.g. "rcm" or "metis_k8"
    run_eval=True,
):
    n = data_perm.num_nodes
    in_channels = data_perm.x.shape[1]
    num_classes = int(data_perm.y.max().item()) + 1

    print(f"\n{'='*60}")
    print(f" Model: {model_name} | Reordering: {reordering_tag}")
    print(f" Nodes: {n:,} | Features: {in_channels} | Classes: {num_classes}")
    print(f"{'='*60}")

    if model_name == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_channels=256, out_channels=num_classes)
    else:
        model = GAT(in_channels, hidden_channels=256, out_channels=num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_idx = split_idx["train"].to(device)

    timer = EpochTimer(warmup=warmup)
    for epoch in range(1, warmup + epochs + 1):
        timer.start()
        train_one_epoch(model, data_perm, optimizer, train_idx, device)
        timer.stop()

    acc = evaluate(model, data_perm, split_idx, device) if run_eval else {}
    if acc:
        print(f"\n[{model_name}] Accuracy after {reordering_tag}:")
        for split, val in acc.items():
            print(f"  {split:5s}: {100*val:.2f}%")

    summary = timer.summary()
    print(f"\n[{model_name}] Timing: {summary['mean_ms_per_epoch']:.2f} ± "
          f"{summary['std_ms_per_epoch']:.2f} ms/epoch")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timer.save_csv(
        os.path.join(RESULTS_DIR, "timing.csv"),
        extra_fields={
            "dataset": dataset_name,
            "model": model_name,
            "reordering": reordering_tag,
            "k": reordering_tag.split("_k")[-1] if "_k" in reordering_tag else "N/A",
            "train_acc": round(acc.get("train", float("nan")), 4),
            "val_acc": round(acc.get("valid", float("nan")), 4),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Baseline Reordering")
    parser.add_argument("--dataset", default="ogbn-arxiv",
                        choices=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--method", required=True,
                        choices=["rcm", "metis"],
                        help="Reordering method to apply.")
    parser.add_argument("--k", type=int, nargs="+", default=[8],
                        help="METIS partition count(s). Can be multiple for a sweep.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["GraphSAGE", "GAT"],
                        choices=["GraphSAGE", "GAT"])
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip accuracy evaluation (faster for sweeps).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip edge-set correctness check.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}")

    # Load dataset
    print(f"\n[main] Loading {args.dataset} ...")
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATASET_ROOT)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    verify = not args.no_verify

    # -------- RCM -----------------------------------------------------------
    if args.method == "rcm":
        print("\n[main] Applying RCM reordering ...")
        data_perm, perm = rcm_reorder(data, verify=verify)

        for model_name in args.models:
            run_model(
                model_name=model_name,
                data_perm=data_perm,
                split_idx=split_idx,
                device=device,
                warmup=args.warmup,
                epochs=args.epochs,
                dataset_name=args.dataset,
                reordering_tag="rcm",
                run_eval=not args.no_eval,
            )

    # -------- METIS (one or several k values) --------------------------------
    elif args.method == "metis":
        for k in args.k:
            print(f"\n[main] Applying METIS reordering (k={k}) ...")
            data_perm, perm, _ = metis_reorder(data, k=k, verify=verify)

            for model_name in args.models:
                run_model(
                    model_name=model_name,
                    data_perm=data_perm,
                    split_idx=split_idx,
                    device=device,
                    warmup=args.warmup,
                    epochs=args.epochs,
                    dataset_name=args.dataset,
                    reordering_tag=f"metis_k{k}",
                    run_eval=not args.no_eval,
                )

    print("\n[main] Phase 2 complete. Results appended to results/timing.csv")
    print("[main] To compare with baseline, run:")
    print("         python compare_results.py")


if __name__ == "__main__":
    main()
