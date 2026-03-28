"""
run_phase1.py
=============
Phase 1: Environment & Baselines

Steps performed:
  1. Download the requested OGB dataset (ogbn-arxiv by default).
  2. Instantiate GraphSAGE and GAT models.
  3. Train for (warmup + epochs) epochs on the original (un-reordered) graph.
  4. Record and save per-epoch wall-clock timing to results/timing.csv.

Usage:
    python run_phase1.py --dataset ogbn-arxiv --epochs 50 --warmup 10
    python run_phase1.py --dataset ogbn-products --epochs 50 --warmup 10

GPU profiling with ncu (run on the GPU machine):
    ncu --metrics l2tex__t_sector_hit_rate.pct \
        --target-processes all \
        -o results/ncu_baseline_arxiv \
        python run_phase1.py --dataset ogbn-arxiv --epochs 1 --warmup 0
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

# Make sure sibling packages resolve correctly when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from models.graphsage import GraphSAGE
from models.gat import GAT
from profiling.timer import EpochTimer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets")


# ---------------------------------------------------------------------------
# Training loop
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
        correct = pred[idx].eq(y[idx]).sum().item()
        results[split] = correct / idx.numel()
    return results


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    data,
    split_idx,
    device,
    warmup: int,
    epochs: int,
    dataset_name: str,
):
    n = data.num_nodes
    in_channels = data.x.shape[1]
    num_classes = data.y.max().item() + 1

    print(f"\n{'='*60}")
    print(f" Model: {model_name} | Dataset: {dataset_name}")
    print(f" Nodes: {n:,} | Features: {in_channels} | Classes: {num_classes}")
    print(f" Device: {device} | Warmup: {warmup} | Epochs: {epochs}")
    print(f"{'='*60}")

    if model_name == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_channels=256, out_channels=num_classes)
    elif model_name == "GAT":
        model = GAT(in_channels, hidden_channels=256, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_idx = split_idx["train"].to(device)

    timer = EpochTimer(warmup=warmup)
    total_epochs = warmup + epochs

    for epoch in range(1, total_epochs + 1):
        timer.start()
        loss = train_one_epoch(model, data, optimizer, train_idx, device)
        timer.stop()

    # Final accuracy
    acc = evaluate(model, data, split_idx, device)
    print(f"\n[{model_name}] Final accuracy:")
    for split, val in acc.items():
        print(f"  {split:5s}: {100*val:.2f}%")

    summary = timer.summary()
    print(f"\n[{model_name}] Timing summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "timing.csv")
    timer.save_csv(
        csv_path,
        extra_fields={
            "dataset": dataset_name,
            "model": model_name,
            "reordering": "baseline",
            "k": "N/A",
            "train_acc": round(acc.get("train", float("nan")), 4),
            "val_acc": round(acc.get("valid", float("nan")), 4),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline GNN Training")
    parser.add_argument("--dataset", default="ogbn-arxiv",
                        choices=["ogbn-arxiv", "ogbn-products"])
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of measured epochs (after warmup).")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup epochs excluded from timing stats.")
    parser.add_argument("--models", nargs="+", default=["GraphSAGE", "GAT"],
                        choices=["GraphSAGE", "GAT"])
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden channel width.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")
    if device.type == "cuda":
        print(f"[main] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[main] L2 cache: check ncu for l2tex__t_sector_hit_rate.pct")

    # Download / load dataset
    print(f"\n[main] Loading {args.dataset} ...")
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATASET_ROOT)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    for model_name in args.models:
        run_model(
            model_name=model_name,
            data=data,
            split_idx=split_idx,
            device=device,
            warmup=args.warmup,
            epochs=args.epochs,
            dataset_name=args.dataset,
        )

    print("\n[main] Phase 1 complete. Results saved to results/timing.csv")


if __name__ == "__main__":
    main()
