"""
run_phase1.py
=============
Phase 1: Environment & Baselines

Steps performed:
  1. Download the requested OGB dataset (ogbn-arxiv by default).
  2. Instantiate GraphSAGE and GAT models.
  3. Move all graph data (x, edge_index, y) to GPU ONCE before training.
  4. Train for (warmup + epochs) epochs on the original (un-reordered) graph.
  5. Record and save per-epoch wall-clock timing to results/timing.csv.

Usage:
    python run_phase1.py --dataset ogbn-arxiv --epochs 50 --warmup 10
    python run_phase1.py --dataset ogbn-arxiv --gpu 0          # use cuda:0
    python run_phase1.py --dataset ogbn-arxiv --gpu 1          # use cuda:1

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
# GPU data pre-loading
# ---------------------------------------------------------------------------

class GPUData:
    """
    Holds graph tensors that have already been moved to the GPU.
    We do this ONCE before training so that .to(device) is never called
    inside the hot training loop.
    """
    def __init__(self, data, split_idx, device):
        print(f"[GPUData] Moving tensors to {device} ...")
        self.x          = data.x.to(device)
        self.edge_index = data.edge_index.to(device)
        self.y          = data.y.squeeze(1).to(device)
        self.train_idx  = split_idx["train"].to(device)
        self.val_idx    = split_idx["valid"].to(device)
        self.test_idx   = split_idx["test"].to(device)
        if device.type == "cuda":
            mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[GPUData] GPU memory used after data load: {mem_gb:.2f} GB")


# ---------------------------------------------------------------------------
# Training loop  (all inputs are already on GPU — no .to() calls here)
# ---------------------------------------------------------------------------

def train_one_epoch(model, gdata, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(gdata.x, gdata.edge_index)
    loss = F.cross_entropy(out[gdata.train_idx], gdata.y[gdata.train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, gdata):
    model.eval()
    out = model(gdata.x, gdata.edge_index)
    pred = out.argmax(dim=-1)
    results = {
        "train": pred[gdata.train_idx].eq(gdata.y[gdata.train_idx]).float().mean().item(),
        "valid": pred[gdata.val_idx  ].eq(gdata.y[gdata.val_idx  ]).float().mean().item(),
        "test" : pred[gdata.test_idx ].eq(gdata.y[gdata.test_idx ]).float().mean().item(),
    }
    return results


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    gdata: GPUData,            # data already on GPU
    device,
    warmup: int,
    epochs: int,
    dataset_name: str,
    in_channels: int,
    num_classes: int,
    n_nodes: int,
):
    print(f"\n{'='*60}")
    print(f" Model: {model_name} | Dataset: {dataset_name}")
    print(f" Nodes: {n_nodes:,} | Features: {in_channels} | Classes: {num_classes}")
    print(f" Device: {device} | Warmup: {warmup} | Epochs: {epochs}")
    print(f"{'='*60}")

    if model_name == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_channels=256, out_channels=num_classes)
    elif model_name == "GAT":
        model = GAT(in_channels, hidden_channels=256, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    if device.type == "cuda":
        mem_gb = torch.cuda.memory_allocated(device) / 1024**3
        print(f" GPU memory after model load: {mem_gb:.2f} GB")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    timer = EpochTimer(warmup=warmup)
    total_epochs = warmup + epochs

    for epoch in range(1, total_epochs + 1):
        timer.start()
        loss = train_one_epoch(model, gdata, optimizer)  # no .to() inside!
        timer.stop()

    # Final accuracy
    acc = evaluate(model, gdata)
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
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which CUDA GPU index to use (default: 0).")
    args = parser.parse_args()

    # ---- Device setup -------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f"[main] GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        total_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
        print(f"[main] Total GPU memory: {total_mem:.1f} GB")
        print(f"[main] Tip: profile L2 hit rate with ncu --metrics l2tex__t_sector_hit_rate.pct")
    else:
        device = torch.device("cpu")
        print("[main] WARNING: No CUDA GPU found — running on CPU (will be slow).")
    print(f"[main] Device: {device}")

    # ---- Load dataset -------------------------------------------------------
    print(f"\n[main] Loading {args.dataset} ...")
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATASET_ROOT)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    in_channels  = data.x.shape[1]
    num_classes  = int(data.y.max().item()) + 1
    n_nodes      = data.num_nodes

    # ---- Move ALL graph data to GPU ONCE (not inside the training loop) -----
    gdata = GPUData(data, split_idx, device)

    for model_name in args.models:
        run_model(
            model_name=model_name,
            gdata=gdata,
            device=device,
            warmup=args.warmup,
            epochs=args.epochs,
            dataset_name=args.dataset,
            in_channels=in_channels,
            num_classes=num_classes,
            n_nodes=n_nodes,
        )

    print("\n[main] Phase 1 complete. Results saved to results/timing.csv")


if __name__ == "__main__":
    main()
