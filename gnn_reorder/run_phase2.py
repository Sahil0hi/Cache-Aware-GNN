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
from run_phase1 import GPUData  # reuse the same GPU pre-loading class

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets")


# ---------------------------------------------------------------------------
# Training / eval  (data is already on GPU in GPUData — no .to() calls)
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
    return {
        "train": pred[gdata.train_idx].eq(gdata.y[gdata.train_idx]).float().mean().item(),
        "valid": pred[gdata.val_idx  ].eq(gdata.y[gdata.val_idx  ]).float().mean().item(),
        "test" : pred[gdata.test_idx ].eq(gdata.y[gdata.test_idx ]).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Per-model runner (reused for both RCM and METIS)
# ---------------------------------------------------------------------------

def run_model(
    model_name,
    gdata: GPUData,          # data already on GPU
    device,
    warmup,
    epochs,
    dataset_name,
    reordering_tag,
    in_channels,
    num_classes,
    n_nodes,
    run_eval=True,
):
    print(f"\n{'='*60}")
    print(f" Model: {model_name} | Reordering: {reordering_tag}")
    print(f" Nodes: {n_nodes:,} | Features: {in_channels} | Classes: {num_classes}")
    print(f"{'='*60}")

    if model_name == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_channels=256, out_channels=num_classes)
    else:
        model = GAT(in_channels, hidden_channels=256, out_channels=num_classes)

    model = model.to(device)
    if device.type == "cuda":
        mem_gb = torch.cuda.memory_allocated(device) / 1024**3
        print(f" GPU memory after model load: {mem_gb:.2f} GB")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    timer = EpochTimer(warmup=warmup)
    for epoch in range(1, warmup + epochs + 1):
        timer.start()
        train_one_epoch(model, gdata, optimizer)   # no .to() inside!
        timer.stop()

    acc = evaluate(model, gdata) if run_eval else {}
    if acc:
        print(f"\n[{model_name}] Accuracy after {reordering_tag}:")
        for split, val in acc.items():
            print(f"  {split:5s}: {100*val:.2f}%")

    print(f"\n[{model_name}] Timing: {timer.summary()['mean_ms_per_epoch']:.2f} "
          f"± {timer.summary()['std_ms_per_epoch']:.2f} ms/epoch")

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
    else:
        device = torch.device("cpu")
        print("[main] WARNING: No CUDA GPU found — running on CPU.")
    print(f"[main] Device: {device}")

    # ---- Load dataset (CPU) -------------------------------------------------
    print(f"\n[main] Loading {args.dataset} ...")
    dataset = PygNodePropPredDataset(name=args.dataset, root=DATASET_ROOT)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    in_channels = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    n_nodes     = data.num_nodes

    verify = not args.no_verify

    def _run_config(data_perm, tag):
        """Move permuted data to GPU once, then train both models."""
        gdata = GPUData(data_perm, split_idx, device)
        for model_name in args.models:
            run_model(
                model_name=model_name,
                gdata=gdata,
                device=device,
                warmup=args.warmup,
                epochs=args.epochs,
                dataset_name=args.dataset,
                reordering_tag=tag,
                in_channels=in_channels,
                num_classes=num_classes,
                n_nodes=n_nodes,
                run_eval=not args.no_eval,
            )

    # -------- RCM ------------------------------------------------------------
    if args.method == "rcm":
        print("\n[main] Applying RCM reordering (CPU) ...")
        data_perm, perm = rcm_reorder(data, verify=verify)
        _run_config(data_perm, "rcm")

    # -------- METIS ----------------------------------------------------------
    elif args.method == "metis":
        for k in args.k:
            print(f"\n[main] Applying METIS reordering (k={k}) on CPU ...")
            data_perm, perm, _ = metis_reorder(data, k=k, verify=verify)
            _run_config(data_perm, f"metis_k{k}")

    print("\n[main] Phase 2 complete. Results appended to results/timing.csv")
    print("[main] Run  python compare_results.py  to see the ablation table.")


if __name__ == "__main__":
    main()
