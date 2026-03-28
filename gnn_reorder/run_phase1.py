"""
run_phase1.py
=============
Phase 1: Environment & Baselines

Training mode is chosen automatically:
  - ogbn-arxiv   (~169K nodes)  → full-batch  (graph fits on GPU)
  - ogbn-products (~2.4M nodes) → mini-batch  (NeighborLoader, 2-hop sampling)

Full-batch:  graph data moved to GPU once; all nodes processed per epoch.
Mini-batch:  each batch is a sampled subgraph; only the batch moves to GPU.
             Sizes [15, 10] means sample 15 neighbors at hop-1, 10 at hop-2.

Usage:
    python run_phase1.py --dataset ogbn-arxiv   --epochs 50 --warmup 10 --gpu 1
    python run_phase1.py --dataset ogbn-products --epochs 10 --warmup 3  --gpu 1

GPU profiling with ncu (1 epoch, no warmup):
    ncu --metrics l2tex__t_sector_hit_rate.pct \\
        --target-processes all \\
        -o results/ncu_graphsage_arxiv_baseline \\
        python run_phase1.py --dataset ogbn-arxiv --epochs 1 --warmup 0 \\
            --models GraphSAGE --gpu 1
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader


# PyTorch 2.6 changed torch.load default to weights_only=True.
# OGB's cached dataset files include torch_geometric globals that must be
# allowlisted so they can be deserialized safely.
try:
    import torch_geometric.data.data as _pyg_data
    _safe_globals = [
        _pyg_data.DataEdgeAttr,
        _pyg_data.DataTensorAttr,
        _pyg_data.GlobalStorage,
    ]
    torch.serialization.add_safe_globals(_safe_globals)
except (ImportError, AttributeError):
    pass  # older PyG / PyTorch versions don't need this

sys.path.insert(0, os.path.dirname(__file__))

from models.graphsage import GraphSAGE
from models.gat import GAT
from profiling.timer import EpochTimer

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "datasets")

# Datasets where full-batch is safe (node count roughly ≤ 500K on 80GB GPU)
FULLBATCH_DATASETS = {"ogbn-arxiv"}

# Defaults (overridden by CLI args; not used as globals after main() starts)
DEFAULT_NEIGHBOR_SIZES = [15, 10]
DEFAULT_BATCH_SIZE     = 1024


# ---------------------------------------------------------------------------
# GPU data pre-loading  (full-batch only)
# ---------------------------------------------------------------------------

class GPUData:
    """
    Pre-loads the entire graph onto GPU so .to(device) is never called
    inside the hot training loop. Only used for full-batch datasets.
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
            print(f"[GPUData] GPU memory after data load: {mem_gb:.2f} GB")


# ---------------------------------------------------------------------------
# CUDA event helpers
# ---------------------------------------------------------------------------

def _cuda_events():
    """Return a pair of CUDA events (start, end) with timing enabled."""
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _elapsed_ms(start, end):
    """Synchronise and return elapsed time in ms between two CUDA events."""
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ---------------------------------------------------------------------------
# Full-batch training  (ogbn-arxiv)
# ---------------------------------------------------------------------------

def train_fullbatch(model, gdata, optimizer):
    """
    Returns (loss, fwd_ms, bwd_ms, opt_ms) measured with CUDA events.
    Falls back gracefully to (loss, 0, 0, 0) on CPU.
    """
    model.train()
    optimizer.zero_grad()

    use_cuda = gdata.x.is_cuda
    if use_cuda:
        e0s, e0e = _cuda_events()   # forward
        e1s, e1e = _cuda_events()   # backward
        e2s, e2e = _cuda_events()   # optimizer

    if use_cuda: e0s.record()
    out  = model(gdata.x, gdata.edge_index)
    loss = F.cross_entropy(out[gdata.train_idx], gdata.y[gdata.train_idx])
    if use_cuda: e0e.record()

    if use_cuda: e1s.record()
    loss.backward()
    if use_cuda: e1e.record()

    if use_cuda: e2s.record()
    optimizer.step()
    if use_cuda: e2e.record()

    if use_cuda:
        fwd_ms = _elapsed_ms(e0s, e0e)
        bwd_ms = _elapsed_ms(e1s, e1e)
        opt_ms = _elapsed_ms(e2s, e2e)
    else:
        fwd_ms = bwd_ms = opt_ms = 0.0

    return loss.item(), fwd_ms, bwd_ms, opt_ms


@torch.no_grad()
def eval_fullbatch(model, gdata):
    model.eval()
    out  = model(gdata.x, gdata.edge_index)
    pred = out.argmax(dim=-1)
    return {
        "train": pred[gdata.train_idx].eq(gdata.y[gdata.train_idx]).float().mean().item(),
        "valid": pred[gdata.val_idx  ].eq(gdata.y[gdata.val_idx  ]).float().mean().item(),
        "test" : pred[gdata.test_idx ].eq(gdata.y[gdata.test_idx ]).float().mean().item(),
    }


# ---------------------------------------------------------------------------
# Mini-batch training  (ogbn-products and any large graph)
# ---------------------------------------------------------------------------

def make_loaders(data, split_idx, neighbor_sizes, batch_size):
    """
    Build NeighborLoaders for train / val splits.
    Each loader samples `neighbor_sizes` neighbors per hop.
    """
    common = dict(
        data=data,
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        num_workers=4,
    )
    train_loader = NeighborLoader(
        input_nodes=split_idx["train"], **common, shuffle=True
    )
    val_loader = NeighborLoader(
        input_nodes=split_idx["valid"], **common, shuffle=False
    )
    return train_loader, val_loader


def train_minibatch(model, loader, optimizer, device):
    """
    Run one full epoch over all training batches.
    Returns (avg_loss, fwd_ms, bwd_ms, opt_ms) accumulated over all batches.
    """
    model.train()
    total_loss = total_examples = 0
    total_fwd = total_bwd = total_opt = 0.0
    use_cuda = (device.type == "cuda")

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        n = batch.batch_size   # seed nodes only

        if use_cuda:
            e0s, e0e = _cuda_events()
            e1s, e1e = _cuda_events()
            e2s, e2e = _cuda_events()

        if use_cuda: e0s.record()
        out  = model(batch.x, batch.edge_index)[:n]
        y    = batch.y.squeeze(1)[:n]
        loss = F.cross_entropy(out, y)
        if use_cuda: e0e.record()

        if use_cuda: e1s.record()
        loss.backward()
        if use_cuda: e1e.record()

        if use_cuda: e2s.record()
        optimizer.step()
        if use_cuda: e2e.record()

        if use_cuda:
            total_fwd += _elapsed_ms(e0s, e0e)
            total_bwd += _elapsed_ms(e1s, e1e)
            total_opt += _elapsed_ms(e2s, e2e)

        total_loss    += float(loss) * n
        total_examples += n

    return total_loss / total_examples, total_fwd, total_bwd, total_opt


@torch.no_grad()
def eval_minibatch(model, loader, device):
    """Evaluate on the provided loader split."""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        n    = batch.batch_size
        out  = model(batch.x, batch.edge_index)[:n]
        pred = out.argmax(dim=-1)
        y    = batch.y.squeeze(1)[:n]
        correct += pred.eq(y).sum().item()
        total   += n
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-model runner  (dispatches to full-batch or mini-batch)
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    data,
    split_idx,
    device,
    warmup: int,
    epochs: int,
    dataset_name: str,
    in_channels: int,
    num_classes: int,
    n_nodes: int,
    neighbor_sizes=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    if neighbor_sizes is None:
        neighbor_sizes = DEFAULT_NEIGHBOR_SIZES
    use_fullbatch = dataset_name in FULLBATCH_DATASETS
    mode_str = "full-batch" if use_fullbatch else f"mini-batch (sizes={neighbor_sizes}, bs={batch_size})"

    print(f"\n{'='*60}")
    print(f" Model: {model_name} | Dataset: {dataset_name}")
    print(f" Nodes: {n_nodes:,} | Features: {in_channels} | Classes: {num_classes}")
    print(f" Mode : {mode_str}")
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

    # Accumulators for CUDA event timings across measured epochs
    phase_fwd_ms: list[float] = []
    phase_bwd_ms: list[float] = []
    phase_opt_ms: list[float] = []

    # ---- Full-batch path (ogbn-arxiv) --------------------------------------
    if use_fullbatch:
        gdata = GPUData(data, split_idx, device)
        for epoch in range(1, warmup + epochs + 1):
            timer.start()
            _, fwd, bwd, opt = train_fullbatch(model, gdata, optimizer)
            timer.stop()
            if epoch > warmup:           # only record measured epochs
                phase_fwd_ms.append(fwd)
                phase_bwd_ms.append(bwd)
                phase_opt_ms.append(opt)
        acc = eval_fullbatch(model, gdata)

    # ---- Mini-batch path (ogbn-products) -----------------------------------
    else:
        train_loader, val_loader = make_loaders(data, split_idx, neighbor_sizes, batch_size)
        for epoch in range(1, warmup + epochs + 1):
            timer.start()
            _, fwd, bwd, opt = train_minibatch(model, train_loader, optimizer, device)
            timer.stop()
            if epoch > warmup:
                phase_fwd_ms.append(fwd)
                phase_bwd_ms.append(bwd)
                phase_opt_ms.append(opt)
        val_acc = eval_minibatch(model, val_loader, device)
        acc = {"train": float("nan"), "valid": val_acc, "test": float("nan")}
        print(f"  (train/test acc skipped for speed on large graph)")

    print(f"\n[{model_name}] Accuracy:")
    for split, val in acc.items():
        if val == val:  # not nan
            print(f"  {split:5s}: {100*val:.2f}%")

    summary = timer.summary()
    print(f"\n[{model_name}] Timing summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # CUDA event phase summary
    def _mean(lst): return round(sum(lst) / len(lst), 3) if lst else float("nan")
    mean_fwd = _mean(phase_fwd_ms)
    mean_bwd = _mean(phase_bwd_ms)
    mean_opt = _mean(phase_opt_ms)
    if device.type == "cuda":
        print(f"\n[{model_name}] CUDA phase timing (mean over {len(phase_fwd_ms)} epochs):")
        print(f"  forward  : {mean_fwd:8.3f} ms")
        print(f"  backward : {mean_bwd:8.3f} ms")
        print(f"  optimizer: {mean_opt:8.3f} ms")
        print(f"  fwd+bwd  : {round(mean_fwd + mean_bwd, 3):8.3f} ms  (compute)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timer.save_csv(
        os.path.join(RESULTS_DIR, "timing.csv"),
        extra_fields={
            "dataset":        dataset_name,
            "model":          model_name,
            "reordering":     "baseline",
            "mode":           "full-batch" if use_fullbatch else "mini-batch",
            "k":              "N/A",
            "train_acc":      round(acc.get("train", float("nan")), 4),
            "val_acc":        round(acc.get("valid", float("nan")), 4),
            "mean_fwd_ms":    mean_fwd,
            "mean_bwd_ms":    mean_bwd,
            "mean_opt_ms":    mean_opt,
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
                        help="Measured epochs (after warmup). "
                             "Suggest 50 for arxiv, 10 for products.")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup epochs excluded from stats. "
                             "Suggest 10 for arxiv, 3 for products.")
    parser.add_argument("--models", nargs="+", default=["GraphSAGE", "GAT"],
                        choices=["GraphSAGE", "GAT"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA GPU index (default: 0).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for large graphs.")
    parser.add_argument("--neighbor-sizes", type=int, nargs="+",
                        default=DEFAULT_NEIGHBOR_SIZES,
                        help="Neighbors sampled per hop (mini-batch only).")
    args = parser.parse_args()

    batch_size     = args.batch_size
    neighbor_sizes = args.neighbor_sizes

    # ---- Device setup -------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f"[main] GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        total_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
        print(f"[main] Total GPU memory: {total_mem:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[main] WARNING: No CUDA GPU found — running on CPU (will be slow).")
    print(f"[main] Device: {device}")

    is_fullbatch = args.dataset in FULLBATCH_DATASETS
    print(f"[main] Training mode: {'full-batch' if is_fullbatch else 'mini-batch (NeighborLoader)'}")

    # ---- Load dataset -------------------------------------------------------
    print(f"\n[main] Loading {args.dataset} ...")
    dataset    = PygNodePropPredDataset(name=args.dataset, root=DATASET_ROOT)
    data       = dataset[0]
    split_idx  = dataset.get_idx_split()

    in_channels = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    n_nodes     = data.num_nodes

    for model_name in args.models:
        run_model(
            model_name=model_name,
            data=data,
            split_idx=split_idx,
            device=device,
            warmup=args.warmup,
            epochs=args.epochs,
            dataset_name=args.dataset,
            in_channels=in_channels,
            num_classes=num_classes,
            n_nodes=n_nodes,
            neighbor_sizes=neighbor_sizes,
            batch_size=batch_size,
        )

    print("\n[main] Phase 1 complete. Results saved to results/timing.csv")


if __name__ == "__main__":
    main()
