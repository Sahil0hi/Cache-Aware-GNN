"""
data/download_datasets.py
Downloads ogbn-arxiv and ogbn-products from the Open Graph Benchmark,
converts them to PyG Data objects, and saves them to disk.
"""

import argparse
import os
import torch
from ogb.nodeproppred import PygNodePropPredDataset

SUPPORTED = ["ogbn-arxiv", "ogbn-products"]
DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets")


def download(dataset_name: str, root: str = DEFAULT_ROOT) -> None:
    """Download and cache the OGB dataset."""
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)

    print(f"[download] Fetching {dataset_name} → {root}")
    dataset = PygNodePropPredDataset(name=dataset_name, root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    n_nodes = data.num_nodes
    n_edges = data.edge_index.shape[1]
    feat_dim = data.x.shape[1]

    print(f"  Nodes      : {n_nodes:,}")
    print(f"  Edges      : {n_edges:,}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Classes    : {dataset.num_classes}")
    print(f"  Train split: {split_idx['train'].numel():,}")
    print(f"  Val split  : {split_idx['valid'].numel():,}")
    print(f"  Test split : {split_idx['test'].numel():,}")
    print("[download] Done.\n")
    return dataset, split_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=SUPPORTED,
        choices=SUPPORTED,
        help="Which OGB datasets to download.",
    )
    args = parser.parse_args()

    for name in args.datasets:
        download(name)
