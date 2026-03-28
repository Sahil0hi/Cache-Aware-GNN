"""
data/download_datasets.py
Downloads ogbn-arxiv and ogbn-products from the Open Graph Benchmark,
converts them to PyG Data objects, and saves them to disk.
"""

import argparse
import os
import torch
from ogb.nodeproppred import PygNodePropPredDataset

import torch.serialization

# Allowlist PyG classes for PyTorch 2.6+ weights_only loading
try:
    from torch_geometric.data.data import DataEdgeAttr
    # You may also need NodeStorage or EdgeStorage depending on the dataset
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    
    torch.serialization.add_safe_globals([DataEdgeAttr, GlobalStorage, NodeStorage, EdgeStorage])
except ImportError:
    # Fallback for environments where PyG is not yet fully initialized
    pass



import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
    torch.serialization.add_safe_globals([
        DataEdgeAttr, 
        DataTensorAttr, 
        GlobalStorage, 
        NodeStorage, 
        EdgeStorage
    ])
except ImportError:
    pass

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
