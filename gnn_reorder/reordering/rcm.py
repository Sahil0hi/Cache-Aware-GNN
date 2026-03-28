"""
reordering/rcm.py
Reverse Cuthill-McKee (RCM) reordering.

scipy.sparse.csgraph.reverse_cuthill_mckee returns an ordering array:
    ordering[0] = original node that becomes node 0 in the reordered graph

We convert this to a permutation tensor and apply it via apply_permutation.
"""

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from reordering.apply_permutation import (
    apply_permutation,
    verify_edge_set,
    permutation_from_ordering,
)


def rcm_reorder(data: Data, verify: bool = True) -> tuple[Data, torch.Tensor]:
    """
    Apply Reverse Cuthill-McKee reordering to a PyG graph.

    Args:
        data   : PyG Data object with edge_index (undirected graph expected).
        verify : If True, runs edge-set correctness check after reordering.

    Returns:
        data_perm : Permuted PyG Data object.
        perm      : LongTensor of shape [n]; perm[old_idx] = new_idx.
    """
    n = data.num_nodes
    print(f"[RCM] Computing RCM ordering for graph with {n:,} nodes ...")

    # Build symmetric CSR sparse matrix from edge_index
    # (scipy RCM expects a symmetric adjacency matrix)
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
    adj = (adj + adj.T)  # ensure symmetry (no-op for undirected graphs)
    adj.data[:] = 1.0    # unweighted

    # RCM returns ordering: ordering[i] = original node assigned new index i
    ordering = reverse_cuthill_mckee(adj, symmetric_mode=True)
    print(f"[RCM] Ordering computed. Building permutation ...")

    # Convert ordering → permutation tensor
    perm = permutation_from_ordering(ordering, n)

    # Apply permutation to the graph
    data_perm = apply_permutation(data, perm)

    if verify:
        verify_edge_set(data, data_perm, perm)

    bw_before = _estimate_bandwidth(data)
    bw_after  = _estimate_bandwidth(data_perm)
    print(f"[RCM] Estimated bandwidth before: {bw_before:,}")
    print(f"[RCM] Estimated bandwidth after : {bw_after:,}  "
          f"({100*(bw_before-bw_after)/bw_before:.1f}% reduction)")

    return data_perm, perm


def _estimate_bandwidth(data: Data, sample_edges: int = 500_000) -> int:
    """
    Estimate matrix bandwidth from a random sample of edges.
    Full computation is O(|E|) and is fast enough for most graphs.
    """
    src, dst = data.edge_index[0], data.edge_index[1]
    if src.numel() > sample_edges:
        idx = torch.randperm(src.numel())[:sample_edges]
        src, dst = src[idx], dst[idx]
    return int((src - dst).abs().max().item())
