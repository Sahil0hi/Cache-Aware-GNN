"""
reordering/metis_reorder.py
METIS-based node reordering.

pymetis.part_graph(nparts, adjacency=...) returns a list of partition labels
(one per node). We group nodes by partition and concatenate to build the
permutation ordering.
"""

import numpy as np
import torch
from torch_geometric.data import Data

try:
    import pymetis
    PYMETIS_AVAILABLE = True
except ImportError:
    PYMETIS_AVAILABLE = False
    print("[metis_reorder] WARNING: pymetis not installed. "
          "Install with:  pip install pymetis")

from reordering.apply_permutation import (
    apply_permutation,
    verify_edge_set,
    permutation_from_ordering,
)


def metis_reorder(
    data: Data,
    k: int,
    verify: bool = True,
) -> tuple[Data, torch.Tensor, np.ndarray]:
    """
    Apply METIS partitioning-based reordering to a PyG graph.

    Nodes within the same partition are grouped together; the ordering within
    each partition is arbitrary (will be refined by RCM in Phase 3 hybrid).

    Args:
        data   : PyG Data object.
        k      : Number of METIS partitions.
        verify : If True, runs edge-set correctness check.

    Returns:
        data_perm  : Permuted PyG Data object.
        perm       : LongTensor[n]; perm[old_idx] = new_idx.
        part_labels: np.ndarray[n] of partition ids (0 ... k-1).
    """
    if not PYMETIS_AVAILABLE:
        raise RuntimeError("pymetis is required. Install with: pip install pymetis")

    n = data.num_nodes
    print(f"[METIS] Partitioning graph ({n:,} nodes) into k={k} parts ...")

    # Build adjacency list (pymetis format): list of np arrays, one per node
    adjacency = _build_adjacency_list(data)

    # METIS partitioning
    n_cuts, part_labels = pymetis.part_graph(k, adjacency=adjacency)
    part_labels = np.array(part_labels, dtype=np.int64)

    print(f"[METIS] Done. Edge cuts={n_cuts:,}  "
          f"Partition sizes: min={_part_sizes(part_labels, k).min()}  "
          f"max={_part_sizes(part_labels, k).max()}")

    # Build ordering: concatenate node ids grouped by partition
    ordering = []
    for p in range(k):
        nodes_in_part = np.where(part_labels == p)[0]
        ordering.extend(nodes_in_part.tolist())
    ordering = np.array(ordering, dtype=np.int64)

    perm = permutation_from_ordering(ordering, n)

    # Apply permutation
    data_perm = apply_permutation(data, perm)

    if verify:
        verify_edge_set(data, data_perm, perm)

    # Report edge cut ratio
    total_edges = data.edge_index.shape[1]
    print(f"[METIS] Edge cut ratio: {n_cuts}/{total_edges//2} = "
          f"{100*n_cuts/(total_edges//2 + 1e-9):.2f}%")

    return data_perm, perm, part_labels


def _build_adjacency_list(data: Data) -> list:
    """
    Convert PyG edge_index to pymetis adjacency list format.
    Returns a list of length n; each element is a list of neighbor indices.
    """
    n = data.num_nodes
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    sort_idx = np.argsort(src, kind="stable")
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]
    boundaries = np.searchsorted(src_sorted, np.arange(n + 1))
    return [dst_sorted[boundaries[i]: boundaries[i + 1]] for i in range(n)]


def _part_sizes(part_labels: np.ndarray, k: int) -> np.ndarray:
    sizes = np.zeros(k, dtype=np.int64)
    for p in range(k):
        sizes[p] = (part_labels == p).sum()
    return sizes


def sweep_k(
    data: Data,
    k_values: list,
    verify: bool = True,
) -> dict:
    """
    Run METIS reordering for each k in k_values and return results dict.
    Useful for the hyperparameter sweep in Phase 2.

    Returns:
        results: dict mapping k -> {"data_perm": ..., "perm": ..., "part_labels": ...}
    """
    results = {}
    for k in k_values:
        print(f"\n{'='*50}")
        print(f" METIS sweep: k = {k}")
        print(f"{'='*50}")
        data_perm, perm, part_labels = metis_reorder(data, k=k, verify=verify)
        results[k] = {
            "data_perm": data_perm,
            "perm": perm,
            "part_labels": part_labels,
        }
    return results
