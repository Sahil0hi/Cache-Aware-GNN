"""
reordering/hybrid_reorder.py
============================
Two-level hybrid reordering: METIS partition → intra-partition RCM.

Implements the two novel contributions from the project proposal:

Contribution 1 — Two-level hybrid permutation:
    pi_hybrid = pi_RCM_local ∘ pi_METIS

    Stage 1 (METIS):  Partition V into k blocks, minimising edge cut.
    Stage 2 (RCM):    Apply Reverse Cuthill-McKee within each partition to
                      reduce the local bandwidth of each diagonal block.

    The composed global index of node v ∈ V_p is:
        pi_hybrid(v) = block_offset(p) + pi_RCM_local(rank(v, V_p))

Contribution 2 — Cache-size-aware partition count:
    k* = ceil(alpha * n * d * b / C)

    where n=nodes, d=feature dim, b=bytes/element, C=L2 cache capacity.
    The headroom factor alpha ∈ [1.0, 1.5] accounts for inter-partition
    spill accesses and should be calibrated experimentally.
"""

import math

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import reverse_cuthill_mckee
from torch_geometric.data import Data

from reordering.apply_permutation import (
    apply_permutation,
    permutation_from_ordering,
    verify_edge_set,
)

try:
    import pymetis
    PYMETIS_AVAILABLE = True
except ImportError:
    PYMETIS_AVAILABLE = False
    print("[hybrid_reorder] WARNING: pymetis not installed. "
          "Install with:  pip install pymetis")

# A100 GPU L2 cache capacity in bytes (40 MB)
A100_L2_BYTES = 40 * 1024 * 1024


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_aware_k_star(
    n_nodes: int,
    feat_dim: int,
    bytes_per_element: int = 4,
    l2_bytes: int = A100_L2_BYTES,
    alpha: float = 1.0,
) -> int:
    """
    Derive the cache-optimal METIS partition count k* analytically.

    Formula (from Eq. 6 / 7 in the proposal):
        k* = ceil(alpha * n * d * b / C)

    Under the balanced-partition assumption |V_p| ≈ n/k, this guarantees
    that each partition's feature sub-matrix fits in the GPU L2 cache.
    The headroom factor alpha > 1.0 reserves cache capacity for
    inter-partition spill accesses.

    Args:
        n_nodes           : number of nodes in the graph
        feat_dim          : feature dimension d_l
        bytes_per_element : bytes per feature element (4 for float32)
        l2_bytes          : GPU L2 cache capacity in bytes (default: A100 40 MB)
        alpha             : headroom factor in [1.0, 1.5]; default 1.0 gives
                            the lower bound (zero edge-cut assumption)

    Returns:
        k_star : optimal number of METIS partitions (minimum 2)
    """
    feat_bytes = n_nodes * feat_dim * bytes_per_element
    k_star = math.ceil(alpha * feat_bytes / l2_bytes)
    # METIS requires k ≥ 2 to be meaningful
    return max(2, k_star)


def hybrid_reorder(
    data: Data,
    k: int,
    verify: bool = True,
) -> tuple[Data, torch.Tensor]:
    """
    Apply the two-level hybrid reordering to a PyG graph.

    Stage 1 – METIS:  Partition V into k blocks, minimising edge cut.
    Stage 2 – RCM:    Apply Reverse Cuthill-McKee within each partition block
                      to minimise the local bandwidth of that block.

    Args:
        data   : PyG Data object.  The graph is treated as undirected;
                 edge_index should contain both (u,v) and (v,u).
        k      : Number of METIS partitions.
                 Use cache_aware_k_star() to obtain the hardware-derived value.
        verify : If True, run an edge-set correctness check (A' = P A P^T).

    Returns:
        data_perm : Permuted PyG Data object.
        perm      : LongTensor[n]; perm[old_idx] = new_idx.
    """
    if not PYMETIS_AVAILABLE:
        raise RuntimeError("pymetis is required. Install with: pip install pymetis")

    n = data.num_nodes
    print(f"[Hybrid] n={n:,} nodes, k={k} METIS partitions")

    # ------------------------------------------------------------------
    # Stage 1: METIS partitioning
    # ------------------------------------------------------------------
    print("[Hybrid] Running METIS ...")
    adjacency = _build_pymetis_adjacency(data)
    n_cuts, part_labels_list = pymetis.part_graph(k, adjacency=adjacency)
    part_labels = np.array(part_labels_list, dtype=np.int64)
    print(f"[Hybrid] METIS done. Edge cuts={n_cuts:,}")

    # nodes_in_part[p] = original node ids assigned to partition p
    nodes_in_part = [np.where(part_labels == p)[0] for p in range(k)]

    # ------------------------------------------------------------------
    # Pre-process edges in O(|E|) using NumPy vectorisation.
    # Identify all intra-partition edges and store their local indices.
    # ------------------------------------------------------------------
    src_np = data.edge_index[0].numpy()
    dst_np = data.edge_index[1].numpy()

    src_part = part_labels[src_np]
    dst_part = part_labels[dst_np]
    intra_mask = src_part == dst_part

    src_intra = src_np[intra_mask]
    dst_intra = dst_np[intra_mask]
    part_intra = src_part[intra_mask]

    # Global local-id array: local_id[orig_node] = index within its partition
    local_id = np.empty(n, dtype=np.int64)
    for p, nodes in enumerate(nodes_in_part):
        local_id[nodes] = np.arange(len(nodes), dtype=np.int64)

    local_src = local_id[src_intra]
    local_dst = local_id[dst_intra]

    # ------------------------------------------------------------------
    # Stage 2: Intra-partition RCM
    # ------------------------------------------------------------------
    print("[Hybrid] Applying intra-partition RCM ...")
    global_ordering = np.empty(n, dtype=np.int64)
    offset = 0

    for p, part_nodes in enumerate(nodes_in_part):
        part_size = len(part_nodes)
        if part_size == 0:
            continue

        # Extract intra-partition edges for this partition
        edge_mask = part_intra == p
        p_rows = local_src[edge_mask]
        p_cols = local_dst[edge_mask]

        if part_size == 1 or p_rows.size == 0:
            # Trivial partition or no internal edges: keep METIS order
            global_ordering[offset: offset + part_size] = part_nodes
            offset += part_size
            continue

        # Build symmetric CSR for the induced subgraph
        vals = np.ones(len(p_rows), dtype=np.float32)
        csr = sp.coo_matrix(
            (vals, (p_rows, p_cols)), shape=(part_size, part_size)
        ).tocsr()
        csr = csr + csr.T
        csr.data[:] = 1.0

        # RCM: local_ordering[new_local_idx] = old_local_idx
        local_ordering = reverse_cuthill_mckee(csr, symmetric_mode=True)

        # Map back to original global node ids
        global_ordering[offset: offset + part_size] = part_nodes[local_ordering]
        offset += part_size

    assert offset == n, f"Node count mismatch: placed {offset}, expected {n}"
    print("[Hybrid] Intra-partition RCM complete.")

    # ------------------------------------------------------------------
    # Build permutation tensor and apply to graph
    # ------------------------------------------------------------------
    perm = permutation_from_ordering(global_ordering, n)
    data_perm = apply_permutation(data, perm)

    if verify:
        verify_edge_set(data, data_perm, perm)

    bw_before = _estimate_bandwidth(data)
    bw_after = _estimate_bandwidth(data_perm)
    reduction = 100.0 * (bw_before - bw_after) / max(1, bw_before)
    print(f"[Hybrid] Bandwidth  before: {bw_before:,}  "
          f"after: {bw_after:,}  ({reduction:.1f}% reduction)")

    return data_perm, perm


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_pymetis_adjacency(data: Data) -> list:
    """
    Build a pymetis-compatible adjacency list from a PyG edge_index.

    Uses NumPy argsort + searchsorted for O(E log E) complexity without
    any Python-level loops over edges.

    Returns:
        list of length n; element i is a numpy array of neighbour indices.
    """
    n = data.num_nodes
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()

    sort_idx = np.argsort(src, kind="stable")
    src_sorted = src[sort_idx]
    dst_sorted = dst[sort_idx]

    # Boundaries between consecutive source nodes
    boundaries = np.searchsorted(src_sorted, np.arange(n + 1))
    return [dst_sorted[boundaries[i]: boundaries[i + 1]] for i in range(n)]


def _estimate_bandwidth(data: Data, sample_edges: int = 500_000) -> int:
    """Estimate adjacency matrix bandwidth from a sample of edges."""
    src, dst = data.edge_index[0], data.edge_index[1]
    if src.numel() > sample_edges:
        idx = torch.randperm(src.numel())[:sample_edges]
        src, dst = src[idx], dst[idx]
    return int((src - dst).abs().max().item())
