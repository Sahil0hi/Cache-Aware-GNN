"""
reordering/apply_permutation.py

Core utility: apply a permutation P to the graph (A, H, y) and verify
that A' = P A P^T preserves the edge set.

Key function signatures
-----------------------
apply_permutation(data, perm)  -> permuted PyG Data object
verify_edge_set(data, data_perm, perm) -> bool (asserts equivalence)
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np


def apply_permutation(data: Data, perm: torch.Tensor) -> Data:
    """
    Apply a node permutation P to a PyG Data object.

    The permutation is represented as a 1-D LongTensor `perm` of length n
    where perm[i] = j means original node i maps to new index j.

    Transforms:
        A'          = P A P^T   (edge_index relabelled)
        H'[perm[i]] = H[i]      (feature rows reordered)
        y'[perm[i]] = y[i]      (label rows reordered)

    Args:
        data : Original PyG Data (must have .x, .edge_index, .y)
        perm : LongTensor of shape [n], a permutation of {0,...,n-1}

    Returns:
        data_perm : A new PyG Data object with permuted fields.
    """
    n = data.num_nodes
    assert perm.shape[0] == n, "perm length must equal number of nodes"
    assert perm.min() == 0 and perm.max() == n - 1, "perm must be a valid permutation"

    # Build inverse permutation: inv_perm[new_idx] = old_idx
    inv_perm = torch.argsort(perm)

    # ---- Permute edge_index ------------------------------------------------
    # edge_index stores [src, dst] with original indices.
    # Apply perm to both src and dst: new_index = perm[old_index]
    src, dst = data.edge_index[0], data.edge_index[1]
    new_src = perm[src]
    new_dst = perm[dst]
    new_edge_index = torch.stack([new_src, new_dst], dim=0)

    # ---- Permute node features and labels ----------------------------------
    # H'[perm[i]] = H[i]  <==>  H'[j] = H[inv_perm[j]]
    new_x = data.x[inv_perm] if data.x is not None else None
    new_y = data.y[inv_perm] if data.y is not None else None

    data_perm = Data(
        x=new_x,
        edge_index=new_edge_index,
        y=new_y,
        num_nodes=n,
    )

    # Carry over any split masks if they were stored on the Data object
    for attr in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, attr) and getattr(data, attr) is not None:
            mask = getattr(data, attr)
            new_mask = torch.zeros_like(mask)
            new_mask[perm] = mask
            setattr(data_perm, attr, new_mask)

    return data_perm


def verify_edge_set(
    data: Data,
    data_perm: Data,
    perm: torch.Tensor,
    verbose: bool = True,
) -> bool:
    """
    Verify that A' = P A P^T preserves the edge *set*.

    Strategy: convert edges to a canonical sorted set of (u, v) tuples
    under the permutation and check that the two sets are equal.

    Returns True if the edge sets match, raises AssertionError otherwise.
    """
    n = data.num_nodes

    # Original edge set in permuted index space
    src_orig, dst_orig = data.edge_index[0], data.edge_index[1]
    orig_edges_permuted = set(
        zip(perm[src_orig].tolist(), perm[dst_orig].tolist())
    )

    # Edge set of the permuted graph
    src_perm, dst_perm = data_perm.edge_index[0], data_perm.edge_index[1]
    perm_edges = set(zip(src_perm.tolist(), dst_perm.tolist()))

    match = orig_edges_permuted == perm_edges

    if verbose:
        status = "✓ PASS" if match else "✗ FAIL"
        print(
            f"[verify_edge_set] Edge set check {status} | "
            f"|E_orig|={len(orig_edges_permuted):,}  "
            f"|E_perm|={len(perm_edges):,}"
        )

    assert match, (
        "Edge set mismatch after permutation! "
        f"Symmetric difference has {len(orig_edges_permuted ^ perm_edges)} edges."
    )
    return True


def permutation_from_ordering(ordering: np.ndarray, n: int) -> torch.Tensor:
    """
    Convert a node ordering (array of node indices in desired order) to a
    permutation tensor where perm[old_idx] = new_idx.

    `ordering` is the output of algorithms like RCM / METIS labelling,
    which return the list of original node indices in the desired new order.
    i.e. ordering[0] = the original node that should become node 0.
    """
    ordering_t = torch.from_numpy(np.asarray(ordering, dtype=np.int64))
    perm = torch.empty(n, dtype=torch.long)
    perm[ordering_t] = torch.arange(n, dtype=torch.long)
    return perm
