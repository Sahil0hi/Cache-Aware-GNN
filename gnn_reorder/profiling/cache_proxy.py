"""
profiling/cache_proxy.py
========================
Software-based cache-efficiency metrics for when hardware counters
(ncu l2tex__t_sector_hit_rate.pct) are unavailable due to admin restrictions.

Two complementary metrics are computed:

1. Temporal Reuse Ratio (TRR)
   ---------------------------
   During one SpMM pass, how often is a neighbor feature vector accessed
   MORE than once across all training rows?

       TRR = (total_accesses - unique_accesses) / total_accesses

   A high TRR means many accesses can be served from cache rather than HBM.
   With random ordering TRR ≈ avg_degree / n_nodes (very low).
   After reordering, nearby rows share neighbors → TRR increases.

2. Analytical Cache Coverage (ACC)
   ---------------------------------
   What fraction of the feature matrix fits in the GPU L2 cache?

       ACC = min(1.0, C / (n * d * b))

   Where C = L2 capacity (bytes), n = nodes, d = feature dim, b = bytes/element.
   ACC is the theoretical upper bound on hit rate assuming perfect spatial locality.
   With random ordering the effective hit rate ≈ ACC × (bandwidth/n),
   which is much smaller.

Both metrics require only the graph's edge_index and feature dimensions —
no GPU hardware access needed.
"""

import torch
import numpy as np
from collections import OrderedDict
from torch_geometric.data import Data

# A100 GPU L2 cache capacity in bytes (40 MB)
A100_L2_BYTES = 40 * 1024 * 1024


def temporal_reuse_ratio(
    edge_index: torch.Tensor,
    train_mask: torch.Tensor,
    sample_size: int = 100_000,
    feat_dim: int = 128,
    bytes_per_element: int = 4,
    l2_bytes: int = A100_L2_BYTES,
) -> dict:
    """
    Estimate temporal reuse ratio for one SpMM pass over training nodes,
    using a finite LRU cache model sized to the GPU L2 capacity.

    For each training node (row), the SpMM fetches feature vectors for all
    its neighbors (column indices). TRR measures how many of those fetches
    hit a node whose feature vector is still resident in the simulated L2.

    The cache capacity in nodes is:
        cache_nodes = L2_bytes / (feat_dim * bytes_per_element)

    An infinite `seen` set would overestimate TRR (every node eventually
    stays "seen" forever), making TRR insensitive to reordering. The finite
    LRU window makes TRR correctly sensitive to locality improvements.

    Args:
        edge_index         : [2, E] edge tensor (src, dst)
        train_mask         : boolean mask or index tensor of training nodes
        sample_size        : max training nodes to evaluate (for speed)
        feat_dim           : feature dimension (used to size the cache window)
        bytes_per_element  : bytes per feature element (4 for float32)
        l2_bytes           : GPU L2 cache capacity in bytes (default: A100 40 MB)

    Returns dict with:
        total_accesses   - total neighbor fetches counted
        unique_accesses  - number of distinct nodes fetched (global)
        trr              - temporal reuse ratio (0=no reuse, 1=perfect reuse)
        mean_degree      - average degree of training nodes
        cache_capacity   - number of node feature vectors that fit in L2
    """
    src, dst = edge_index[0], edge_index[1]

    # Get training node indices
    if train_mask.dtype == torch.bool:
        train_nodes = train_mask.nonzero(as_tuple=True)[0]
    else:
        train_nodes = train_mask

    # Sample for speed
    if train_nodes.numel() > sample_size:
        idx = torch.randperm(train_nodes.numel())[:sample_size]
        train_nodes = train_nodes[idx]

    # Build a fast neighbor lookup (COO → per-node list)
    src_np = src.numpy()
    dst_np = dst.numpy()
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in zip(src_np, dst_np):
        adj[u].append(v)

    # Finite LRU cache sized to GPU L2 capacity.
    # Using an OrderedDict as an O(1) LRU approximation:
    # most-recently-used items are at the end; evict from the front.
    cache_capacity = max(1, l2_bytes // (feat_dim * bytes_per_element))
    lru_cache: OrderedDict = OrderedDict()

    total_accesses = 0
    total_reused   = 0
    all_seen: set  = set()   # for unique_accesses count only

    train_list = train_nodes.numpy().tolist()
    degrees = []
    for node in train_list:
        neighbors = adj[node]
        degrees.append(len(neighbors))
        for nb in neighbors:
            total_accesses += 1
            all_seen.add(nb)
            if nb in lru_cache:
                # Cache hit — move to most-recently-used end
                lru_cache.move_to_end(nb)
                total_reused += 1
            else:
                # Cache miss — insert; evict LRU entry if over capacity
                lru_cache[nb] = True
                if len(lru_cache) > cache_capacity:
                    lru_cache.popitem(last=False)

    trr = total_reused / total_accesses if total_accesses > 0 else 0.0
    mean_degree = float(np.mean(degrees)) if degrees else 0.0

    return {
        "total_accesses":        total_accesses,
        "unique_accesses":       len(all_seen),
        "temporal_reuse_ratio":  round(trr, 4),
        "mean_train_degree":     round(mean_degree, 2),
        "nodes_sampled":         len(train_list),
        "cache_capacity":        cache_capacity,
    }


def analytical_cache_coverage(
    n_nodes: int,
    feat_dim: int,
    bytes_per_element: int = 4,
    l2_bytes: int = A100_L2_BYTES,
) -> dict:
    """
    Compute the analytical cache coverage: what fraction of H^(l) fits in L2.

    Also computes the working-set size and the estimated miss rate under
    random ordering (bandwidth-based).

    Args:
        n_nodes           : number of nodes
        feat_dim          : feature dimension d_l
        bytes_per_element : 4 for float32
        l2_bytes          : GPU L2 cache capacity in bytes

    Returns dict with:
        feature_matrix_mb   - size of full feature matrix in MB
        l2_cache_mb         - L2 cache size in MB
        cache_coverage      - fraction of feature matrix that fits in L2
        estimated_miss_rate - 1 - cache_coverage (lower bound on miss rate)
        k_star              - optimal METIS k from the paper's formula
    """
    feat_bytes = n_nodes * feat_dim * bytes_per_element
    feat_mb    = feat_bytes / (1024 ** 2)
    l2_mb      = l2_bytes  / (1024 ** 2)
    coverage   = min(1.0, l2_bytes / feat_bytes)
    miss_rate  = 1.0 - coverage

    import math
    k_star = math.ceil(feat_bytes / l2_bytes)

    return {
        "feature_matrix_mb":   round(feat_mb, 1),
        "l2_cache_mb":         round(l2_mb, 1),
        "cache_coverage":      round(coverage, 4),
        "estimated_miss_rate": round(miss_rate, 4),
        "k_star":              k_star,
    }


def profile_graph(data: Data, split_idx: dict, label: str = "") -> None:
    """
    Run both metrics on a graph and print a summary table.

    Args:
        data      : PyG Data object
        split_idx : OGB split dict with 'train' key
        label     : description string (e.g. 'baseline', 'RCM', 'METIS k=8')
    """
    n = data.num_nodes
    d = data.x.shape[1] if data.x is not None else 0

    print(f"\n{'='*55}")
    print(f"  Cache metrics [{label}]  |  n={n:,}  d={d}")
    print(f"{'='*55}")

    # Analytical
    acc = analytical_cache_coverage(n, d)
    print(f"  Feature matrix size : {acc['feature_matrix_mb']} MB")
    print(f"  A100 L2 cache       : {acc['l2_cache_mb']} MB")
    print(f"  Cache coverage (ACC): {100*acc['cache_coverage']:.1f}%")
    print(f"  Est. miss rate      : {100*acc['estimated_miss_rate']:.1f}%")
    print(f"  Optimal k* (METIS)  : {acc['k_star']}")

    # Temporal reuse (sample up to 50K training nodes for speed)
    train_idx = split_idx["train"]
    print(f"\n  Computing temporal reuse over {min(50000, train_idx.numel()):,} train nodes ...")
    trr = temporal_reuse_ratio(data.edge_index, train_idx, sample_size=50_000)
    print(f"  Mean train degree   : {trr['mean_train_degree']}")
    print(f"  Total accesses      : {trr['total_accesses']:,}")
    print(f"  Unique accesses     : {trr['unique_accesses']:,}")
    print(f"  Temporal Reuse Ratio: {100*trr['temporal_reuse_ratio']:.2f}%")
    print(f"{'='*55}")

    return {**acc, **trr, "label": label}
