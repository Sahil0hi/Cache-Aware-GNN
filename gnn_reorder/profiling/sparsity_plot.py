"""
profiling/sparsity_plot.py
==========================
Visualise the non-zero pattern of the graph adjacency matrix before and
after reordering.

For graphs with millions of nodes a full spy-plot is impractical.  Instead
we render a downsampled density heatmap: the node index range [0, n) is
divided into `grid_size` equal-width bins and the plot shows the log-scaled
edge density in each (row-bin, col-bin) cell.

Public functions
----------------
plot_adjacency_density(data, title, out_path, grid_size=500)
    Save a single density heatmap for `data`.

plot_before_after(data_before, data_after, label_before, label_after, out_dir, stem)
    Save a side-by-side figure comparing two orderings.
"""

import os

import numpy as np

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from torch_geometric.data import Data


def plot_adjacency_density(
    data: Data,
    title: str,
    out_path: str,
    grid_size: int = 500,
) -> None:
    """
    Save a downsampled adjacency density heatmap.

    The adjacency matrix is binned into a `grid_size` × `grid_size` grid.
    Each cell value is log(1 + count) where count is the number of edges
    whose (src, dst) falls in that cell.

    Args:
        data      : PyG Data object whose edge_index is used.
        title     : Figure title string.
        out_path  : Absolute path for the saved PNG file.
        grid_size : Resolution of the density grid (default 500).
    """
    if not MPL_AVAILABLE:
        print("[sparsity_plot] matplotlib not available; skipping plot.")
        return

    n = data.num_nodes
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()

    # Bin node indices into [0, grid_size)
    bin_src = (src.astype(np.int64) * grid_size // n).clip(0, grid_size - 1)
    bin_dst = (dst.astype(np.int64) * grid_size // n).clip(0, grid_size - 1)

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(grid, (bin_src, bin_dst), 1.0)

    # Log scale for legibility across sparse / dense regions
    grid = np.log1p(grid)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="hot_r", interpolation="nearest", aspect="auto",
                   origin="upper")
    plt.colorbar(im, ax=ax, label="log(1 + edge count)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Column node index (binned)")
    ax.set_ylabel("Row node index (binned)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[sparsity_plot] Saved → {out_path}")


def plot_before_after(
    data_before: Data,
    data_after: Data,
    label_before: str,
    label_after: str,
    out_dir: str,
    stem: str,
    grid_size: int = 500,
) -> None:
    """
    Save a side-by-side adjacency density comparison figure.

    Args:
        data_before  : Original graph.
        data_after   : Reordered graph.
        label_before : Title suffix for the left panel (e.g. "baseline").
        label_after  : Title suffix for the right panel (e.g. "hybrid k=3").
        out_dir      : Directory for the output PNG.
        stem         : Base filename (without extension).
        grid_size    : Resolution of density grid.
    """
    if not MPL_AVAILABLE:
        print("[sparsity_plot] matplotlib not available; skipping plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    n = data_before.num_nodes

    def _density_grid(data):
        src = data.edge_index[0].numpy()
        dst = data.edge_index[1].numpy()
        bin_src = (src.astype(np.int64) * grid_size // n).clip(0, grid_size - 1)
        bin_dst = (dst.astype(np.int64) * grid_size // n).clip(0, grid_size - 1)
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        np.add.at(grid, (bin_src, bin_dst), 1.0)
        return np.log1p(grid)

    grid_before = _density_grid(data_before)
    grid_after = _density_grid(data_after)

    # Shared colour scale
    vmax = max(grid_before.max(), grid_after.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, grid, label in zip(
        axes, [grid_before, grid_after], [label_before, label_after]
    ):
        im = ax.imshow(grid, cmap="hot_r", interpolation="nearest",
                       aspect="auto", origin="upper", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="log(1 + edge count)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Column (binned)")
        ax.set_ylabel("Row (binned)")

    fig.suptitle(f"Adjacency density  |  n={n:,}  |  grid={grid_size}×{grid_size}",
                 fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[sparsity_plot] Saved → {out_path}")
