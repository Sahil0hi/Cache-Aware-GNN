# Cache-Aware Graph Reordering for GNN Training

This directory contains the full implementation for the project:
**"Cache-Aware Graph Reordering for Efficient GNN Execution on GPUs"**

## Directory Layout

```
gnn_reorder/
├── requirements.txt              # Python dependencies
├── data/
│   └── download_datasets.py      # Downloads ogbn-arxiv and ogbn-products via OGB
├── models/
│   ├── graphsage.py              # 2-layer GraphSAGE model
│   └── gat.py                    # 2-layer GAT model
├── reordering/
│   ├── apply_permutation.py      # Applies P to A, H, y and verifies A' = PAP^T
│   ├── rcm.py                    # RCM-only reordering via scipy
│   ├── metis_reorder.py          # METIS-only reordering via pymetis (swept k)
│   └── hybrid_reorder.py         # Two-level hybrid: METIS → intra-partition RCM
│                                 # + cache_aware_k_star() formula
├── profiling/
│   ├── timer.py                  # Epoch timer with CSV logging
│   ├── cache_proxy.py            # TRR and ACC software cache-efficiency metrics
│   └── sparsity_plot.py          # Adjacency density heatmap visualisation
├── results/                      # CSV and figure output
│   ├── timing.csv                # Phase 1 & 2 timing rows
│   ├── phase3_results.csv        # Phase 3 full ablation (timing + TRR + ACC)
│   └── figures/                  # Sparsity plots and Pareto scatter plot
├── run_phase1.py                 # Phase 1: baseline training & profiling
├── run_phase2.py                 # Phase 2: RCM + METIS reordering
├── run_phase3.py                 # Phase 3: cache-aware k* + hybrid reordering
└── compare_results.py            # Merge all CSVs → ablation table + figures
```

## Setup

```bash
# 1. Install dependencies (GPU machine with CUDA)
cd gnn_reorder  # from the repository root
python -m pip install -r requirements.txt

# 2. Download datasets (optional — done automatically by the run scripts)
python data/download_datasets.py
```

## Running the Experiments

### Phase 1 — Baseline training

```bash
# Full-batch on ogbn-arxiv (GraphSAGE + GAT)
python run_phase1.py --dataset ogbn-arxiv --epochs 50 --warmup 10

# Mini-batch on ogbn-products (GraphSAGE only)
python run_phase1.py --dataset ogbn-products --epochs 10 --warmup 3
```

### Phase 2 — Baseline reordering (RCM and METIS sweep)

```bash
# RCM reordering
python run_phase2.py --dataset ogbn-arxiv --method rcm

# METIS with a single k
python run_phase2.py --dataset ogbn-arxiv --method metis --k 8

# METIS with a sweep over several k values
python run_phase2.py --dataset ogbn-arxiv --method metis --k 4 8 16 32
```

### Phase 3 — Novel contributions

Phase 3 implements and evaluates the two project contributions:

1. **Cache-size-aware `k*`** — `k* = ⌈α · n·d·b / C⌉`
2. **Two-level hybrid reordering** — `METIS(k*)` then intra-partition RCM

```bash
# Quick run: GraphSAGE, arxiv, default alpha=[1.0, 1.25]
python run_phase3.py --dataset ogbn-arxiv --models GraphSAGE --epochs 20

# Full ablation: both models, METIS sweep, alpha calibration
python run_phase3.py --dataset ogbn-arxiv \
    --models GraphSAGE GAT \
    --epochs 50 --warmup 10 \
    --sweep-k 4 8 16 32 \
    --alpha 1.0 1.25 1.5

# Cache metrics only (no training, very fast)
python run_phase3.py --dataset ogbn-arxiv --no-train

# Skip baseline/RCM (only novel configs)
python run_phase3.py --dataset ogbn-arxiv --no-baseline --no-rcm
```

Phase 3 outputs:
- `results/phase3_results.csv` — timing, TRR, ACC for all configs
- `results/figures/` — adjacency density before/after + Pareto scatter plot

### Combined ablation table

```bash
python compare_results.py
```

Reads both `timing.csv` (Phase 1/2) and `phase3_results.csv` (Phase 3),
prints the merged ablation table, and saves:
- `results/timing_comparison.png` — bar chart (ms/epoch per config)
- `results/figures/pareto.png`    — TRR vs ms/epoch Pareto frontier

## Cache-Efficiency Metrics

Since hardware L2 counters (`ncu`) require admin-level access on shared
servers, two software proxy metrics are used:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **ACC** (Analytical Cache Coverage) | `min(1, C / (n·d·b))` | Fraction of feature matrix that fits in the assumed L2 capacity |
| **TRR** (Temporal Reuse Ratio) | `simulated_LRU_hits / total_fetches` | Fraction of neighbor fetches hitting a finite software LRU cache |

## Hardware Profiling with ncu (if admin access available)

```bash
ncu --metrics l2tex__t_sector_hit_rate.pct \
    --target-processes all \
    -o results/baseline_arxiv \
    python run_phase1.py --dataset ogbn-arxiv --epochs 1
```

## Novel Method: Hybrid Reordering

```python
from reordering.hybrid_reorder import hybrid_reorder, cache_aware_k_star

# Derive k* from hardware
k_star = cache_aware_k_star(n_nodes=169343, feat_dim=128, alpha=1.0)
# k_star = 3 for ogbn-arxiv on an A100 (40 MiB L2)

# Apply two-level permutation
data_reordered, perm = hybrid_reorder(data, k=k_star, verify=True)
```

## Saved results and interpretation

The committed CSVs are the recorded experiment outputs. Regenerate their charts
without downloading datasets or running GPU training with:

```bash
python compare_results.py
```

Run this command inside `gnn_reorder/`. It requires only Matplotlib and NumPy.
Training additionally requires PyTorch, PyG, OGB, SciPy, and METIS. For
`ogbn-products`, PyG's `NeighborLoader` also requires a compatible `pyg-lib` or
`torch-sparse` installation for your PyTorch/CUDA build.

TRR is a software simulation, not a measured hardware L2 hit rate. ACC depends
only on the feature matrix size and assumed 40 MiB L2 capacity, so it does not
change under permutation. The recorded TRR implementation randomly subsamples
training nodes when there are more than 50,000; its access order therefore does
not directly model the complete GPU aggregation schedule. Treat these proxies
as exploratory evidence alongside the recorded epoch times.

The timing CSV uses one shared column schema across phases. Fields not measured
in a phase are blank. The saved Phase 2 rows were normalized from their original
shorter schema; timing and accuracy values were preserved. Re-running experiments
appends rows, and the comparison script prefers Phase 3 configurations when both
CSVs contain the same dataset/model/reordering.

For the complete sequence, use `bash run_pipeline.sh` (GPU 0),
`bash run_pipeline.sh 1`, or `GPU=1 bash run_pipeline.sh`. Set `PYTHON` to select
a Python interpreter. The pipeline installs requirements, downloads both
datasets, and runs all phases; it is intended for a CUDA machine and can take
substantial time.
