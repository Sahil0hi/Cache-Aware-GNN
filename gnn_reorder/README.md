# Cache-Aware Graph Reordering for GNN Training

This directory contains the full implementation for the project:
**"Cache-Aware Graph Reordering for Efficient GNN Execution on GPUs"**

## Directory Layout

```
gnn_reorder/
├── requirements.txt          # Python dependencies
├── data/
│   └── download_datasets.py  # Downloads ogbn-arxiv and ogbn-products via OGB
├── models/
│   ├── graphsage.py          # 2-layer GraphSAGE model
│   └── gat.py                # 2-layer GAT model
├── reordering/
│   ├── apply_permutation.py  # Applies P to A, H, y and verifies correctness
│   ├── rcm.py                # RCM-only reordering via scipy
│   └── metis_reorder.py      # METIS-only reordering via pymetis (swept k)
├── profiling/
│   └── timer.py              # Epoch timer + optional ncu wrapper
├── results/                  # CSV/JSON output lands here
├── run_phase1.py             # Phase 1: download data + train & profile baseline
└── run_phase2.py             # Phase 2: RCM + METIS reordering, correctness check
```

## Setup

```bash
# 1. Install dependencies (GPU machine with CUDA)
pip install -r requirements.txt

# 2. (Phase 1) Download datasets and run baseline training
python run_phase1.py --dataset ogbn-arxiv

# 3. (Phase 2) Run reordering methods and validate
python run_phase2.py --dataset ogbn-arxiv --method rcm
python run_phase2.py --dataset ogbn-arxiv --method metis --k 8
```

## Hardware Profiling with ncu

To capture L2 cache hit rate, wrap the training script with Nsight Compute:

```bash
ncu --metrics l2tex__t_sector_hit_rate.pct \
    --target-processes all \
    -o results/baseline_arxiv \
    python run_phase1.py --dataset ogbn-arxiv --epochs 1
```

## Results

All timing and metric output is saved to `results/` as CSV files that can
be loaded directly into the ablation table in the final report.
