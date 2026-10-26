#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
GPU="${1:-${GPU:-0}}"
PYTHON="${PYTHON:-python3}"

# Setup

"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" data/download_datasets.py

# Phase 1
"$PYTHON" run_phase1.py --dataset ogbn-arxiv --models GraphSAGE GAT --epochs 50 --warmup 10 --gpu "$GPU"
"$PYTHON" run_phase1.py --dataset ogbn-products --models GraphSAGE --epochs 25 --warmup 5 --gpu "$GPU"

# Phase 2
"$PYTHON" run_phase2.py --dataset ogbn-arxiv --method rcm --models GraphSAGE GAT --epochs 50 --warmup 10 --gpu "$GPU"
"$PYTHON" run_phase2.py --dataset ogbn-arxiv --method metis --k 4 8 16 32 --models GraphSAGE GAT --epochs 50 --warmup 10 --gpu "$GPU"

# Phase 3
"$PYTHON" run_phase3.py --dataset ogbn-arxiv --models GraphSAGE GAT --epochs 50 --warmup 10 --sweep-k 4 8 16 32 --alpha 1.0 1.25 1.5 --no-baseline --no-rcm --gpu "$GPU"
"$PYTHON" run_phase3.py --dataset ogbn-products --models GraphSAGE --epochs 25 --warmup 5 --sweep-k 4 8 16 32 --alpha 1.0 1.25 1.5 --gpu "$GPU"

# Final output
"$PYTHON" compare_results.py
