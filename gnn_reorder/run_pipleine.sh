#!/bin/bash
set -e

# Setup

pip install -r requirements.txt
python data/download_datasets.py

# Phase 1
python run_phase1.py --dataset ogbn-arxiv --models GraphSAGE GAT --epochs 50 --warmup 10 --gpu 3
python run_phase1.py --dataset ogbn-products --models GraphSAGE --epochs 25 --warmup 5 --gpu 3

# Phase 2
python run_phase2.py --dataset ogbn-arxiv --method rcm --models GraphSAGE GAT --epochs 50 --warmup 10 --gpu 3
python run_phase2.py --dataset ogbn-arxiv --method metis --k 4 8 16 32 --models GraphSAGE GAT --epochs 50 --warmup 10 --gpu 3

# Phase 3
python run_phase3.py --dataset ogbn-arxiv --models GraphSAGE GAT --epochs 50 --warmup 10 --sweep-k 4 8 16 32 --alpha 1.0 1.25 1.5 --no-baseline --no-rcm --gpu 3
python run_phase3.py --dataset ogbn-products --models GraphSAGE --epochs 25 --warmup 5 --sweep-k 4 8 16 32 --alpha 1.0 1.25 1.5 --gpu 3

# Final output
python compare_results.py