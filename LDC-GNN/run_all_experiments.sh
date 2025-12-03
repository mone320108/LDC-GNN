#!/usr/bin/env bash
# run_all_experiments.sh - convenience script. Edit parameters for full runs.
DATA=ml1m_mdgrec.pt
MLDIR=data/ml-1m
DEVICE=cpu
# Preprocess (compute SBERT embeddings)
python preprocess_ml1m.py --ml_dir $MLDIR --out $DATA
# Quick single-epoch smoke training (edit epochs for full runs)
# python train_full.py --data $DATA --device $DEVICE --epochs 100 --batch 2048
python train_full.py --data $DATA --epochs 50 --batch 4096 --lr 1e-3 --wd 1e-4 --dim 64 --layers 3 --K 4 --cl_weight 0.1 --device cpu

echo "Finished quick smoke test. Edit run_all_experiments.sh to run full experiments (increase epochs)."
