LDC-GNN: MDGRec experiment pipeline (SBERT + Modal Disentanglement GNN)

Quick start:
1. Ensure MovieLens-1M is under `data/ml-1m` with files ratings.dat and movies.dat.
2. Create a conda env with pytorch + sentence-transformers (or use your torch312 env).
3. Preprocess and compute SBERT item embeddings:
   python preprocess_ml1m.py --ml_dir data/ml-1m --out ml1m_mdgrec.pt
4. Run a quick train+eval (smoke test):
   bash run_all_experiments.sh

Files:
- preprocess_ml1m.py   : preprocess ML-1M, compute SBERT embeddings, train/val/test split
- utils_data.py        : helpers for loading and full-ranking evaluation (no data leakage)
- models.py            : MF-BPR, LightGCN, MDGRec model implementations
- train_full.py        : train MDGRec on train split, validate on val, final eval on test
- evaluate.py          : example script to run evaluate_full
- run_all_experiments.sh: convenience script to run preprocess + one epoch training (edit for full runs)
