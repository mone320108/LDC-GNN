#!/usr/bin/env python3
# evaluate.py
# Minimal helper showing how to call evaluate_full from utils_data
import argparse, torch
from utils_data import prepare_data, evaluate_full
from models import MDGRec

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    args=p.parse_args()
    args.device = torch.device(args.device)
    data = prepare_data(args.data, device=args.device)
    model = MDGRec(int(data['num_users']), int(data['num_items']), emb_dim=64, n_layers=2, text_dim=data['text_feats'].shape[1], K=4)
    model.to(args.device)
    try:
        model.load_state_dict(torch.load("mdgrec_best.pt", map_location=args.device))
        print("Loaded mdgrec_best.pt")
    except Exception:
        print("No checkpoint found; will run propagate on untrained weights (for debugging).")
    if hasattr(model, 'propagate'):
        model.propagate(data['adj_norm'].to(args.device), data['text_feats'].to(args.device), data.get('tail_mask', None))
    metrics = evaluate_full(model, data, device=args.device, topk=(10,20))
    print("Evaluation:", metrics)
