#!/usr/bin/env python3
# utils_data.py
import torch, math
from collections import defaultdict

def safe_torch_load(path):
    try:
        return torch.load(path)
    except Exception:
        return torch.load(path, weights_only=False)

def prepare_data(pt_path, device=None):
    data = safe_torch_load(pt_path)
    if device is not None:
        if isinstance(data.get('adj_norm'), torch.Tensor):
            data['adj_norm'] = data['adj_norm'].to(device)
        if isinstance(data.get('text_feats'), torch.Tensor):
            data['text_feats'] = data['text_feats'].to(device)
        if 'tail_mask' in data and isinstance(data['tail_mask'], torch.Tensor):
            data['tail_mask'] = data['tail_mask'].to(device)
    return data

def recall_at_k(pred, true_set, k):
    pred = pred[:k]
    if len(true_set) == 0: return 0.0
    return len(set(pred) & true_set) / len(true_set)

def ndcg_at_k(pred, true_set, k):
    dcg = 0.0
    for i,p in enumerate(pred[:k]):
        if p in true_set:
            dcg += 1.0 / math.log2(i+2)
    idcg = sum(1.0 / math.log2(i+2) for i in range(min(len(true_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_full(model, data, device='cpu', topk=(10,20), users=None):
    model.eval()
    num_users = int(data['num_users']); num_items = int(data['num_items'])
    test_dict = data['test_dict']
    train_list = data['train_interactions']
    train_mask = defaultdict(set)
    for u,i in train_list:
        train_mask[u].add(i)
    if users is None:
        users = list(test_dict.keys())
    results = {f"Recall@{k}":[] for k in topk}
    results.update({f"NDCG@{k}":[] for k in topk})
    results.update({f"HitRate@{k}":[] for k in topk})
    with torch.no_grad():
        item_ids = torch.arange(num_items, device=device)
        item_emb = model.encode_items(item_ids.to(device))
        for u in users:
            u_emb = model.encode_users(torch.tensor([u], device=device))
            scores = (u_emb @ item_emb.T).squeeze(0)
            tr = train_mask.get(u, set())
            if len(tr) > 0:
                scores[list(tr)] = -1e9
            maxk = max(topk)
            _, top_idx = torch.topk(scores, maxk)
            top_idx = top_idx.cpu().numpy().tolist()
            true_set = set(test_dict.get(u, []))
            if len(true_set) == 0: continue
            for k in topk:
                results[f"Recall@{k}"].append(recall_at_k(top_idx, true_set, k))
                results[f"NDCG@{k}"].append(ndcg_at_k(top_idx, true_set, k))
                results[f"HitRate@{k}"].append(1.0 if len(set(top_idx[:k]) & true_set) > 0 else 0.0)
    out = {k:(sum(v)/len(v) if len(v)>0 else 0.0) for k,v in results.items()}
    return out
