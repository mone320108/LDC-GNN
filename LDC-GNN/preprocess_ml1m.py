#!/usr/bin/env python3
# preprocess_ml1m.py
# Preprocess MovieLens-1M and compute SBERT item embeddings.
# Output: a single .pt file containing adj_norm, text_feats, pairs, train/val/test splits, etc.

import os, argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict


def load_ml1m(ml_dir):
    ratings = pd.read_csv(
        os.path.join(ml_dir, "ratings.dat"),
        sep="::", engine="python", encoding="latin-1",
        names=["user", "item", "rating", "ts"]
    )
    movies = pd.read_csv(
        os.path.join(ml_dir, "movies.dat"),
        sep="::", engine="python", encoding="latin-1",
        names=["item", "title", "genres"]
    )
    return ratings, movies


def remap_ids(ratings):
    uids = sorted(ratings.user.unique().tolist())
    iids = sorted(ratings.item.unique().tolist())
    u2id = {u: i for i, u in enumerate(uids)}
    i2id = {it: i for i, it in enumerate(iids)}
    ratings['uid'] = ratings.user.map(u2id)
    ratings['iid'] = ratings.item.map(i2id)
    return ratings, u2id, i2id


def build_adj_norm(num_users, num_items, interactions):
    N = num_users + num_items
    rows = []
    cols = []

    for (u, i) in interactions:
        rows.extend([u, num_users + i])
        cols.extend([num_users + i, u])

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)

    deg = np.zeros(N, dtype=np.float32)
    for r in rows:
        deg[r] += 1
    deg = np.power(deg, -0.5)
    deg[np.isinf(deg)] = 0.0

    vals = deg[rows] * deg[cols]
    idx = torch.LongTensor(np.vstack([rows, cols]))
    vals = torch.FloatTensor(vals)
    shape = torch.Size([N, N])

    adj_norm = torch.sparse.FloatTensor(idx, vals, shape)
    return adj_norm


def compute_item_text_embeddings(movies, i2id, sbert_name="all-MiniLM-L6-v2"):
    print("Loading SBERT model:", sbert_name)
    model = SentenceTransformer(sbert_name)

    texts = []
    for item, idx in sorted(i2id.items(), key=lambda x: x[1]):
        row = movies[movies.item == item].iloc[0]
        txt = f"{row.title}. Genres: {row.genres.replace('|', ', ')}"
        texts.append(txt)

    print("Encoding item texts with SBERT ...")
    emb = model.encode(texts, show_progress_bar=True)
    emb = torch.tensor(emb, dtype=torch.float32)
    return emb


def build_pairs(train_list, num_users, num_items):
    grouped = defaultdict(list)
    for u, i in train_list:
        grouped[u].append(i)

    pairs = []
    for u, pos_list in grouped.items():
        pos_set = set(pos_list)
        for pos in pos_list:
            neg = np.random.randint(0, num_items)
            while neg in pos_set:
                neg = np.random.randint(0, num_items)
            pairs.append((int(u), int(pos), int(neg)))

    return pairs


def compute_item_counts(train_list, num_items):
    counts = defaultdict(int)
    for u, i in train_list:
        counts[i] += 1
    for i in range(num_items):
        counts.setdefault(i, 0)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sbert", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    ratings, movies = load_ml1m(args.ml_dir)
    ratings, u2id, i2id = remap_ids(ratings)

    num_users = len(u2id)
    num_items = len(i2id)
    print("num_users", num_users, "num_items", num_items)

    interactions = [(int(r.uid), int(r.iid)) for _, r in ratings.iterrows()]
    adj_norm = build_adj_norm(num_users, num_items, interactions)

    item_text_emb = compute_item_text_embeddings(movies, i2id, sbert_name=args.sbert)

    # user text = average of interacted item embeddings
    user_text = torch.zeros((num_users, item_text_emb.size(1)), dtype=torch.float32)
    counts = torch.zeros(num_users, dtype=torch.float32)
    for u, i in interactions:
        user_text[int(u)] += item_text_emb[int(i)]
        counts[int(u)] += 1
    counts[counts == 0] = 1
    user_text = user_text / counts.unsqueeze(1)

    text_feats = torch.zeros((num_users + num_items, item_text_emb.size(1)), dtype=torch.float32)
    text_feats[:num_users] = user_text
    text_feats[num_users:] = item_text_emb

    # split
    user_items = defaultdict(list)
    for u, i in interactions:
        user_items[u].append(i)

    train_list = []
    val_list = []
    test_list = []

    for u, items in user_items.items():
        items = sorted(list(dict.fromkeys(items)))
        n = len(items)
        if n == 1:
            train_list.append((u, items[0]))
            continue

        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))

        train_list += [(u, items[j]) for j in range(0, n_train)]
        val_list += [(u, items[j]) for j in range(n_train, n_train + n_val)]
        test_list += [(u, items[j]) for j in range(n_train + n_val, n)]

    pairs = build_pairs(train_list, num_users, num_items)
    item_counts = compute_item_counts(train_list, num_items)

    tail_items = [i for i, c in item_counts.items() if c < 5]
    tail_mask = torch.zeros(num_users + num_items, dtype=torch.bool)
    tail_indices = [num_users + i for i in tail_items]

    if len(tail_indices) > 0:
        tail_mask[torch.tensor(tail_indices)] = True

    test_dict = defaultdict(set)
    for u, i in test_list:
        test_dict[u].add(i)

    out = {
        'adj_norm': adj_norm,
        'text_feats': text_feats,
        'pairs': pairs,
        'train_interactions': train_list,
        'val_interactions': val_list,
        'test_interactions': test_list,
        'test_dict': dict(test_dict),
        'num_users': num_users,
        'num_items': num_items,
        'tail_mask': tail_mask,
        'item_counts': item_counts
    }

    torch.save(out, args.out)
    print("Saved preprocessed data to", args.out)
