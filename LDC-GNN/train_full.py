\
#!/usr/bin/env python3
# train_full.py
import argparse, time, torch
from utils_data import prepare_data, evaluate_full
from models import MDGRec
from torch.utils.data import DataLoader, Dataset
import torch, collections
torch.serialization.add_safe_globals([collections.defaultdict])


class TripletDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        u,p,n = self.pairs[idx]
        return torch.LongTensor([u]).squeeze(0), torch.LongTensor([p]).squeeze(0), torch.LongTensor([n]).squeeze(0)

def train(args):
    # ------------------------------------------------------------
    # 正确方式：只用 preprocess_ml1m.py 生成的数据
    # ------------------------------------------------------------
    device = args.device
    data = torch.load(args.data, map_location=device, weights_only=False)

    num_users  = int(data['num_users'])
    num_items  = int(data['num_items'])
    adj        = data['adj_norm'].to(device)
    text_feats = data['text_feats'].to(device)      # (num_items, 384)
    tail_mask  = data.get('tail_mask', None)
    if tail_mask is not None:
        tail_mask = tail_mask.to(device)

    pairs      = data['pairs']

    # ------------------------------------------------------------
    # 初始化模型
    # ------------------------------------------------------------
    model = MDGRec(
        num_users, 
        num_items,
        emb_dim=args.dim,
        n_layers=args.layers,
        text_dim=text_feats.shape[1],
        K=args.K
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dl = DataLoader(TripletDS(pairs), batch_size=args.batch, shuffle=True)

    best_val = -1.0

    # ------------------------------------------------------------
    # ---- training loop (修正后) ----
    model.train()
    total = 0.0; n = 0
    for u, p, nn in dl:
        u = u.to(device); p = p.to(device); nn = nn.to(device)

        # 1) 重新做一次 propagate（不要用 no_grad）
        #    这会把 final embeddings 的计算置于计算图，使得 backward 能回传到 user_emb/item_emb/text_proj 等
        model.propagate(adj, text_feats, tail_mask)

        # 2) 计算损失并更新
        rec_loss = model.forward_triplet(u, p, nn)

        cl_loss = 0.0
        if hasattr(model, 'compute_cl'):
            num_nodes = model.n_users + model.n_items
            sample_size = min(512, num_nodes)
            node_ids = torch.randperm(num_nodes, device=device)[:sample_size]
            cl_loss = model.compute_cl(node_ids)
        loss = rec_loss + args.cl_weight * cl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    total += loss.item(); n += 1

    # ------------------------------------------------------------
    # Final test
    # ------------------------------------------------------------
    print("Loading best checkpoint...")
    model.load_state_dict(torch.load("mdgrec_best.pt", map_location=device))
    test_metrics = evaluate_full(model, data, device=device, topk=(10,20))
    print("Test metrics:", test_metrics)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--cl_weight", type=float, default=0.1)
    args=p.parse_args()
    args.device = torch.device(args.device)
    train(args)
