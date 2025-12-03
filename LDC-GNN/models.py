#!/usr/bin/env python3
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MF_BPR(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.u_emb = nn.Embedding(num_users, emb_dim)
        self.i_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        nn.init.xavier_uniform_(self.i_emb.weight)

    def encode_users(self, u_ids):
        return self.u_emb(u_ids)

    def encode_items(self, i_ids):
        return self.i_emb(i_ids)

    def forward_triplet(self, u, pos, neg):
        u_e = self.u_emb(u)
        p_e = self.i_emb(pos)
        n_e = self.i_emb(neg)
        pos_scores = (u_e * p_e).sum(dim=1)
        neg_scores = (u_e * n_e).sum(dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=2):
        super().__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.n_layers = n_layers
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.cached = None

    def propagate(self, adj_norm, text_feats=None, tail_mask=None):
        emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [emb]
        h = emb
        for _ in range(self.n_layers):
            h = torch.sparse.mm(adj_norm, h)
            embs.append(h)
        out = torch.mean(torch.stack(embs, 0), 0)
        self.cached = out
        return out

    def encode_users(self, u_ids):
        return self.cached[u_ids]

    def encode_items(self, i_ids):
        return self.cached[self.n_users + i_ids]

    def forward_triplet(self, u, pos, neg):
        ue = self.encode_users(u)
        pe = self.encode_items(pos)
        ne = self.encode_items(neg)
        pos_scores = (ue * pe).sum(dim=1)
        neg_scores = (ue * ne).sum(dim=1)
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


class MDGRec(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=2,
                 text_dim=384, K=4, cl_temp=0.2):
        super().__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.n_nodes = num_users + num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.K = K
        self.d_k = max(1, emb_dim // K)
        self.cl_temp = cl_temp

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.text_proj = nn.Linear(text_dim, emb_dim)

        self.id_projs = nn.ModuleList([nn.Linear(emb_dim, self.d_k) for _ in range(K)])
        self.text_projs = nn.ModuleList([nn.Linear(emb_dim, self.d_k) for _ in range(K)])

        self.fusion = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.Sigmoid()
        )

        self.tail_amp = nn.Parameter(torch.tensor(1.0))

        self.cached_id = None
        self.cached_text = None
        self.cached_fused = None

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    @torch.no_grad()
    def propagate(self, adj_norm, text_feats, tail_mask=None):
        emb_id = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        emb_text = self.text_proj(text_feats)

        id_list = [emb_id]
        text_list = [emb_text]
        h_id = emb_id
        h_text = emb_text

        for _ in range(self.n_layers):
            h_id = torch.sparse.mm(adj_norm, h_id)
            h_text = torch.sparse.mm(adj_norm, h_text)
            id_list.append(h_id)
            text_list.append(h_text)

        id_final = torch.mean(torch.stack(id_list, 0), 0)
        text_final = torch.mean(torch.stack(text_list, 0), 0)

        if tail_mask is not None:
            amp = 1.0 + torch.sigmoid(self.tail_amp)
            text_final[tail_mask] = text_final[tail_mask] * amp

        gate = self.fusion(torch.cat([id_final, text_final], dim=1))
        fused = gate * id_final + (1 - gate) * text_final

        self.cached_id = id_final
        self.cached_text = text_final
        self.cached_fused = fused
        return fused

    def encode_users(self, u_ids):
        return self.cached_fused[u_ids]

    def encode_items(self, i_ids):
        return self.cached_fused[self.n_users + i_ids]

    def forward_triplet(self, u, pos, neg):
        ue = self.encode_users(u)
        pe = self.encode_items(pos)
        ne = self.encode_items(neg)
        pos_scores = (ue * pe).sum(dim=1)
        neg_scores = (ue * ne).sum(dim=1)
        rec_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return rec_loss

    def compute_cl(self, node_ids):
        id_emb = self.cached_id[node_ids]
        txt_emb = self.cached_text[node_ids]
        loss = 0.0

        for k in range(self.K):
            id_k = F.normalize(self.id_projs[k](id_emb), dim=1)
            txt_k = F.normalize(self.text_projs[k](txt_emb), dim=1)
            sim = (id_k @ txt_k.T) / self.cl_temp
            labels = torch.arange(sim.size(0), device=sim.device)
            loss += (
                F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.T, labels)
            ) / 2

        return loss / self.K
