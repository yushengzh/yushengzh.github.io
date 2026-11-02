#!/usr/bin/env python3
"""GIN+GCN+GAT graph classifier for ogbg-molbbbp with multi-architecture ensemble."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class FlatGraphDataset:
    def __init__(self, d: dict):
        self.d = d
        self.x = d["x"]
        self.edge_index = d["edge_index"]
        self.edge_attr = d.get("edge_attr", None)
        self.node_ptr = d["node_ptr"]
        self.edge_ptr = d["edge_ptr"]
        self.y = d["y"]
        self.num_graphs = d["num_graphs"]

    def build_batch(self, indices):
        d = self.d
        xs, eis, eats, batches, ys = [], [], [], [], []
        offset = 0
        for i, gi in enumerate(indices):
            n0 = int(d["node_ptr"][gi])
            n1 = int(d["node_ptr"][gi + 1])
            e0 = int(d["edge_ptr"][gi])
            e1 = int(d["edge_ptr"][gi + 1])
            n = n1 - n0
            xs.append(self.x[n0:n1])
            eis.append(d["edge_index"][:, e0:e1] - n0 + offset)
            batches.append(torch.full((n,), i, dtype=torch.long))
            if self.edge_attr is not None:
                eats.append(self.edge_attr[e0:e1])
            ys.append(self.y[gi])
            offset += n
        x = torch.cat(xs, dim=0)
        edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
        batch_ids = torch.cat(batches) if batches else torch.zeros(0, dtype=torch.long)
        y = torch.stack(ys) if ys else torch.zeros(0)
        if self.edge_attr is not None:
            edge_attr = torch.cat(eats, dim=0) if eats else torch.zeros(0, self.edge_attr.shape[1])
        else:
            edge_attr = None
        return x, edge_index, edge_attr, batch_ids, y


class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None

    def forward(self, h, edge_index, edge_attr, batch_ids, n_graphs):
        row, col = edge_index
        if self.edge_proj is not None and edge_attr is not None and edge_attr.shape[0] > 0:
            msg = h[col] + self.edge_proj(edge_attr)
        else:
            msg = h[col]
        agg = torch.zeros_like(h).index_add_(0, row, msg)
        out = (1 + self.eps) * h + agg
        return self.mlp(out)


class GIN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=5, dropout=0.2, pooling='mean'):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([
            GINConv(hidden, hidden, edge_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        if pooling == 'meanmax':
            head_in = 2 * hidden
        elif pooling in ('mean', 'max', 'add'):
            head_in = hidden
        else:
            head_in = hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch_ids, n_graphs):
        h = self.input_proj(x)
        for conv in self.convs:
            h = self.dropout(conv(h, edge_index, edge_attr, batch_ids, n_graphs))
        if self.pooling == 'mean':
            pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h)
            counts = torch.bincount(batch_ids, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
            pooled = pooled / counts
        elif self.pooling == 'max':
            pooled = torch.full((n_graphs, h.size(1)), -1e9, device=h.device).scatter_reduce_(
                0, batch_ids.unsqueeze(1).expand_as(h), h, reduce='amax')
        elif self.pooling == 'add':
            pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h)
        else:  # 'meanmax'
            mean_p = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h)
            counts = torch.bincount(batch_ids, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
            mean_p = mean_p / counts
            max_p = torch.full((n_graphs, h.size(1)), -1e9, device=h.device).scatter_reduce_(
                0, batch_ids.unsqueeze(1).expand_as(h), h, reduce='amax')
            pooled = torch.cat([mean_p, max_p], dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits


class GCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=5, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(nn.Linear(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.edge_proj = nn.Linear(edge_dim, hidden, bias=False) if edge_dim > 0 else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch_ids, n_graphs):
        h = self.input_proj(x)
        row, col = edge_index
        for lin, bn in zip(self.convs, self.bns):
            h_lin = bn(lin(h))
            msg = h_lin[col]
            if self.edge_proj is not None and edge_attr is not None and edge_attr.shape[0] > 0:
                msg = msg + self.edge_proj(edge_attr)
            agg = torch.zeros_like(h).index_add_(0, row, msg)
            deg = torch.bincount(row, minlength=h.size(0)).clamp(min=1).float()
            deg = torch.sqrt(deg)
            h = self.dropout(bn(h_lin + agg / deg.unsqueeze(1)))
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h)
        counts = torch.bincount(batch_ids, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pooled = pooled / counts
        logits = self.head(pooled).squeeze(-1)
        return logits


class GATConv(nn.Module):
    """Single-head GAT layer with edge attributes."""
    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None
        self.attn_lin = nn.Linear(out_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr, batch_ids, n_graphs):
        row, col = edge_index
        h_proj = self.lin(h)  # [N, out_dim]
        # Compute attention scores for each edge
        # For each edge (col -> row), score = a^T * [h_proj[col] || h_proj[row]]
        src = h_proj[col]  # [E, out_dim]
        dst = h_proj[row]  # [E, out_dim]
        if self.edge_proj is not None and edge_attr is not None and edge_attr.shape[0] > 0:
            src = src + self.edge_proj(edge_attr)
            dst = dst + self.edge_proj(edge_attr)
        # Attention: score = attn_lin(src) + attn_lin(dst)
        attn_scores = (self.attn_lin(src) + self.attn_lin(dst)).squeeze(-1)  # [E]
        # Mask by graph boundaries
        mask = torch.ones(attn_scores.size(0), dtype=torch.bool, device=h.device)
        # Group attention by destination node
        # Use softmax over incoming edges per node
        # For simplicity: use log-sum-exp trick
        attn_scores_max = torch.full((h.size(0),), -1e9, device=h.device).scatter_reduce_(
            0, row, attn_scores, reduce='amax', include_self=False)
        attn_scores = attn_scores - attn_scores_max[row]
        attn_weights = torch.exp(attn_scores)  # [E]
        # Normalize per destination node
        denom = torch.zeros(h.size(0), device=h.device).index_add_(0, row, attn_weights)
        denom = denom.clamp(min=1e-8)
        attn_weights = attn_weights / denom[row]  # [E]
        # Weighted aggregation
        msg = src * attn_weights.unsqueeze(-1)  # [E, out_dim]
        agg = torch.zeros_like(h).index_add_(0, row, msg)
        out = self.bn(h_proj + agg)
        out = self.dropout(F.elu(out))
        return out


class GAT(nn.Module):
    """Graph Attention Network with multi-layer attention and mean pooling."""
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=5, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden, edge_dim, dropout=dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch_ids, n_graphs):
        h = self.input_proj(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr, batch_ids, n_graphs)
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h)
        counts = torch.bincount(batch_ids, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pooled = pooled / counts
        logits = self.head(pooled).squeeze(-1)
        return logits


def train_model(model, dataset, train_idx, val_idx, device, epochs=100, lr=0.001,
                pos_weight=None, patience=20, batch_size=256, weight_decay=1e-5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        total_loss = 0.0
        for i in range(0, len(train_idx), batch_size):
            chunk = [int(train_idx[j]) for j in perm[i:i + batch_size]]
            x, ei, ea, batch_ids, y = dataset.build_batch(chunk)
            x, ei, batch_ids, y = x.to(device), ei.to(device), batch_ids.to(device), y.to(device)
            if ea is not None:
                ea = ea.to(device)

            opt.zero_grad()
            logits = model(x, ei, ea, batch_ids, len(chunk))
            loss = F.binary_cross_entropy_with_logits(
                logits, y.squeeze(-1).float(), pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * len(chunk)

        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, len(val_idx), 512):
                chunk = [int(val_idx[j]) for j in range(i, min(i + 512, len(val_idx)))]
                x, ei, ea, batch_ids, y = dataset.build_batch(chunk)
                x, ei, batch_ids = x.to(device), ei.to(device), batch_ids.to(device)
                if ea is not None:
                    ea = ea.to(device)
                out = model(x, ei, ea, batch_ids, len(chunk))
                val_logits.append(out)
            val_logits = torch.cat(val_logits).cpu().numpy()
            val_y = dataset.y[val_idx].squeeze(-1).numpy()
            try:
                val_auc = roc_auc_score(val_y, val_logits)
            except Exception:
                val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val_auc


def predict_graphs(model, dataset, indices, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            chunk = [int(indices[j]) for j in range(i, min(i + batch_size, len(indices)))]
            x, ei, ea, batch_ids, _ = dataset.build_batch(chunk)
            x, ei, batch_ids = x.to(device), ei.to(device), batch_ids.to(device)
            if ea is not None:
                ea = ea.to(device)
            out = model(x, ei, ea, batch_ids, len(chunk))
            preds.append(torch.sigmoid(out).cpu())
    return torch.cat(preds) if preds else torch.zeros(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t_start = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    dataset = FlatGraphDataset(d)

    tr, va, te = d["split_sizes"]["train"], d["split_sizes"]["val"], d["split_sizes"]["test"]
    train_idx = torch.arange(tr)
    val_idx = torch.arange(tr, tr + va)
    target = (val_idx if a.split == "val"
              else torch.arange(tr + va, d["num_graphs"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = dataset.x.size(1)
    edge_dim = dataset.edge_attr.size(1) if dataset.edge_attr is not None else 0

    # Class weights
    y_train = dataset.y[train_idx].squeeze(-1)
    pos_count = (y_train == 1).float().sum().item()
    neg_count = (y_train == 0).float().sum().item()
    n_total = pos_count + neg_count
    w_pos = np.sqrt(n_total / (2 * pos_count))
    w_neg = np.sqrt(n_total / (2 * neg_count))
    pos_weight_tensor = torch.tensor([w_pos / w_neg], device=device)
    print(f"pos_weight={pos_weight_tensor.item():.3f} (pos={pos_count:.0f}, neg={neg_count:.0f})")

    all_preds = []

    def train_one(model, label):
        print(f"{label}", end=" ", flush=True)
        model, val_auc = train_model(
            model, dataset, train_idx, val_idx, device,
            epochs=100, lr=0.001, pos_weight=pos_weight_tensor,
            patience=20, batch_size=256,
        )
        print(f"val_auc={val_auc:.4f}", flush=True)
        pred = predict_graphs(model, dataset, target, device)
        all_preds.append(pred)
        del model
        torch.cuda.empty_cache()

    # --- GIN mean-pooling: 8 models with diverse hyperparams ---
    gin_mean_specs = [
        # (hidden, n_layers, dropout, seed)
        (128, 5, 0.2, 42),
        (128, 5, 0.2, 1042),
        (128, 5, 0.2, 2042),
        (128, 5, 0.2, 3042),
        (128, 5, 0.2, 4042),
        (128, 5, 0.2, 5042),
        (96, 4, 0.3, 6042),
        (160, 6, 0.1, 7042),
    ]
    for h, nl, dp, seed in gin_mean_specs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GIN(node_dim, edge_dim, hidden=h, n_layers=nl, dropout=dp, pooling='mean')
        model = model.to(device)
        train_one(model, f"GIN-mean h={h} L={nl} d={dp} s={seed}")

    # --- GIN meanmax-pooling: 4 models ---
    gin_mm_specs = [
        (128, 5, 0.2, 8042),
        (128, 5, 0.2, 9042),
        (96, 4, 0.3, 10042),
        (160, 6, 0.1, 11042),
    ]
    for h, nl, dp, seed in gin_mm_specs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GIN(node_dim, edge_dim, hidden=h, n_layers=nl, dropout=dp, pooling='meanmax')
        model = model.to(device)
        train_one(model, f"GIN-mm h={h} L={nl} d={dp} s={seed}")

    # --- GCN: 4 models with diverse hyperparams ---
    gcn_specs = [
        (128, 5, 0.2, 12042),
        (128, 5, 0.2, 13042),
        (96, 4, 0.3, 14042),
        (160, 6, 0.1, 15042),
    ]
    for h, nl, dp, seed in gcn_specs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GCN(node_dim, edge_dim, hidden=h, n_layers=nl, dropout=dp)
        model = model.to(device)
        train_one(model, f"GCN h={h} L={nl} d={dp} s={seed}")

    # --- GAT: 4 models with diverse hyperparams ---
    gat_specs = [
        (128, 5, 0.2, 16042),
        (128, 5, 0.2, 17042),
        (96, 4, 0.3, 18042),
        (160, 6, 0.1, 19042),
    ]
    for h, nl, dp, seed in gat_specs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GAT(node_dim, edge_dim, hidden=h, n_layers=nl, dropout=dp)
        model = model.to(device)
        train_one(model, f"GAT h={h} L={nl} d={dp} s={seed}")


    # Average ensemble
    ensemble_pred = torch.stack(all_preds).mean(dim=0)
    out = {"pred": ensemble_pred.unsqueeze(-1)}
    torch.save(out, Path(a.out))
    elapsed = time.time() - t_start
    print(f"\nWrote predictions for {len(target)} {a.split} graphs -> {a.out}")
    print(f"Total time: {elapsed:.1f}s, {len(all_preds)} models in ensemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
