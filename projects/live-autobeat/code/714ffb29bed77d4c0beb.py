#!/usr/bin/env python3
"""Diverse graph-classifier ensemble for ogbg-molbbbp (BBB-penetration).

Architectures: GIN (mean pooling), GIN (mean+max pooling), GCN.
Trained with class-weighted BCE, cosine LR, weight-averaged checkpoint
(no val-set based early stopping / weight selection, to avoid leaking the
visible val into the trained weights). Final score = rank-average of the
ensemble's probability outputs.
"""
from __future__ import annotations

import argparse
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
            n0 = int(d["node_ptr"][gi]); n1 = int(d["node_ptr"][gi + 1])
            e0 = int(d["edge_ptr"][gi]); e1 = int(d["edge_ptr"][gi + 1])
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
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None

    def forward(self, h, edge_index, edge_attr, batch_ids, n_graphs):
        row, col = edge_index
        if self.edge_proj is not None and edge_attr is not None and edge_attr.shape[0] > 0:
            msg = h[col] + self.edge_proj(edge_attr)
        else:
            msg = h[col]
        agg = torch.zeros_like(h).index_add_(0, row, msg)
        out = self.mlp((1 + self.eps) * h + agg)
        return out


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None

    def forward(self, h, edge_index, edge_attr, batch_ids, n_graphs):
        row, col = edge_index
        h_lin = self.bn(self.lin(h))
        msg = h_lin[col]
        if self.edge_proj is not None and edge_attr is not None and edge_attr.shape[0] > 0:
            msg = msg + self.edge_proj(edge_attr)
        agg = torch.zeros_like(h_lin).index_add_(0, row, msg)
        deg = torch.bincount(row, minlength=h.size(0)).clamp(min=1).float().sqrt()
        agg = agg / deg.unsqueeze(1)
        return self.bn(h_lin + agg)


class Head(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled):
        return self.net(pooled).squeeze(-1)


def _graph_max(h, batch_ids, n_graphs):
    """Autograd-friendly per-graph max pooling."""
    idx = batch_ids.unsqueeze(1).expand_as(h)
    filled = torch.full((n_graphs, h.size(1)), -1e9, device=h.device)
    filled.scatter_reduce_(0, idx, h, reduce='amax', include_self=True)
    return filled


def _pool(h, batch_ids, n_graphs, pooling):
    counts = torch.bincount(batch_ids, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
    if pooling in ("mean", "meanmax"):
        mean_p = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h) / counts
    else:
        mean_p = None
    if pooling in ("max", "meanmax"):
        max_p = _graph_max(h, batch_ids, n_graphs)
    else:
        max_p = None
    if pooling == "mean":
        return mean_p
    if pooling == "max":
        return max_p
    return torch.cat([mean_p, max_p], dim=1)


class GIN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=5, dropout=0.2, eps=0.0, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GINConv(hidden, hidden, edge_dim, eps) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        hdim = hidden * (2 if pooling == "meanmax" else 1)
        self.head = Head(hdim, hidden, dropout)

    def forward(self, x, edge_index, edge_attr, batch_ids, n_graphs):
        h = self.input_proj(x)
        for conv in self.convs:
            h = self.dropout(conv(h, edge_index, edge_attr, batch_ids, n_graphs))
        pooled = _pool(h, batch_ids, n_graphs, self.pooling)
        return self.head(pooled)


class GCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=5, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden, edge_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = Head(hidden, hidden, dropout)

    def forward(self, x, edge_index, edge_attr, batch_ids, n_graphs):
        h = self.input_proj(x)
        for conv in self.convs:
            h = self.dropout(conv(h, edge_index, edge_attr, batch_ids, n_graphs))
        pooled = _pool(h, batch_ids, n_graphs, "mean")
        return self.head(pooled)


MODELS = {
    "gin_mean": lambda nd, ed, **kw: GIN(nd, ed, pooling="mean", **kw),
    "gin_max": lambda nd, ed, **kw: GIN(nd, ed, pooling="max", **kw),
    "gin_meanmax": lambda nd, ed, **kw: GIN(nd, ed, pooling="meanmax", **kw),
    "gcn": lambda nd, ed, **kw: GCN(nd, ed, **kw),
}


def train_model(model, dataset, train_idx, device, epochs=150, lr=0.001,
                pos_weight=None, batch_size=256, weight_decay=1e-5,
                swa_start=0.8, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    swa_ckpt = None
    swa_count = 0
    swa_thresh = int(epochs * swa_start)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), batch_size):
            chunk = [int(train_idx[j]) for j in perm[i:i + batch_size]]
            x, ei, ea, batch_ids, y = dataset.build_batch(chunk)
            x, ei, batch_ids, y = x.to(device), ei.to(device), batch_ids.to(device), y.to(device)
            if ea is not None:
                ea = ea.to(device)
            opt.zero_grad()
            logits = model(x, ei, ea, batch_ids, len(chunk))
            loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1).float(), pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        sched.step()
        # SWA: average weights over the last portion of training (no val used)
        if epoch >= swa_thresh:
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if swa_ckpt is None:
                swa_ckpt = sd
            else:
                for k in swa_ckpt:
                    swa_ckpt[k] = swa_ckpt[k] + sd[k]
            swa_count += 1
    if swa_ckpt is not None and swa_count > 0:
        model.load_state_dict({k: (v / swa_count).to(device) for k, v in swa_ckpt.items()})
    return model


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


def rank_average(prob_mats: list) -> np.ndarray:
    """Combine per-model probability vectors by averaging their ranks."""
    n = prob_mats[0].shape[0]
    arr = np.stack([np.asarray(p, dtype=float) for p in prob_mats], axis=0)  # [M, n]
    from scipy.stats import rankdata
    ranks = np.stack([rankdata(arr[m]) for m in range(arr.shape[0])], axis=0)
    return ranks.mean(axis=0)


def build_ensemble_specs():
    # (arch, n_seeds, hidden, n_layers, dropout, eps)
    specs = []
    for s in range(6):
        specs.append(("gin_mean", 101 + s * 1000, dict(hidden=128, n_layers=5, dropout=0.2, eps=0.0)))
    for s in range(3):
        specs.append(("gin_meanmax", 501 + s * 1000, dict(hidden=128, n_layers=5, dropout=0.2, eps=0.0)))
    for s in range(3):
        specs.append(("gin_max", 901 + s * 1000, dict(hidden=128, n_layers=5, dropout=0.2, eps=0.0)))
    for s in range(3):
        specs.append(("gcn", 1301 + s * 1000, dict(hidden=128, n_layers=5, dropout=0.2)))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t_start = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    dataset = FlatGraphDataset(d)
    tr, va = d["split_sizes"]["train"], d["split_sizes"]["val"]
    train_idx = torch.arange(tr)
    val_idx = torch.arange(tr, tr + va)
    target = (val_idx if a.split == "val" else torch.arange(tr + va, d["num_graphs"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = dataset.x.size(1)
    edge_dim = dataset.edge_attr.size(1) if dataset.edge_attr is not None else 0

    y_train = dataset.y[train_idx].squeeze(-1)
    pos_count = (y_train == 1).float().sum().item()
    neg_count = (y_train == 0).float().sum().item()
    n_total = pos_count + neg_count
    w_pos = np.sqrt(n_total / (2 * pos_count))
    w_neg = np.sqrt(n_total / (2 * neg_count))
    pos_weight_tensor = torch.tensor([w_pos / w_neg], device=device)
    print(f"pos_weight={pos_weight_tensor.item():.3f} (pos={pos_count:.0f}, neg={neg_count:.0f})")

    specs = build_ensemble_specs()
    all_preds = []
    for k, (arch, seed, kw) in enumerate(specs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MODELS[arch](node_dim, edge_dim, **kw).to(device)
        print(f"[{k+1}/{len(specs)}] {arch} seed={seed}", end=" ", flush=True)
        model = train_model(model, dataset, train_idx, device,
                            epochs=150, lr=0.001, pos_weight=pos_weight_tensor,
                            batch_size=256, swa_start=0.8, seed=seed)
        pred = predict_graphs(model, dataset, target, device)
        all_preds.append(pred)
        # quick val auc for diagnostics only (not used for selection)
        if a.split == "test":
            vp = predict_graphs(model, dataset, val_idx, device)
            try:
                print(f"val_auc={roc_auc_score(dataset.y[val_idx].squeeze(-1).numpy(), vp.numpy()):.4f}", flush=True)
            except Exception:
                print("(no val)", flush=True)
        del model
        torch.cuda.empty_cache()

    ensemble = rank_average(all_preds)
    out = {"pred": torch.tensor(ensemble, dtype=torch.float32).unsqueeze(-1)}
    torch.save(out, Path(a.out))
    print(f"\nWrote predictions for {len(target)} {a.split} graphs -> {a.out}")
    print(f"Total time: {time.time()-t_start:.1f}s, {len(all_preds)} models in ensemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
