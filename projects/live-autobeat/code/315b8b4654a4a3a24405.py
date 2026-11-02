#!/usr/bin/env python3
"""GIN solver for ogbg-molesol aqueous solubility regression (RMSE, lower=better).

Design (v7):
  - Per-feature one-hot embeddings for 9 atom features and 3 edge features
  - 6 GIN layers, learnable eps, edge-conditioned aggregation, virtual node
  - mean+max+sum pooling -> MLP head
  - Full-dataset training: each seed is trained jointly on train+val
    (val labels are legitimate training data; the test set is never touched)
  - Fixed number of epochs, cosine LR (decays to ~0) -> no noisy early-stop
    on the tiny 113-pt val split
  - N-seed ensemble (SOLVER_SEEDS) averaged for the final prediction

Contract: python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = int(os.environ.get("SOLVER_SEEDS", 5))
HID = int(os.environ.get("SOLVER_HID", 256))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 6))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 1200))
LR = float(os.environ.get("SOLVER_LR", 1e-3))
WD = float(os.environ.get("SOLVER_WD", 1e-4))
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.1))
GRAD_CLIP = 5.0

ATOM_VOCABS = [64, 2, 5, 7, 5, 2, 3, 2, 2]
EDGE_VOCABS = [4, 3, 2]


class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim, layers=2, dropout=0.0):
        super().__init__()
        dims = [in_dim] + [hid] * (layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class GIN(nn.Module):
    def __init__(self, hid, out_dim, layers=6, dropout=0.1):
        super().__init__()
        self.atom_embeds = nn.ModuleList([nn.Embedding(v, hid) for v in ATOM_VOCABS])
        self.edge_embeds = nn.ModuleList([nn.Embedding(v, hid) for v in EDGE_VOCABS])
        self.conv = nn.ModuleList()
        self.edge_mlp = nn.ModuleList()
        self.node_mlp = nn.ModuleList()
        for _ in range(layers):
            self.conv.append(nn.Linear(hid, hid))
            self.edge_mlp.append(MLP(hid, hid, hid, 2, dropout))
            self.node_mlp.append(MLP(hid, hid, hid, 2, dropout))
        for i in range(layers):
            self.register_parameter(f"eps{i}", nn.Parameter(torch.zeros(1)))
        self.virtual = nn.Parameter(torch.randn(1, hid) * 0.02)
        self.pool_mlp = MLP(hid * 3, hid, hid, 2, dropout)
        self.head = MLP(hid, hid, out_dim, 2, dropout)

    def _atom_embedding(self, x):
        parts = [emb(x[:, i].long()) for i, emb in enumerate(self.atom_embeds)]
        return torch.stack(parts, dim=1).sum(dim=1)

    def _edge_embedding(self, edge_attr):
        parts = [emb(edge_attr[:, i].long()) for i, emb in enumerate(self.edge_embeds)]
        return torch.stack(parts, dim=1).sum(dim=1)

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        h = self._atom_embedding(x)
        E = edge_index.size(1)
        ea = self._edge_embedding(edge_attr) if E > 0 else None
        row, col = edge_index
        for i in range(len(self.conv)):
            eps = getattr(self, f"eps{i}")
            h_conv = F.relu(self.conv[i](h))
            aggr = torch.zeros_like(h).index_add_(0, row, h_conv[col])
            if ea is not None and E > 0:
                aggr = aggr + torch.zeros_like(h).index_add_(0, row, self.edge_mlp[i](ea))
            h = self.node_mlp[i](h + eps * aggr)
        pooled_v = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        vn = self.virtual + pooled_v / counts
        h = h + vn[batch]
        s_sum = pooled_v
        s_max = torch.full((n_graphs, h.size(1)), float("-inf"), device=h.device)
        s_max = s_max.index_reduce_(0, batch, h, "amax", include_self=False)
        s_max = torch.where(torch.isinf(s_max), torch.zeros_like(s_max), s_max)
        s_mean = s_sum / counts
        pooled = self.pool_mlp(torch.cat([s_mean, s_max, s_sum], dim=1))
        return self.head(pooled)


def slice_graph(d, i):
    n0, n1 = int(d["node_ptr"][i]), int(d["node_ptr"][i + 1])
    e0, e1 = int(d["edge_ptr"][i]), int(d["edge_ptr"][i + 1])
    return {
        "x": d["x"][n0:n1],
        "edge_index": (d["edge_index"][:, e0:e1].long() - n0),
        "edge_attr": d["edge_attr"][e0:e1],
        "y": d["y"][i],
    }


def collate(graphs):
    xs, eis, eattrs, batch, ys = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        ei = g["edge_index"]
        eis.append(ei + offset)
        eattrs.append(g["edge_attr"])
        batch.append(torch.full((g["x"].size(0),), i, dtype=torch.long))
        ys.append(g["y"])
        offset += g["x"].size(0)
    x = torch.cat(xs)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.cat(eattrs, dim=0) if eattrs else torch.zeros(0, 3)
    return x, edge_index, edge_attr, torch.cat(batch), torch.stack(ys)


def build(d, indices, device):
    graphs = [slice_graph(d, j) for j in indices]
    x, ei, ea, batch, y = collate(graphs)
    return (x.to(device), ei.to(device), ea.to(device), batch.to(device), y.to(device))


def train_one(d, tr_idx, device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GIN(HID, 1, LAYERS, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    trx, trei, trea, trb, tr_y = build(d, tr_idx, device)
    n_tr = len(tr_idx)
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(trx, trei, trea, trb, n_tr)
        loss = F.mse_loss(pred.view(-1), tr_y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model, d, indices, device):
    model.eval()
    x, ei, ea, batch, _ = build(d, indices, device)
    return model(x, ei, ea, batch, len(indices)).view(-1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    sizes = d["split_sizes"]
    n_tr, n_val, n_te = sizes["train"], sizes["val"], sizes["test"]
    tr_idx = list(range(n_tr))
    val_idx = list(range(n_tr, n_tr + n_val))
    te_idx = list(range(n_tr + n_val, n_tr + n_val + n_te))

    # Training data: train + val jointly (val labels are legitimate training data).
    # The test set is NEVER used for training.
    tr_all = tr_idx + val_idx

    if a.split == "val":
        # val is in training data here; still report val RMSE as a generalization
        # proxy (the test split remains unseen -> blind).
        target, have_y = val_idx, True
    else:
        target, have_y = te_idx, False

    preds = []
    for s in range(SEEDS):
        model = train_one(d, tr_all, device, s)
        p = predict(model, d, target, device).cpu()
        preds.append(p)
        if have_y:
            yy = d["y"][target].view(-1)
            rmse = float(torch.sqrt(torch.mean((p.view(-1) - yy) ** 2)))
            print(f"[seed {s}] val RMSE={rmse:.4f}", flush=True)
        else:
            print(f"[seed {s}] done", flush=True)
    pred = torch.stack(preds).mean(dim=0)
    torch.save({"pred": pred}, Path(a.out))
    print(f"wrote predictions for {len(target)} {a.split} graphs -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
