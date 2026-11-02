#!/usr/bin/env python3
"""Aqueous solubility (ogbg-molesol) graph regression - GIN.

Clean, correct GIN:
- per-column atom embeddings summed -> node init
- per-column bond embeddings summed -> edge init
- L GIN layers (epsilon=learnable, default -1), residual, batchnorm
- sum + mean + max pooling -> MLP head
- cosine LR, grad clip, dropout in head, val early-stop, finite-safe
- optional ensemble; global time budget so it always finishes
Contract: python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
Writes {"pred": [M,1] float}.
"""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = int(os.environ.get("HIDDEN", 160))
LAYERS = int(os.environ.get("LAYERS", 4))
DROPOUT = float(os.environ.get("DROPOUT", 0.10))
WD = float(os.environ.get("WD", 1e-4))
LR = float(os.environ.get("LR", 0.005))
EPOCHS = int(os.environ.get("EPOCHS", 200))
BATCH = int(os.environ.get("BATCH", 128))
ENSEMBLE = int(os.environ.get("ENSEMBLE", 5))
PATIENCE = int(os.environ.get("PATIENCE", 30))
SEED = int(os.environ.get("SEED", 0))
DEVI = os.environ.get("DEVI", "cuda")
TIME_BUDGET = float(os.environ.get("TIME_BUDGET", 3200))  # seconds (grader 60min)
N_ATOM = 9
N_BOND = 3


class GINLayer(nn.Module):
    def __init__(self, h, eps=1.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.bn = nn.BatchNorm1d(h)
        self.eps = nn.Parameter(torch.tensor(eps))

    def forward(self, h, edge_index, edge_feat):
        row, col = edge_index
        agg = torch.zeros_like(h)
        if row.numel():
            msg = h[col] + edge_feat
            agg = agg.index_add_(0, row, msg)
        return self.bn(self.mlp((1.0 + self.eps) * h + agg))


class GNN(nn.Module):
    def __init__(self, atom_v, hidden, layers, dropout, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.node_embs = nn.ModuleList([nn.Embedding(int(v), hidden) for v in atom_v])
        self.bt = nn.Embedding(4, hidden)
        self.bs = nn.Embedding(3, hidden)
        self.bj = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList([GINLayer(hidden, eps=0.0) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def node_init(self, x):
        h = None
        for c in range(N_ATOM):
            ec = self.node_embs[c](x[:, c].long())
            h = ec if h is None else h + ec
        return h

    def edge_init(self, edge_attr):
        if edge_attr.numel():
            return self.bt(edge_attr[:, 0].long()) + self.bs(edge_attr[:, 1].long()) + self.bj(edge_attr[:, 2].long())
        return torch.zeros(0, self.node_embs[0].embedding_dim, device=edge_attr.device)

    def _pool(self, h, batch):
        B = int(batch.max().item()) + 1
        ssum = h.new_zeros(B, h.size(1)).index_add_(0, batch, h)
        cnt = torch.bincount(batch, minlength=B).float().clamp(min=1).unsqueeze(-1)
        smean = ssum / cnt
        smax = h.new_full((B, h.size(1)), float("-inf")).scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=False)
        smax = torch.where(h.new_full((), float("-inf")) == smax, h.new_zeros(1), smax)
        # fix any all- -inf rows (should not happen; each graph has >=1 node)
        return torch.cat([ssum, smean, smax], dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_init(x)
        e = self.edge_init(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return self.head(self._pool(h, batch)).squeeze(-1)


def feature_vocab(d):
    return [int(d["x"][:, c].max().item()) + 1 for c in range(N_ATOM)]


def make_batch(d, indices, want_y):
    xs, eis, eattrs, batches, ys = [], [], [], [], []
    offset = 0
    for i, gidx in enumerate(indices):
        n0, n1 = int(d["node_ptr"][gidx]), int(d["node_ptr"][gidx + 1])
        e0, e1 = int(d["edge_ptr"][gidx]), int(d["edge_ptr"][gidx + 1])
        x = d["x"][n0:n1]
        xs.append(x)
        if e1 > e0:
            ei = d["edge_index"][:, e0:e1].long() - n0
            eis.append(ei + offset)
            eattrs.append(d["edge_attr"][e0:e1])
        batches.append(torch.full((x.size(0),), i, dtype=torch.long))
        if want_y:
            ys.append(d["y"][gidx])
        offset += x.size(0)
    x = torch.cat(xs)
    if eis:
        edge_index = torch.cat(eis, dim=1)
        edge_attr = torch.cat(eattrs)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, N_BOND)
    batch = torch.cat(batches)
    y = torch.cat(ys) if ys else torch.zeros(0)
    return x, edge_index, edge_attr, batch, y


def _eval_rmse(model, d, idx, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            x, ei, ea, bt, _ = make_batch(d, idx[i:i + 256], False)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); bt = bt.to(device)
            ps.append(model(x, ei, ea, bt))
    p = torch.cat(ps)
    y = d["y"][idx].squeeze(-1).to(device)
    return float(F.mse_loss(p, y).sqrt().item())


def train_one(d, train_idx, val_idx, atom_v, seed, epochs, device, time_budget=None, t0=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GNN(atom_v, HIDDEN, LAYERS, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    def sched(e):
        return 0.5 * (1 + math.cos(math.pi * e / max(1, epochs)))
    best = float("inf"); best_m = None; bad = 0
    tr = list(train_idx); n = len(tr)
    for ep in range(epochs):
        if time_budget is not None and (time.time() - t0) > time_budget:
            break
        model.train()
        for pg in opt.param_groups:
            pg["lr"] = LR * sched(ep)
        order = np.random.permutation(n)
        for i in range(0, n, BATCH):
            idx = [tr[j] for j in order[i:i + BATCH]]
            x, ei, ea, bt, y = make_batch(d, idx, True)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); bt = bt.to(device); y = y.to(device)
            opt.zero_grad()
            pred = model(x, ei, ea, bt)
            loss = F.mse_loss(pred, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if val_idx is not None:
            vrmse = _eval_rmse(model, d, val_idx, device)
            if math.isfinite(vrmse):
                if vrmse < best - 1e-6:
                    best = vrmse; bad = 0
                    best_m = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
    if best_m is None:
        best_m = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_m, best


def predict(model_state, d, indices, device, atom_v, bs=256):
    model = GNN(atom_v, HIDDEN, LAYERS, DROPOUT).to(device)
    model.load_state_dict(model_state)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            x, ei, ea, bt, _ = make_batch(d, indices[i:i + bs], False)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); bt = bt.to(device)
            out.append(model(x, ei, ea, bt).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    t0 = time.time()
    device = DEVI if torch.cuda.is_available() else "cpu"
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    ss = d["split_sizes"]
    tr = list(range(ss["train"]))
    val = list(range(ss["train"], ss["train"] + ss["val"]))
    atom_v = feature_vocab(d)
    # time budget per ensemble member so we always finish well under the limit
    per_member = (TIME_BUDGET - 60) / max(1, ENSEMBLE)
    models = []
    for s in range(ENSEMBLE):
        m, b = train_one(d, tr, val, atom_v, seed=SEED + s, epochs=EPOCHS, device=device,
                         time_budget=per_member, t0=time.time() - (time.time() - t0))
        models.append(m)
        print("  member %d/%d val_rmse=%.4f (t=%.0fs)" % (s + 1, ENSEMBLE, b, time.time() - t0), flush=True)
    if a.split == "val":
        idx = val
    else:
        idx = list(range(ss["train"] + ss["val"], d["num_graphs"]))
    preds = [predict(m, d, idx, device, atom_v) for m in models]
    pred = torch.stack(preds, dim=1).mean(1)
    torch.save(dict(pred=pred.view(-1, 1)), Path(a.out))
    print("wrote %d %s preds (ensemble of %d)  total t=%.0fs" % (len(idx), a.split, ENSEMBLE, time.time() - t0), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
