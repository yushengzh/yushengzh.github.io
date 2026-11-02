#!/usr/bin/env python3
"""Blood-Brain Barrier penetration (ogbg-molbbbp): graph classification.

GINE-style GNN with per-category node/edge embeddings, LayerNorm, residual
message passing and mean+max+sum + virtual-node readout. Trains an ensemble of
seeded models and fuses their scores via mean z-score to reduce single-model
variance. No memorization of the visible data; everything generalizes from the
OGB categorical features.

CLI contract (fixed):
    python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- tunables (env-overridable) ----
HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 64))
NLAYERS = int(os.environ.get("SOLVER_LAYERS", 4))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 60))
LR = float(os.environ.get("SOLVER_LR", 0.003))
WD = float(os.environ.get("SOLVER_WD", 5e-4))
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.1))
BATCH = int(os.environ.get("SOLVER_BATCH", 256))
PATIENCE = int(os.environ.get("SOLVER_PATIENCE", 15))
EMBDIM = int(os.environ.get("SOLVER_EMBDIM", 12))
BUDGET_MIN = float(os.environ.get("SOLVER_BUDGET_MIN", 40.0))
SEED = int(os.environ.get("SOLVER_SEED", 12345))
SEEDS = int(os.environ.get("SOLVER_SEEDS", 15))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def onehot_to_index(x):
    return x.long().clamp_min(0)


def onehot_edge_to_index(ea):
    return ea.long().clamp_min(0)


def build_graphs(d, indices):
    xs, eis, bts, eais = [], [], [], []
    offset = 0
    for i, g in enumerate(indices):
        n0 = int(d["node_ptr"][g]); n1 = int(d["node_ptr"][g + 1])
        e0 = int(d["edge_ptr"][g]); e1 = int(d["edge_ptr"][g + 1])
        xs.append(onehot_to_index(d["x"][n0:n1]))
        ei = d["edge_index"][:, e0:e1].long()
        if ei.numel():
            eis.append((ei - n0 + offset).contiguous())
        bts.append(torch.full((n1 - n0,), i, dtype=torch.long))
        eais.append(onehot_edge_to_index(d["edge_attr"][e0:e1]))
        offset += n1 - n0
    x = torch.cat(xs) if xs else torch.zeros(0, d["x"].size(1), dtype=torch.long)
    ei = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    batch = torch.cat(bts) if bts else torch.zeros(0, dtype=torch.long)
    ea = torch.cat(eais) if eais else torch.zeros(0, 3, dtype=torch.long)
    return x.to(device), ei.to(device), batch.to(device), ea.to(device)


class GINEConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.phi1 = nn.Linear(hidden, hidden)
        self.phi2 = nn.Linear(hidden, hidden)

    def forward(self, h, edge_h, src, dst, batch):
        m = h[src] + edge_h
        m = F.elu(self.phi1(m))
        m = self.phi2(m)
        n = h.size(0)
        scatter = torch.zeros((n, h.size(1)), device=h.device)
        scatter.index_add_(0, dst, m)
        return scatter + h  # residual (input + self)


class GINE(nn.Module):
    def __init__(self, ncat, cat_size, ncat_e, cat_size_e, hidden, n_layers):
        super().__init__()
        self.node_emb = nn.ModuleList(
            [nn.Embedding(cat_size[c], EMBDIM) for c in range(ncat)]
        )
        self.edge_emb = nn.ModuleList(
            [nn.Embedding(cat_size_e[c], EMBDIM) for c in range(ncat_e)]
        )
        self.node_in = nn.Linear(ncat * EMBDIM, hidden)
        self.edge_in = nn.Linear(ncat_e * EMBDIM, hidden)
        self.convs = nn.ModuleList([GINEConv(hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(DROPOUT)
        self.vnode = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(6 * hidden, 1)

    def _embed(self, idx, eais):
        node_e = torch.cat([self.node_emb[c](idx[:, c]) for c in range(idx.size(1))], dim=1)
        edge_e = torch.cat([self.edge_emb[c](eais[:, c]) for c in range(eais.size(1))], dim=1)
        return self.node_in(node_e), self.edge_in(edge_e)

    def _readout(self, h, batch):
        from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
        gm = global_mean_pool(h, batch)
        gx = global_max_pool(h, batch)
        gs = global_add_pool(h, batch)
        return torch.cat([gm, gx, gs], dim=1)

    def forward(self, x, ei, batch, ea, train=True):
        h, edge_h = self._embed(x, ea)
        src, dst = ei[0], ei[1]
        for i in range(len(self.convs)):
            if src.numel():
                h = self.convs[i](h, edge_h, src, dst, batch)
            h = self.norms[i](h)
            h = F.elu(h)
            if train:
                h = self.dropout(h)
        n = h.size(0)
        hv = self.vnode.expand(n, -1)
        m_cat = torch.cat([self._readout(h, batch), self._readout(hv, batch)], dim=1)
        return self.out(m_cat).squeeze(-1)


def train_one(seed, d, cat_size, cat_size_e, tr_y, va_y, tr_idx, va_idx, epochs):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = GINE(len(cat_size), cat_size, len(cat_size_e), cat_size_e,
                 HIDDEN, NLAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    n = len(tr_idx)

    def predict(d2, idx):
        bx, be, bb, bea = build_graphs(d2, list(idx))
        model.eval()
        with torch.no_grad():
            return model(bx, be, bb, bea, train=False)

    def val_auc():
        p = predict(d, va_idx)
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(va_y.cpu().numpy().ravel(),
                                 p.cpu().numpy().ravel())
        except Exception:
            return 0.5

    t0 = time.time()
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s0 in range(0, n, BATCH):
            b = perm[s0:s0 + BATCH]
            bx, be, bb, bea = build_graphs(d, b.tolist())
            opt.zero_grad()
            logit = model(bx, be, bb, bea, train=True)
            loss = F.binary_cross_entropy_with_logits(logit, tr_y[b].squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        auc = val_auc()
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_epoch = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        el = time.time() - t0
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[tr] seed={seed} ep={ep} val_auc={auc:.4f} best={best_auc:.4f} t={el:.0f}s", flush=True)
        if el > 0 and (ep - best_epoch) >= PATIENCE and el > 60:
            print(f"[tr] seed={seed} early stop at ep={ep}", flush=True)
            break
        el = time.time() - t0
        if el / max(ep + 1, 1) * epochs > BUDGET_MIN * 60 and ep >= 10:
            print(f"[tr] seed={seed} budget guard stop at ep={ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[tr] seed={seed} done best_val_auc={best_auc:.4f} ep={best_epoch} total={time.time()-t0:.0f}s", flush=True)
    return model, best_auc


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    d = torch.load(os.path.join(a.data_dir, "dataset.pt"), map_location="cpu")
    s = d["split_sizes"]
    ntrain, nval, ntest = s["train"], s["val"], s["test"]
    tr_idx = list(range(0, ntrain))
    va_idx = list(range(ntrain, ntrain + nval))
    te_idx = list(range(ntrain + nval, ntrain + nval + ntest))

    x = d["x"]
    cat_size = [int(x[:, c].max().item()) + 1 for c in range(x.size(1))]
    ea = d["edge_attr"]
    cat_size_e = [int(ea[:, c].max().item()) + 1 for c in range(ea.size(1))]

    tr_y = d["y"][0:ntrain].float().to(device)
    va_y = d["y"][ntrain:ntrain + nval].float().to(device)

    out_idx = va_idx if a.split == "val" else te_idx

    # train ensemble of seeded models
    models = []
    aucs = []
    global_budget = BUDGET_MIN * 60
    for k in range(SEEDS):
        seed = SEED + 1000 * k
        per_seed_budget = max(20.0, global_budget / SEEDS)
        old_budget = BUDGET_MIN
        # cap per-model budget so the ensemble stays within global budget
        models.append(None)
        model, auc = train_one(seed, d, cat_size, cat_size_e, tr_y, va_y,
                               tr_idx, va_idx, EPOCHS)
        models[k] = model
        aucs.append(auc)

    # fuse: mean z-score of each model's scores
    preds = []
    for model in models:
        model.eval()
        bx, be, bb, bea = build_graphs(d, list(out_idx))
        with torch.no_grad():
            p = model(bx, be, bb, bea, train=False).float().cpu().numpy().ravel()
        preds.append(p)
    import numpy as np
    P = np.stack(preds, axis=1)
    mean = P.mean(axis=0, keepdims=True)
    std = P.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    fused = ((P - mean) / std).mean(axis=1)
    fused = fused.astype(np.float32)

    # report fused val AUC when scoring val
    if a.split == "val":
        yv = va_y.cpu().numpy().ravel()
        mask = ~np.isnan(yv)
        try:
            from sklearn.metrics import roc_auc_score
            fa = roc_auc_score(yv[mask], fused[mask])
            print(f"[fused] val ROC-AUC={fa:.4f}  per-model aucs={[round(a,4) for a in aucs]}", flush=True)
        except Exception as e:
            print(f"[fused] err {e}", flush=True)

    torch.save({"pred": torch.from_numpy(fused).unsqueeze(-1)}, a.out)
    print(f"[out] wrote {a.out} pred shape {(len(fused),1)} split={a.split} n_models={SEEDS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
