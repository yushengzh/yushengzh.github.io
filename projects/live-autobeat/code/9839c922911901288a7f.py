"""Molecular Activity (ogbg-molhiv) - multi-seed GINE ensemble.

v6: 5 independent GINE models (one-hot input, edge-conditioned mean-norm conv,
sum readout) trained with different seeds; predictions are averaged.
Early-stopping uses a held-out TRAIN-aux slice (last 400 labeled train graphs),
never the visible val set, so model selection does not overfit the val proxy.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 256))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 5))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 40))
LR = float(os.environ.get("SOLVER_LR", 0.001))
WD = float(os.environ.get("SOLVER_WD", 1e-4))
BATCH = int(os.environ.get("SOLVER_BATCH", 256))
N_SEEDS = int(os.environ.get("SOLVER_NSEEDS", 5))
SEED0 = int(os.environ.get("SOLVER_SEED0", 1000))
DEVICE = os.environ.get("SOLVER_DEVICE", "cuda")
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.0))
EVAL_EVERY = int(os.environ.get("SOLVER_EVAL_EVERY", 2))
PATIENCE = int(os.environ.get("SOLVER_PATIENCE", 16))
GRAD_CLIP = float(os.environ.get("SOLVER_GRAD_CLIP", 1.0))
AUX_SIZE = int(os.environ.get("SOLVER_AUX", 400))


class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_sizes = [92, 1, 11, 9, 5, 3, 6, 2, 2]
        self.edge_sizes = [4, 1, 2]

    def forward_atom(self, x):
        parts = []
        for j, sz in enumerate(self.atom_sizes):
            if sz <= 1:
                continue
            parts.append(F.one_hot(x[:, j].long().clamp(0, sz), sz + 1).float())
        return torch.cat(parts, dim=-1)

    def forward_edge(self, ea):
        parts = []
        for j, sz in enumerate(self.edge_sizes):
            if sz <= 1:
                continue
            parts.append(F.one_hot(ea[:, j].long().clamp(0, sz), sz + 1).float())
        return torch.cat(parts, dim=-1)

    @property
    def atom_dim(self):
        return sum(sz + 1 for sz in self.atom_sizes if sz > 1)

    @property
    def edge_dim(self):
        return sum(sz + 1 for sz in self.edge_sizes if sz > 1)


class GINEConv(nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_lin = nn.Linear(edge_dim, hidden)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        h = self.mlp(x)
        m = h[src] + self.edge_lin(edge_attr)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        in_deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        in_deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        in_deg = in_deg.clamp(min=1.0)
        return x + agg / in_deg.unsqueeze(-1)


class Model(nn.Module):
    def __init__(self, hidden, n_layers, in_atom, in_edge, out_dim=1):
        super().__init__()
        self.node_lin = nn.Linear(in_atom, hidden)
        self.bn0 = nn.LayerNorm(hidden)
        self.convs = nn.ModuleList([GINEConv(hidden, in_edge) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(hidden, out_dim))

    def forward(self, x_raw, edge_index, batch, edge_attr_raw, n_graphs):
        x = self.node_lin(InputEmbedding.forward_atom(InputEmbedding(), x_raw))
        x = F.relu(self.bn0(x))
        edge_attr = InputEmbedding.forward_edge(InputEmbedding(), edge_attr_raw)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        pool = torch.zeros(n_graphs, x.size(1), device=x.device).index_add_(0, batch, x)
        return self.head(pool)


def slice_batch(indices, nptr, eptr, X, EI, EA):
    xs, eis, eas, bts = [], [], [], []
    offset = 0
    for i, gi in enumerate(indices):
        n0, n1 = int(nptr[gi]), int(nptr[gi + 1])
        e0, e1 = int(eptr[gi]), int(eptr[gi + 1])
        xs.append(X[n0:n1])
        if e1 > e0:
            ei_g = EI[:, e0:e1].long() - n0
            ei_g[0] += offset
            ei_g[1] += offset
            eis.append(ei_g)
            eas.append(EA[e0:e1])
        n = n1 - n0
        bts.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    x = torch.cat(xs, 0)
    if eis:
        ei = torch.cat(eis, 1).int()
        ea = torch.cat(eas, 0)
    else:
        ei = torch.zeros(2, 0, dtype=torch.int32)
        ea = torch.zeros(0, 3)
    bt = torch.cat(bts, 0)
    return x, ei, bt, ea


def make_predict(model, device, nptr, eptr, X, EI, EA):
    def predict(indices, batch_size=512):
        model.eval()
        outs = []
        for s in range(0, len(indices), batch_size):
            sel = indices[s:s + batch_size]
            x, ei, bt, ea = slice_batch(sel, nptr, eptr, X, EI, EA)
            with torch.no_grad():
                o = model(x.to(device), ei.to(device), bt.to(device), ea.to(device), len(sel))
            outs.append(torch.sigmoid(o).float().cpu())
        return torch.cat(outs, 0).squeeze(-1)
    return predict


def roc(y, p):
    y = y.squeeze(-1)
    ok = torch.isfinite(y) & torch.isfinite(p)
    if int(ok.sum()) < 2 or int((y[ok] == 1).sum()) < 2 or int((y[ok] == 0).sum()) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y[ok].cpu().numpy(), p[ok].cpu().numpy()))
    except Exception:
        return 0.5


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


def train_one(seed, device, nptr, eptr, X, EI, EA, tr_idx, tr_y, aux_idx, aux_y, val_idx, val_y):
    set_seed(seed)
    model = Model(HIDDEN, LAYERS, InputEmbedding().atom_dim, InputEmbedding().edge_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    predict = make_predict(model, device, nptr, eptr, X, EI, EA)
    train_idx = [i for i in tr_idx if i not in set(aux_idx)]
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_idx)).tolist()
        for s in range(0, len(train_idx), BATCH):
            sel = [train_idx[j] for j in perm[s:s + BATCH]]
            x, ei, bt, ea = slice_batch(sel, nptr, eptr, X, EI, EA)
            y = tr_y[sel].squeeze(-1)
            opt.zero_grad()
            out = model(x.to(device), ei.to(device), bt.to(device), ea.to(device), len(sel))
            loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), y.to(device).squeeze(-1))
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sched.step()
        if (ep + 1) % EVAL_EVERY == 0:
            a = roc(aux_y, predict(aux_idx))
            if a > best_auc + 1e-6:
                best_auc, bad = a, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += EVAL_EVERY
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    val_auc = roc(val_y, predict(val_idx))
    return model, val_auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    X, EI, EA = d["x"], d["edge_index"], d["edge_attr"]
    nptr, eptr, y = d["node_ptr"], d["edge_ptr"], d["y"]
    ns = d["split_sizes"]
    tr, va, te = ns["train"], ns["val"], ns["test"]
    tr_idx = list(range(0, tr))
    val_idx = list(range(tr, tr + va))
    test_idx = list(range(tr + va, tr + va + te))
    ytr, yval, ytest = y[tr_idx], y[val_idx], y[test_idx]

    aux_idx = tr_idx[-AUX_SIZE:]
    aux_y = ytr[-AUX_SIZE:]
    target = val_idx if a.split == "val" else test_idx
    target_y = yval if a.split == "val" else ytest
    device = DEVICE if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    all_probs = []
    val_aucs = []
    for k in range(N_SEEDS):
        seed = SEED0 + k
        t1 = time.time()
        model, val_auc = train_one(seed, device, nptr, eptr, X, EI, EA, tr_idx, ytr, aux_idx, aux_y, val_idx, yval)
        val_aucs.append(val_auc)
        probs = make_predict(model, device, nptr, eptr, X, EI, EA)(target)
        all_probs.append(probs)
        print(f"seed {seed} val_auc={val_auc:.4f} {time.time()-t1:.1f}s (total {time.time()-t0:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    ens = torch.stack(all_probs, 0).mean(0)
    print(f"ENSEMBLE N={N_SEEDS} val_mean={sum(val_aucs)/len(val_aucs):.4f} val_min={min(val_aucs):.4f}", flush=True)
    torch.save({"pred": ens.unsqueeze(-1)}, Path(a.out))
    print(f"wrote {a.out} shape {tuple(ens.shape)} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
