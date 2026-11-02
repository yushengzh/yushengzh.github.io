"""GIN-based graph-level binary classifier for OGBG-MolHIV.

Improvements over v4 baseline:
  - One-hot / embedding input for categorical features (atomic_num is 92 cats)
  - GINE message passing (edge-conditioned, mean-normalized)
  - Sum readout (standard for GIN/GCN on molhiv)
  - Early stopping on val ROC-AUC with best-state restore
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- hyperparameters (env-overridable) ----
HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 256))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 5))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 40))
LR = float(os.environ.get("SOLVER_LR", 0.001))
WD = float(os.environ.get("SOLVER_WD", 1e-4))
BATCH = int(os.environ.get("SOLVER_BATCH", 256))
SEED = int(os.environ.get("SOLVER_SEED", 42))
DEVICE = os.environ.get("SOLVER_DEVICE", "cuda")
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.0))
EVAL_EVERY = int(os.environ.get("SOLVER_EVAL_EVERY", 2))
PATIENCE = int(os.environ.get("SOLVER_PATIENCE", 12))
GRAD_CLIP = float(os.environ.get("SOLVER_GRAD_CLIP", 1.0))
POS_W = float(os.environ.get("SOLVER_POS_W", 0.0))


def _onehot(x, max_val):
    """One-hot encode integer tensor x along last dim, size max_val+1."""
    if max_val <= 0:
        return x  # no valid categories
    return F.one_hot(x.long().clamp(0, max_val), num_classes=max_val + 1).float()


class InputEmbedding(nn.Module):
    """One-hot encode all 9 atom features and 3 edge features."""
    def __init__(self):
        super().__init__()
        # atom features: [92, 1, 11, 9, 5, 3, 6, 2, 2]
        self.atom_sizes = [92, 1, 11, 9, 5, 3, 6, 2, 2]
        # edge features: [4, 1, 2]
        self.edge_sizes = [4, 1, 2]

    def forward_atom(self, x):
        parts = []
        for j, sz in enumerate(self.atom_sizes):
            if sz <= 1:
                continue
            parts.append(_onehot(x[:, j], sz))
        return torch.cat(parts, dim=-1)

    def forward_edge(self, ea):
        parts = []
        for j, sz in enumerate(self.edge_sizes):
            if sz <= 1:
                continue
            parts.append(_onehot(ea[:, j], sz))
        return torch.cat(parts, dim=-1)

    @property
    def atom_dim(self):
        return sum(sz + 1 for sz in self.atom_sizes if sz > 1)

    @property
    def edge_dim(self):
        return sum(sz + 1 for sz in self.edge_sizes if sz > 1)


class GINEConv(nn.Module):
    """GINE convolution with edge conditioning and mean normalization."""
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_lin = nn.Linear(edge_dim, hidden)

    def forward(self, x, edge_index, edge_attr, batch):
        src, dst = edge_index[0], edge_index[1]
        h = self.mlp(x)
        m = h[src] + self.edge_lin(edge_attr)
        # sum aggregate
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        # mean-normalize per node (approximate: divide by in-degree)
        in_deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        in_deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        in_deg = in_deg.clamp(min=1.0)
        return x + agg / in_deg.unsqueeze(-1)


class Model(nn.Module):
    def __init__(self, hidden, n_layers, out_dim=1):
        super().__init__()
        self.input_embed = InputEmbedding()
        self.node_lin = nn.Linear(self.input_embed.atom_dim, hidden)
        self.edge_lin_init = nn.Linear(self.input_embed.edge_dim, hidden)
        self.bn0 = nn.LayerNorm(hidden)
        self.convs = nn.ModuleList(
            [GINEConv(hidden, self.input_embed.edge_dim) for _ in range(n_layers)]
        )
        self.bn1 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_raw, edge_index, batch, edge_attr_raw, n_graphs):
        x = self.node_lin(self.input_embed.forward_atom(x_raw))
        x = F.relu(self.bn0(x))
        edge_attr = self.input_embed.forward_edge(edge_attr_raw)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr, batch))
        # sum readout
        pool = torch.zeros(n_graphs, x.size(1), device=x.device).index_add_(0, batch, x)
        return self.head(pool)


def slice_batch(indices, nptr, eptr, X, EI, EA):
    xs, eis, bts, eas = [], [], [], []
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


def train_epoch(model, train_idx, opt, device, nptr, eptr, X, EI, EA, batch_size):
    model.train()
    perm = torch.randperm(len(train_idx)).tolist()
    total, nsteps = 0.0, 0
    n = len(train_idx)
    for s in range(0, n, batch_size):
        sel = [train_idx[j] for j in perm[s:s + batch_size]]
        x, ei, bt, ea = slice_batch(sel, nptr, eptr, X, EI, EA)
        y = None  # set by caller via closure or return
        yield sel, x, ei, bt, ea


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} hidden={HIDDEN} layers={LAYERS} epochs={EPOCHS} "
          f"lr={LR} batch={BATCH} seed={SEED}", flush=True)

    t0 = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    sizes = d["split_sizes"]
    nptr, eptr = d["node_ptr"], d["edge_ptr"]
    X, EI, EA = d["x"], d["edge_index"], d["edge_attr"]
    out_dim = int(d["y"].size(1))
    train_idx = list(range(sizes["train"]))
    val_idx = list(range(sizes["train"], sizes["train"] + sizes["val"]))
    target = val_idx if a.split == "val" else list(range(sizes["train"] + sizes["val"], d["num_graphs"]))
    print(f"Loaded in {time.time()-t0:.1f}s, {d['num_graphs']} graphs, "
          f"train={sizes['train']} val={sizes['val']} test={sizes['test']}", flush=True)

    model = Model(HIDDEN, LAYERS, out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    from sklearn.metrics import roc_auc_score

    val_y = d["y"][sizes["train"]:sizes["train"] + sizes["val"]].squeeze(-1)
    val_mask = torch.isfinite(val_y)
    val_y_f = val_y[val_mask].numpy()
    val_idx_f = [val_idx[i] for i in range(len(val_idx)) if val_mask[i]]

    def predict(indices, batch_size=512):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                chunk = indices[i:i + batch_size]
                x, ei, bt, ea = slice_batch(chunk, nptr, eptr, X, EI, EA)
                x, ei, bt, ea = x.to(device), ei.to(device), bt.to(device), ea.to(device)
                out = model(x, ei, bt, ea, len(chunk))
                preds.append(torch.sigmoid(out).float().cpu())
        return torch.cat(preds)

    best_val_auc = -1.0
    best_state = None
    bad = 0
    for ep in range(EPOCHS):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(len(train_idx)).tolist()
        n = len(train_idx)
        total_loss, nsteps = 0.0, 0
        for s in range(0, n, BATCH):
            sel = [train_idx[j] for j in perm[s:s + BATCH]]
            x, ei, bt, ea = slice_batch(sel, nptr, eptr, X, EI, EA)
            x, ei, bt, ea = x.to(device), ei.to(device), bt.to(device), ea.to(device)
            y = d["y"][sel].float().to(device)
            logit = model(x, ei, bt, ea, len(sel))
            if POS_W > 0:
                pos_w = torch.full_like(logit, POS_W)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
            else:
                loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_loss += loss.item() * len(sel)
            nsteps += len(sel)
        sched.step()

        val_str = ""
        if (ep + 1) % EVAL_EVERY == 0:
            vp = predict(val_idx_f).squeeze(-1).numpy()
            if np.isfinite(vp).all():
                val_auc = float(roc_auc_score(val_y_f, vp))
                val_str = f"  val_auc={val_auc:.4f}"
                if val_auc > best_val_auc + 1e-5:
                    best_val_auc = val_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += EVAL_EVERY
                if bad >= PATIENCE:
                    print(f"  early stop at epoch {ep+1} (best {best_val_auc:.4f})", flush=True)
                    break
        print(f"epoch {ep+1}/{EPOCHS} loss={total_loss/max(1,nsteps):.4f} "
              f"lr={opt.param_groups[0]['lr']:.5f} time={time.time()-t_ep:.1f}s{val_str}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Restored best-val model (auc={best_val_auc:.4f})", flush=True)

    t0 = time.time()
    preds = predict(target)
    torch.save({"pred": preds}, Path(a.out))
    print(f"Predicted {len(target)} {a.split} graphs {tuple(preds.shape)} in {time.time()-t0:.1f}s -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
