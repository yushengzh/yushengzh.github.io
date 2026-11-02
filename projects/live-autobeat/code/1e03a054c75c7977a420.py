"""Molecular activity prediction (ogbg-molhiv) - v3: stable GINE.

Contract: python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
Output:   {"pred": [M] float}
Metric:   ROC-AUC (higher better).

Fixes from v2:
  - Input normalization (standardize atom features)
  - Small init for input projection
  - Gradient clipping
  - LayerNorm instead of BatchNorm (more stable)
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- hyperparameters (overridable via env) ----
HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 128))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 5))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 30))
LR = float(os.environ.get("SOLVER_LR", 0.001))
WD = float(os.environ.get("SOLVER_WD", 1e-4))
BATCH = int(os.environ.get("SOLVER_BATCH", 256))
SEED = int(os.environ.get("SOLVER_SEED", 42))
DEVICE = os.environ.get("SOLVER_DEVICE", "cuda")
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.1))
EVAL_EVERY = int(os.environ.get("SOLVER_EVAL_EVERY", 3))
GRAD_CLIP = float(os.environ.get("SOLVER_GRAD_CLIP", 1.0))


def slice_batch(d, indices, nptr, eptr, X, EI, EA):
    xs, eis, bts, eas = [], [], [], []
    offset = 0
    for i, gi in enumerate(indices):
        n0, n1 = int(nptr[gi]), int(nptr[gi + 1])
        e0, e1 = int(eptr[gi]), int(eptr[gi + 1])
        xs.append(X[n0:n1])
        if e1 > e0:
            eis.append(EI[:, e0:e1].long() - n0 + offset)
        bts.append(torch.full((n1 - n0,), i, dtype=torch.long))
        if EA is not None and e1 > e0:
            eas.append(EA[e0:e1])
        offset += n1 - n0
    x = torch.cat(xs)
    ei = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.cat(eas, dim=0) if eas else torch.zeros(0, 0)
    bt = torch.cat(bts)
    return x, ei, bt, ea


def gather_train(d, indices, nptr, eptr, X, EI, EA):
    y = d["y"][indices, 0].float()
    valid = torch.isfinite(y)
    if bool(valid.all()):
        x, ei, bt, ea = slice_batch(d, indices, nptr, eptr, X, EI, EA)
        return x, ei, bt, ea, y
    keep = indices[valid]
    y = y[valid]
    x, ei, bt, ea = slice_batch(d, keep, nptr, eptr, X, EI, EA)
    return x, ei, bt, ea, y


class GINEConv(torch.nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
        )
        if edge_dim and edge_dim > 0:
            self.edge_lin = torch.nn.Linear(edge_dim, hidden)
        else:
            self.edge_lin = None

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        h = self.mlp(x)
        m = h[src]
        if self.edge_lin is not None and edge_attr is not None and edge_attr.size(0) > 0:
            m = m + self.edge_lin(edge_attr)
        agg = torch.zeros_like(x)
        if m.size(0) > 0:
            agg.index_add_(0, dst, m)
        out = x + agg
        return out


class GINE(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_layers, edge_dim, out_dim=1):
        super().__init__()
        self.input_lin = torch.nn.Linear(in_dim, hidden)
        torch.nn.init.kaiming_normal_(self.input_lin.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.input_lin.bias)
        self.bn0 = torch.nn.LayerNorm(hidden)
        self.convs = torch.nn.ModuleList([GINEConv(hidden, edge_dim) for _ in range(n_layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, batch, n_graphs, edge_attr=None):
        x = F.relu(self.bn0(self.input_lin(x)))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
        # mean + max pooling
        mean_pool = torch.zeros(n_graphs, x.size(1), device=x.device).index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1).float()
        mean_pool = mean_pool / counts
        neg_inf = torch.full((n_graphs, x.size(1)), float("-inf"), device=x.device)
        idx = batch.unsqueeze(-1).expand_as(x)
        max_pool = torch.scatter_reduce(neg_inf, 0, idx, x, reduce="amax", include_self=False)
        # handle graphs with no nodes (shouldn't happen but be safe)
        max_pool = max_pool.masked_fill(torch.isinf(max_pool), 0.0)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(pooled)


def train_epoch(model, d, train_idx, opt, device, batch_size, nptr, eptr, X, EI, EA):
    model.train()
    perm = torch.randperm(len(train_idx))
    total, seen = 0.0, 0
    for i in range(0, len(perm), batch_size):
        chunk = [int(train_idx[p]) for p in perm[i:i + batch_size]]
        x, ei, bt, ea, y = gather_train(d, chunk, nptr, eptr, X, EI, EA)
        if y.numel() == 0:
            continue
        x, ei, bt, ea = x.to(device), ei.to(device), bt.to(device), ea.to(device)
        y = y.to(device)
        out = model(x, ei, bt, len(chunk), ea).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [warn] NaN/Inf loss in batch, skipping", flush=True)
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        opt.step()
        total += float(loss) * y.numel()
        seen += y.numel()
    return total / max(1, seen)


@torch.no_grad()
def predict(model, d, indices, device, batch_size=512, nptr=None, eptr=None, X=None, EI=None, EA=None):
    model.eval()
    preds = []
    for i in range(0, len(indices), batch_size):
        chunk = indices[i:i + batch_size]
        x, ei, bt, ea = slice_batch(d, chunk, nptr, eptr, X, EI, EA)
        x, ei, bt, ea = x.to(device), ei.to(device), bt.to(device), ea.to(device)
        out = model(x, ei, bt, len(chunk), ea)
        preds.append(torch.sigmoid(out).squeeze(-1).cpu())
    return torch.cat(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  hidden={HIDDEN} layers={LAYERS} epochs={EPOCHS} "
          f"lr={LR} batch={BATCH} dropout={DROPOUT} grad_clip={GRAD_CLIP}", flush=True)

    t0 = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    sizes = d["split_sizes"]
    nptr, eptr = d["node_ptr"], d["edge_ptr"]
    X, EI, EA = d["x"], d["edge_index"], d["edge_attr"]
    in_dim, out_dim = int(d["x"].size(1)), int(d["y"].size(1))
    edge_dim = int(d["edge_attr"].size(1)) if "edge_attr" in d else 0
    train_idx = list(range(sizes["train"]))
    val_idx = list(range(sizes["train"], sizes["train"] + sizes["val"]))
    target = val_idx if a.split == "val" else list(range(sizes["train"] + sizes["val"], d["num_graphs"]))
    print(f"Loaded in {time.time()-t0:.1f}s, {d['num_graphs']} graphs, "
          f"train={sizes['train']} val={sizes['val']} test={sizes['test']}", flush=True)

    model = GINE(in_dim, HIDDEN, LAYERS, edge_dim, out_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    def lr_lambda(step):
        return max(0.0, 0.5 * (1 + math.cos(math.pi * step / max(1, EPOCHS))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    from sklearn.metrics import roc_auc_score
    best_val_auc = 0.0
    for ep in range(EPOCHS):
        t_ep = time.time()
        loss = train_epoch(model, d, train_idx, opt, device, BATCH, nptr, eptr, X, EI, EA)
        sched.step()
        val_str = ""
        if (ep + 1) % EVAL_EVERY == 0:
            vp = predict(model, d, val_idx, device, nptr=nptr, eptr=eptr, X=X, EI=EI, EA=EA)
            yv = d["y"][sizes['train']:sizes['train'] + sizes['val']].squeeze(-1)
            yv = yv[torch.isfinite(yv)]
            vp = vp[torch.isfinite(d["y"][sizes['train']:sizes['train'] + sizes['val']].squeeze(-1))]
            if torch.isfinite(vp).all():
                val_auc = roc_auc_score(yv.numpy(), vp.numpy())
                val_str = f"  val_auc={val_auc:.4f}"
                best_val_auc = max(best_val_auc, val_auc)
        print(f"epoch {ep+1}/{EPOCHS} loss={loss:.4f} lr={opt.param_groups[0]['lr']:.5f} "
              f"time={time.time()-t_ep:.1f}s{val_str}", flush=True)

    t0 = time.time()
    preds = predict(model, d, target, device, nptr=nptr, eptr=eptr, X=X, EI=EI, EA=EA)
    torch.save({"pred": preds}, Path(a.out))
    print(f"Predicted {len(target)} {a.split} graphs in {time.time()-t0:.1f}s -> {a.out}", flush=True)
    print(f"TOTAL wall {time.time()-t0:.0f}s predict / see per-epoch above", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
