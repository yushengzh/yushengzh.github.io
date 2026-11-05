#!/usr/bin/env python3
"""3-layer GCN for ogbn-mag node classification with symmetrized edges.

Uses GPU sparse.mm for fast propagation. No PyG dependency.
Multi-seed logit ensemble with early stopping on validation CE.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


# ── hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN = 256
DEPTH  = 3
EPOCHS = 150
LR     = 0.01
WD     = 5e-4
SEEDS  = [0, 1, 2, 3, 4]
DROP   = 0.1
PATIENCE = 30


def build_S_sym(edge_index: torch.Tensor, n: int, device: str) -> torch.Tensor:
    """Build sparse D^{-1/2} A D^{-1/2} adjacency with self-loops, symmetrized, no PyG."""
    ei = edge_index.to(device)
    # Symmetrize: add reverse edges
    ei_rev = ei.flip(0)  # [col, row] for each [row, col]
    ei = torch.cat([ei, ei_rev], dim=1)
    # Self-loops
    self_loops = torch.arange(n, device=device).unsqueeze(0).expand(2, -1)
    ei = torch.cat([ei, self_loops], dim=1).to(torch.int64)
    row, col = ei[0], ei[1]
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float32, device=device))
    dinv = deg.clamp(min=1).pow(-0.5)
    w = dinv[row] * dinv[col]
    S = torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()
    return S


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, S, depth=3):
        super().__init__()
        self.S = S
        self.depth = depth
        lins = [torch.nn.Linear(in_dim, hidden)]
        for _ in range(depth - 2):
            lins.append(torch.nn.Linear(hidden, hidden))
        lins.append(torch.nn.Linear(hidden, out_dim))
        self.lins = torch.nn.ModuleList(lins)

    def forward(self, x):
        h = x
        for i in range(self.depth):
            h = self.lins[i](h)
            if i < self.depth - 1:
                h = F.relu(h)
                h = F.dropout(h, p=DROP, training=self.training)
                h = self.S.mm(h)
        return h


def train_one_seed(x, y, tr, va, S, num_classes, device, seed,
                   hidden=HIDDEN, depth=DEPTH, epochs=EPOCHS,
                   lr=LR, wd=WD, patience=PATIENCE):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GCN(x.size(1), hidden, num_classes, S, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)[tr]
        loss = F.cross_entropy(logits, y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(x)[va], y[va]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        del logits, loss
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    d = torch.load(Path(a.data_dir, "dataset.pt"), map_location="cpu")
    x, y = d["x"].float(), d["y"]
    n, num_classes = x.size(0), int(d["num_classes"])
    tr, va = d["train_idx"], d["val_idx"]
    target = va if a.split == "val" else d["test_idx"]

    device = "cuda"
    mu = x.mean(0)
    sd = x.std(0).clamp(min=1e-6)
    x = ((x - mu) / sd).to(device)
    y = y.to(device)
    tr = tr.to(device)
    va = va.to(device)
    target = target.to(device)

    S = build_S_sym(d["edge_index"], n, device)
    print(f"N={n} F={x.size(1)} E={d['edge_index'].size(1)} C={num_classes} (symmetrized)", flush=True)

    models = []
    for seed in SEEDS:
        print(f"Training seed {seed}...", flush=True)
        m = train_one_seed(x, y, tr, va, S, num_classes, device, seed)
        models.append(m)
        torch.cuda.empty_cache()

    # Ensemble: average logits over seeds
    logit_sum = torch.zeros(len(target), num_classes, device=device)
    with torch.no_grad():
        for m in models:
            m.eval()
            logit_sum += m(x)[target]
    logit_avg = logit_sum / len(models)
    pred = logit_avg.argmax(dim=1).cpu()

    torch.save({"pred": pred}, Path(a.out))
    print(f"Wrote {pred.numel()} {a.split} predictions -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
