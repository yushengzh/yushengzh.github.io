#!/usr/bin/env python3
"""GCN for ogbn-mag node classification using GPU sparse.mm propagation.

Fast: full training loop on GPU via torch.sparse_coo_tensor.mm().
Multi-seed logit ensemble with early stopping on validation CE.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = 64
EPOCHS = 100
LR = 0.01
WD = 5e-4
SEEDS = [0, 1, 2]
DROP = 0.1
PATIENCE = 20


def build_sparse_adj(edge_index, n):
    """Build a sparse adjacency matrix with self-loops and D^{-1/2} A D^{-1/2} normalization."""
    from torch_geometric.utils import add_self_loops
    ei = add_self_loops(edge_index, fill_value=1.0, num_nodes=n)[0]
    row, col = ei[0], ei[1]
    deg = torch.zeros(n, dtype=torch.float32)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float32))
    dinv = deg.clamp(min=1).pow(-0.5)
    w = (dinv[row] * dinv[col]).float()
    S = torch.sparse_coo_tensor(ei, w, (n, n))
    return S


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, S):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_dim)
        self.S = S  # sparse adj (device set externally)

    def forward(self, x):
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=DROP, training=self.training)
        h = self.S.mm(h)
        h = F.relu(self.lin2(h))
        h = F.dropout(h, p=DROP, training=self.training)
        h = self.S.mm(h)
        return h


def train_one_seed(x, y, tr, va, S, num_classes, device, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GCN(x.size(1), HIDDEN, num_classes, S).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
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
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(x)[va]
            val_loss = F.cross_entropy(val_logits, y[va]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
        del logits, val_logits, loss
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    x = d["x"].float()
    y = d["y"]
    n = x.size(0)
    num_classes = int(d["num_classes"])
    tr, va = d["train_idx"], d["val_idx"]
    target = va if a.split == "val" else d["test_idx"]

    # Standardize features
    mu = x.mean(0)
    sd = x.std(0).clamp(min=1e-6)
    x = (x - mu) / sd

    # Build sparse adjacency (on GPU)
    S = build_sparse_adj(d["edge_index"], n).to(device)
    x = x.to(device)
    y = y.to(device)
    tr = tr.to(device)
    va = va.to(device)
    target = target.to(device)

    models = []
    for seed in SEEDS:
        print(f"Training seed {seed}...", flush=True)
        m = train_one_seed(x, y, tr, va, S, num_classes, device, seed)
        models.append(m)
        torch.cuda.empty_cache()

    # Ensemble: average logits
    target_dev = target
    acc = torch.zeros(len(target_dev), num_classes, device=device)
    with torch.no_grad():
        for m in models:
            m.eval()
            acc += m(x)[target_dev]
        acc /= len(models)
    pred = acc.argmax(1).cpu()
    torch.save({"pred": pred}, Path(a.out))
    print(f"Wrote {pred.numel()} {a.split} predictions -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
