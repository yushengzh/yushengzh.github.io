#!/usr/bin/env python3
"""Weak baseline, node classification: 2-layer GCN over the provided graph.

Neighbourhood aggregation is chunked: these graphs do not fit an [E, H]
intermediate tensor.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 32))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 30))
LR = float(os.environ.get("SOLVER_LR", 0.01))
CHUNK = int(os.environ.get("SOLVER_CHUNK", 1_000_000))


def normalise(edge_index, n):
    row, col = edge_index
    deg = torch.zeros(n, dtype=torch.float).index_add_(
        0, row, torch.ones(row.numel(), dtype=torch.float))
    dinv = deg.clamp(min=1).pow(-0.5)
    return row, col, dinv[row] * dinv[col]


def aggregate(h, row, col, w):
    out = torch.zeros_like(h)
    for i in range(0, row.numel(), CHUNK):
        r = row[i:i + CHUNK]
        out.index_add_(0, r, h[col[i:i + CHUNK]] * w[i:i + CHUNK].unsqueeze(-1))
    return out


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, row, col, w):
        h = F.relu(aggregate(self.l1(x), row, col, w))
        h = F.dropout(h, p=0.5, training=self.training)
        return aggregate(self.l2(h), row, col, w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    d = torch.load(Path(a.data_dir) / "dataset.pt")
    x, y = d["x"].float(), d["y"]
    tr, va = d["train_idx"], d["val_idx"]
    target = va if a.split == "val" else d["test_idx"]
    row, col, w = normalise(d["edge_index"], x.size(0))

    torch.manual_seed(0)
    model = GCN(x.size(1), HIDDEN, int(d["num_classes"]))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    best_acc, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, row, col, w)[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, row, col, w)[va].argmax(1) == y[va]).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1}/{EPOCHS} loss={loss.item():.4f} val_acc={acc:.4f}",
              flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, row, col, w)[target].argmax(1)

    torch.save({"pred": pred}, Path(a.out))
    print(f"wrote {pred.numel()} {a.split} predictions -> {a.out} "
          f"(best val acc {best_acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
