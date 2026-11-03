#!/usr/bin/env python3
"""Weak baseline, link prediction: GCN encoder + dot-product decoder.

Test pairs arrive at grading time in <data>/test_inputs.pt as a shuffled mix of
positives and negatives; we score every row without knowing which is which.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 32))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 10))
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


class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)

    def forward(self, x, row, col, w):
        h = F.relu(aggregate(self.l1(x), row, col, w))
        return aggregate(self.l2(h), row, col, w)


def dot(z, edges, chunk=2_000_000):
    out = []
    for i in range(0, edges.size(1), chunk):
        e = edges[:, i:i + chunk]
        out.append((z[e[0]] * z[e[1]]).sum(-1))
    return torch.cat(out) if out else torch.zeros(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    d = torch.load(data_dir / "dataset.pt")
    x = d["x"].float()
    row, col, w = normalise(d["edge_index"], x.size(0))
    tr_pos, tr_neg = d["train_edges"], d["train_neg"]

    torch.manual_seed(0)
    enc = Encoder(x.size(1), HIDDEN)
    opt = torch.optim.Adam(enc.parameters(), lr=LR, weight_decay=1e-5)

    for ep in range(EPOCHS):
        enc.train()
        opt.zero_grad()
        z = enc(x, row, col, w)
        pos, neg = dot(z, tr_pos), dot(z, tr_neg)
        logits = torch.cat([pos, neg])
        target = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        opt.step()
        print(f"epoch {ep + 1}/{EPOCHS} loss={loss.item():.4f}", flush=True)

    enc.eval()
    with torch.no_grad():
        z = enc(x, row, col, w)
        if a.split == "val":
            # candidate order fixed by the task (see instruction.md)
            pos, neg = d["val_edges"], d["val_neg"]
            if neg.dim() == 3:   # grouped [V, K, 2]: per query, positive then its K negatives
                cand = torch.cat([pos.t().unsqueeze(1), neg], dim=1).reshape(-1, 2).t()
            else:                # flat [2, V']
                cand = torch.cat([pos, neg], dim=1)
            cand = cand.long()
        else:
            ti_path = data_dir / "test_inputs.pt"
            if not ti_path.exists():
                raise SystemExit(f"missing {ti_path}: the grader reveals test inputs here")
            cand = torch.load(ti_path)["candidates"].long()
        scores = dot(z, cand)

    torch.save({"scores": scores}, Path(a.out))
    print(f"wrote {scores.numel()} {a.split} edge scores -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
