#!/usr/bin/env python3
"""GPU GCN for ogbn-mag node classification, multi-seed logit ensemble.

Memory-safe: standardize features, chunked scatter-aggregate, low HIDDEN,
expandable segments. Exact CLI contract preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = 32
EPOCHS = 60
LR = 0.01
WD = 5e-4
SEEDS = [0]
DROP = 0.1
CHUNK = 500_000


def build_graph(edge_index, n, device):
    from torch_geometric.utils import add_self_loops
    edge_index = edge_index.to(device)
    edge_index, _ = add_self_loops(edge_index, fill_value=1.0, num_nodes=n)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(n, device=device, dtype=torch.float32)
    deg.index_add_(0, row, torch.ones(row.numel(), device=device, dtype=torch.float32))
    dinv = deg.clamp(min=1).pow(-0.5)
    w = (dinv[row] * dinv[col]).float()
    return edge_index, w


def propagate(h, edge_index, w):
    row, col = edge_index
    out = torch.zeros_like(h)
    for i in range(0, row.numel(), CHUNK):
        r = row[i:i + CHUNK]
        c = col[i:i + CHUNK]
        wt = w[i:i + CHUNK].unsqueeze(-1)
        out.index_add_(0, r, h[c] * wt)
    return out


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = torch.nn.Linear(in_dim, hidden)
        self.conv2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, w):
        h = F.relu(self.conv1(x))
        h = F.dropout(h, p=DROP, training=self.training)
        h = propagate(h, edge_index, w)
        h = F.relu(self.conv2(h))
        h = F.dropout(h, p=DROP, training=self.training)
        out = propagate(h, edge_index, w)
        return out


def train_one_seed(x, y, tr, va, edge_index, w, num_classes, device, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GCN(x.size(1), HIDDEN, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best = float('inf')
    best_state = None
    no_improve = 0
    x_dev, y_dev = x.to(device), y.to(device)
    tr_dev, va_dev = tr.to(device), va.to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_dev, edge_index, w)[tr_dev]
        loss = F.cross_entropy(logits, y_dev[tr_dev])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_dev, edge_index, w)[va_dev]
            val_loss = F.cross_entropy(val_logits, y_dev[va_dev]).item()
        del logits, val_logits, loss
        if val_loss < best:
            best = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 15:
                break
    # reload best state into a fresh module
    final = GCN(x.size(1), HIDDEN, num_classes).to(device)
    if best_state is not None:
        final.load_state_dict(best_state)
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    x = d["x"].float().cpu()
    y = d["y"]
    n = x.size(0)
    tr, va = d["train_idx"], d["val_idx"]
    target = va if a.split == "val" else d["test_idx"]
    num_classes = int(d["num_classes"])

    # standardize features (computed on the visible input; generalizes)
    mu = x.mean(0)
    sd = x.std(0).clamp(min=1e-6)
    x = (x - mu) / sd

    edge_index, w = build_graph(d["edge_index"], n, device)
    models = []
    for seed in SEEDS:
        print(f"Training seed {seed}...", flush=True)
        m = train_one_seed(x, y, tr, va, edge_index, w, num_classes, device, seed)
        models.append(m)
        torch.cuda.empty_cache()

    x_dev = x.to(device)
    target_dev = target.to(device)
    acc = torch.zeros(len(target), num_classes, device=device)
    with torch.no_grad():
        for m in models:
            m.eval()
            acc += m(x_dev, edge_index, w)[target_dev]
        acc /= len(models)
    pred = acc.argmax(1)
    torch.save({"pred": pred.cpu()}, Path(a.out))
    print(f"Wrote {pred.numel()} {a.split} predictions -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
