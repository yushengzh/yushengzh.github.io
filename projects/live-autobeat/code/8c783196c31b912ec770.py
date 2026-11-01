#!/usr/bin/env python3
"""Weak baseline, graph-level prediction: 2-layer GIN-lite with mean pooling.

The graph collection is stored in a compact form: every graph's nodes and edges
are concatenated into flat arrays, with `node_ptr` / `edge_ptr` giving the slice
for each graph. This keeps 158k-graph datasets (ogbg-ppa) inside memory.

    graph i -> nodes [node_ptr[i], node_ptr[i+1])
               edges [edge_ptr[i], edge_ptr[i+1])

Graph index ranges per split come from `split_sizes`: train, then val, then test.
Each epoch trains on a subsample, because the graph counts are large.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 64))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 5))
LR = float(os.environ.get("SOLVER_LR", 0.01))
PER_EPOCH = int(os.environ.get("SOLVER_PER_EPOCH", 4000))
BATCH = int(os.environ.get("SOLVER_BATCH", 128))


def slice_graph(d, i: int) -> dict:
    n0, n1 = int(d["node_ptr"][i]), int(d["node_ptr"][i + 1])
    e0, e1 = int(d["edge_ptr"][i]), int(d["edge_ptr"][i + 1])
    g = {
        "x": d["x"][n0:n1],
        "edge_index": (d["edge_index"][:, e0:e1].long() - n0),
        "y": d["y"][i],
    }
    if "edge_attr" in d:
        g["edge_attr"] = d["edge_attr"][e0:e1]
    return g


def collate(graphs):
    xs, eis, batch, ys = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        eis.append(g["edge_index"] + offset)
        batch.append(torch.full((g["x"].size(0),), i, dtype=torch.long))
        ys.append(g["y"])
        offset += g["x"].size(0)
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(batch), torch.stack(ys)


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = torch.nn.Linear(in_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch, n_graphs):
        row, col = edge_index
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h + torch.zeros_like(h).index_add_(0, row, h[col])))
        h = h + torch.zeros_like(h).index_add_(0, row, h[col])
        pooled = torch.zeros(n_graphs, h.size(1)).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        return self.head(pooled / counts)


def masked_loss(logits, y):
    nan = torch.isnan(y)
    if nan.any():
        logits, y = logits[~nan], y[~nan]
    if y.numel() == 0:
        return logits.sum() * 0.0
    if logits.dim() == 1 or logits.size(1) == 1:
        return F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1).float())
    return F.binary_cross_entropy_with_logits(logits, y.float())


@torch.no_grad()
def predict(model, d, indices):
    model.eval()
    preds = []
    for i in range(0, len(indices), BATCH):
        chunk = [slice_graph(d, j) for j in indices[i:i + BATCH]]
        x, ei, batch, _ = collate(chunk)
        preds.append(torch.sigmoid(model(x, ei, batch, len(chunk))))
    return torch.cat(preds) if preds else torch.zeros(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    d = torch.load(Path(a.data_dir) / "dataset.pt")
    sizes = d["split_sizes"]
    tr = list(range(sizes["train"]))
    target = (list(range(sizes["train"], sizes["train"] + sizes["val"]))
              if a.split == "val"
              else list(range(sizes["train"] + sizes["val"], d["num_graphs"])))
    out_dim = int(d["y"].size(1))

    torch.manual_seed(0)
    model = GIN(d["x"].size(1), HIDDEN, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    g = torch.Generator().manual_seed(0)
    for ep in range(EPOCHS):
        idx = torch.randperm(len(tr), generator=g)[:min(PER_EPOCH, len(tr))]
        idx = [tr[int(j)] for j in idx]
        model.train()
        total = 0.0
        for i in range(0, len(idx), BATCH):
            chunk = [slice_graph(d, j) for j in idx[i:i + BATCH]]
            x, ei, batch, y = collate(chunk)
            opt.zero_grad()
            loss = masked_loss(model(x, ei, batch, len(chunk)), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(chunk)
        print(f"epoch {ep + 1}/{EPOCHS} train_loss={total / max(1, len(idx)):.4f}",
              flush=True)

    torch.save({"pred": predict(model, d, target)}, Path(a.out))
    print(f"wrote predictions for {len(target)} {a.split} graphs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
