#!/usr/bin/env python3
"""Improved graph-level prediction: 4-layer GIN with edge features, skip connections, batch norm.

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

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 128))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 30))
LR = float(os.environ.get("SOLVER_LR", 0.001))
PER_EPOCH = int(os.environ.get("SOLVER_PER_EPOCH", 8000))
BATCH = int(os.environ.get("SOLVER_BATCH", 64))


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
    xs, eis, batch, ys, eas = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        eis.append(g["edge_index"] + offset)
        batch.append(torch.full((g["x"].size(0),), i, dtype=torch.long))
        ys.append(g["y"])
        if "edge_attr" in g:
            eas.append(g["edge_attr"])
        offset += g["x"].size(0)
    ea = torch.cat(eas) if eas else None
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(batch), torch.stack(ys), ea


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, edge_dim=None):
        super().__init__()
        self.hidden = hidden
        self.has_edges = edge_dim is not None

        self.l1 = torch.nn.Linear(in_dim, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.l3 = torch.nn.Linear(hidden, hidden)
        self.bn3 = torch.nn.BatchNorm1d(hidden)
        self.l4 = torch.nn.Linear(hidden, hidden)
        self.bn4 = torch.nn.BatchNorm1d(hidden)

        if self.has_edges:
            self.edge_proj = torch.nn.Linear(edge_dim, hidden)

        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch, n_graphs, edge_attr=None):
        row, col = edge_index

        h = self.l1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h0 = h
        h = self.dropout(h)

        if self.has_edges and edge_attr is not None:
            e = F.relu(self.edge_proj(edge_attr))
            msg = h[col] * e
        else:
            msg = h[col]

        h = h + torch.zeros_like(h).index_add_(0, row, msg)
        h = self.l2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h1 = h + h0
        h = self.dropout(h)

        if self.has_edges and edge_attr is not None:
            msg = h[col] * e
        else:
            msg = h[col]

        h = h + torch.zeros_like(h).index_add_(0, row, msg)
        h = self.l3(h)
        h = self.bn3(h)
        h = F.relu(h)
        h2 = h + h1
        h = self.dropout(h)

        msg = h[col]
        h = h + torch.zeros_like(h).index_add_(0, row, msg)
        h = self.l4(h)
        h = self.bn4(h)
        h = F.relu(h)
        h = h + h2

        pooled = torch.zeros(n_graphs, self.hidden).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pooled = pooled / counts

        return self.head(pooled)


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
        x, ei, batch, _, ea = collate(chunk)
        preds.append(torch.sigmoid(model(x, ei, batch, len(chunk), ea)))
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
    edge_dim = int(d["edge_attr"].size(1)) if "edge_attr" in d else None

    torch.manual_seed(0)
    model = GIN(d["x"].size(1), HIDDEN, out_dim, edge_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    g = torch.Generator().manual_seed(0)
    for ep in range(EPOCHS):
        idx = torch.randperm(len(tr), generator=g)[:min(PER_EPOCH, len(tr))]
        idx = [tr[int(j)] for j in idx]
        model.train()
        total = 0.0
        for i in range(0, len(idx), BATCH):
            chunk = [slice_graph(d, j) for j in idx[i:i + BATCH]]
            x, ei, batch, y, ea = collate(chunk)
            opt.zero_grad()
            loss = masked_loss(model(x, ei, batch, len(chunk), ea), y)
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
