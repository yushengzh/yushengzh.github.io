#!/usr/bin/env python3
"""GIN solver for ogbg-molesol aqueous solubility regression.

Architecture:
  - Per-feature one-hot embeddings for 9 atom features and 3 edge features
  - 6 GIN layers with learnable epsilon, edge-conditioned aggregation
  - Virtual node for global context
  - Mean+max+sum pooling head
  - AdamW + cosine LR, early stop on val RMSE
  - 5-seed ensemble

Contract: python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = int(os.environ.get("SOLVER_SEEDS", 5))
HID = int(os.environ.get("SOLVER_HID", 256))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 6))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 1200))
LR = float(os.environ.get("SOLVER_LR", 1e-3))
WD = float(os.environ.get("SOLVER_WD", 1e-4))
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.1))
PATIENCE = int(os.environ.get("SOLVER_PATIENCE", 120))
GRAD_CLIP = 5.0
DESC = int(os.environ.get("SOLVER_DESC", 0))  # 1 = augment head with graph-level descriptors
# descriptor dim: 1(nc)+1(ec)+2(deg,degmax)+9(atom col means)+3(edge col means)=16
DESC_DIM = 16

# Atom feature vocabs (from data: col0 max=52, col2 max=4, col3 max=6, col4 max=4, col6 max=2, col7 max=1, col8 max=1)
# Cols 1 and 5 are always 0 (dead)
ATOM_VOCABS = [64, 2, 5, 7, 5, 2, 3, 2, 2]  # index = column index
EDGE_VOCABS = [4, 3, 2]  # edge col max values


class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim, layers=2, dropout=0.0):
        super().__init__()
        dims = [in_dim] + [hid] * (layers - 1) + [out_dim]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class GIN(nn.Module):
    def __init__(self, hid, out_dim, layers=6, dropout=0.1):
        super().__init__()
        # Per-feature one-hot embeddings
        self.atom_embeds = nn.ModuleList(
            [nn.Embedding(v, hid) for v in ATOM_VOCABS]
        )
        # Per-feature edge embeddings
        self.edge_embeds = nn.ModuleList(
            [nn.Embedding(v, hid) for v in EDGE_VOCABS]
        )
        # Per-layer: conv (atom), edge bias MLP, node MLP
        self.conv = nn.ModuleList()
        self.edge_mlp = nn.ModuleList()
        self.node_mlp = nn.ModuleList()
        for _ in range(layers):
            self.conv.append(nn.Linear(hid, hid))
            self.edge_mlp.append(MLP(hid, hid, hid, 2, dropout))
            self.node_mlp.append(MLP(hid, hid, hid, 2, dropout))
        for i in range(layers):
            self.register_parameter(f"eps{i}", nn.Parameter(torch.zeros(1)))
        # Virtual node
        self.virtual = nn.Parameter(torch.randn(1, hid) * 0.02)
        # Pooling head
        self.pool_mlp = MLP(hid * 3, hid, hid, 2, dropout)
        self.head = MLP(hid + DESC_DIM * DESC, hid, out_dim, 2, dropout)

    def _atom_embedding(self, x):
        # x: [N, 9] float (integer values)
        parts = []
        for i, emb in enumerate(self.atom_embeds):
            parts.append(emb(x[:, i].long()))
        return torch.stack(parts, dim=1).sum(dim=1)  # [N, hid]

    def _edge_embedding(self, edge_attr):
        # edge_attr: [E, 3] float (integer values)
        parts = []
        for i, emb in enumerate(self.edge_embeds):
            parts.append(emb(edge_attr[:, i].long()))
        return torch.stack(parts, dim=1).sum(dim=1)  # [E, hid]

    def _graph_desc(self, x, edge_index, edge_attr, batch, n_graphs):
        dev = x.device
        cnt = torch.bincount(batch, minlength=n_graphs).float().clamp(min=1)
        D = [torch.zeros(n_graphs, device=dev)]
        # node count
        D[0] = torch.zeros(n_graphs, device=dev).index_add_(0, batch, torch.ones(x.size(0), device=dev))
        # edge count
        if edge_index.size(1) > 0:
            D[0].index_add_(0, batch, torch.ones(edge_index.size(1), device=dev) * 0)  # no-op safety
            ecnt = torch.zeros(n_graphs, device=dev)
            ecnt.index_add_(0, batch, torch.ones(edge_index.size(1), device=dev))
        else:
            ecnt = torch.zeros(n_graphs, device=dev)
        D.append(ecnt)
        # degree
        deg = torch.zeros(n_graphs, device=dev)
        if edge_index.size(1) > 0:
            nd = torch.bincount(edge_index[0], minlength=x.size(0)).float()
            deg.index_add_(0, batch, nd[batch])
        D.append(deg)
        D.append(deg.max(1).values)
        # atom per-column mean
        for c in range(x.size(1)):
            m = torch.zeros(n_graphs, device=dev)
            m.index_add_(0, batch, x[:, c].float())
            D.append(m / cnt)
        # edge per-column mean
        if edge_attr.size(0) > 0:
            for c in range(edge_attr.size(1)):
                m = torch.zeros(n_graphs, device=dev)
                m.index_add_(0, batch, edge_attr[:, c].float())
                D.append(m / cnt)
        return torch.stack(D, dim=1)

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        h = self._atom_embedding(x)  # [N, hid]
        E = edge_index.size(1)
        ea = self._edge_embedding(edge_attr) if E > 0 else None  # [E, hid]
        row, col = edge_index

        for i in range(len(self.conv)):
            eps = getattr(self, f"eps{i}")
            h_conv = F.relu(self.conv[i](h))
            # Aggregate from neighbors: sum of h_conv at neighbors + edge bias
            aggr = torch.zeros_like(h).index_add_(0, row, h_conv[col])
            if ea is not None and E > 0:
                # Edge-conditioned: add edge feature to aggregated message
                aggr = aggr + torch.zeros_like(h).index_add_(0, row, self.edge_mlp[i](ea))
            h = self.node_mlp[i](h + eps * aggr)

        # Virtual node: global context
        pooled_v = torch.zeros(n_graphs, h.size(1), device=h.device, dtype=h.dtype).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        vn = self.virtual + pooled_v / counts
        # Inject virtual node back to all nodes
        h = h + vn[batch]

        # Pooling: mean, max, sum
        s_sum = pooled_v
        s_max = torch.full((n_graphs, h.size(1)), float("-inf"), device=h.device)
        s_max = s_max.index_reduce_(0, batch, h, "amax", include_self=False)
        s_max = torch.where(torch.isinf(s_max), torch.zeros_like(s_max), s_max)
        s_mean = s_sum / counts
        pooled = self.pool_mlp(torch.cat([s_mean, s_max, s_sum], dim=1))
        if DESC:
            desc = self._graph_desc(x, edge_index, edge_attr, batch, n_graphs)
            pooled = torch.cat([pooled, desc], dim=1)
        return self.head(pooled)


def slice_graph(d, i):
    n0, n1 = int(d["node_ptr"][i]), int(d["node_ptr"][i + 1])
    e0, e1 = int(d["edge_ptr"][i]), int(d["edge_ptr"][i + 1])
    return {
        "x": d["x"][n0:n1],
        "edge_index": (d["edge_index"][:, e0:e1].long() - n0),
        "edge_attr": d["edge_attr"][e0:e1],
        "y": d["y"][i],
    }


def collate(graphs):
    xs, eis, eattrs, batch, ys = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        ei = g["edge_index"]
        eis.append(ei + offset)
        eattrs.append(g["edge_attr"])
        batch.append(torch.full((g["x"].size(0),), i, dtype=torch.long))
        ys.append(g["y"])
        offset += g["x"].size(0)
    x = torch.cat(xs)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.cat(eattrs, dim=0) if eattrs else torch.zeros(0, 3)
    return x, edge_index, edge_attr, torch.cat(batch), torch.stack(ys)


def build(d, indices, device):
    graphs = [slice_graph(d, j) for j in indices]
    x, ei, ea, batch, y = collate(graphs)
    return (x.to(device), ei.to(device), ea.to(device), batch.to(device), y.to(device))


def train_one(d, tr_idx, val_idx, device, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GIN(HID, 1, LAYERS, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    trx, trei, trea, trb, tr_y = build(d, tr_idx, device)
    valx, valei, valea, valb, val_y = build(d, val_idx, device)
    n_tr = len(tr_idx)
    best = float("inf")
    best_state = None
    patience = 0
    best_ep = -1
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(trx, trei, trea, trb, n_tr)
        loss = F.mse_loss(pred.view(-1), tr_y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(valx, valei, valea, valb, len(val_idx)).view(-1)
            vr = float(torch.sqrt(torch.mean((vp - val_y.view(-1)) ** 2)))
        if vr < best - 1e-5:
            best = vr
            patience = 0
            best_ep = ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_ep


@torch.no_grad()
def predict(model, d, indices, device):
    model.eval()
    x, ei, ea, batch, _ = build(d, indices, device)
    return model(x, ei, ea, batch, len(indices)).view(-1, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    sizes = d["split_sizes"]
    tr_idx = list(range(sizes["train"]))
    val_idx = list(range(sizes["train"], sizes["train"] + sizes["val"]))
    target = (val_idx if a.split == "val"
              else list(range(sizes["train"] + sizes["val"], d["num_graphs"])))

    preds = []
    for s in range(SEEDS):
        model, best, best_ep = train_one(d, tr_idx, val_idx, device, s)
        print(f"[seed {s}] val RMSE={best:.4f} @ ep {best_ep}", flush=True)
        preds.append(predict(model, d, target, device).cpu())
    pred = torch.stack(preds).mean(dim=0)
    torch.save({"pred": pred}, Path(a.out))
    print(f"wrote predictions for {len(target)} {a.split} graphs -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
