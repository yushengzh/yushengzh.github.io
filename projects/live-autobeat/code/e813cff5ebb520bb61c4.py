#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim, eps_init=0.0, dropout=0.0):
        super().__init__()
        self.conv = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge = nn.Linear(edge_dim, dim) if edge_dim and edge_dim > 0 else None
        self.eps = nn.Parameter(torch.tensor([eps_init]))
        self.drop = dropout

    def forward(self, x, ei, edge_attr):
        row, col = ei[0], ei[1]
        if edge_attr is not None and self.edge is not None:
            ea = self.edge(edge_attr)
        else:
            ea = torch.ones(row.size(0), x.size(1), device=x.device)
        msg = x[col] * ea
        h = torch.zeros_like(x).index_add_(0, row, msg)
        out = (1 + self.eps) * x + h
        out = self.bn(out)
        out = F.dropout(self.mlp(out) + out, p=self.drop, training=self.training)
        return out


class GINModel(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        layers = [GINLayer(hidden_dim, edge_dim, eps_init=0.0, dropout=dropout)]
        for _ in range(1, n_layers):
            layers.append(GINLayer(hidden_dim, edge_dim, eps_init=1.0, dropout=dropout))
        self.convs = nn.ModuleList(layers)
        self.att = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))
        self.head = nn.Linear(hidden_dim * 3, 1)

    def forward(self, x, ei, edge_attr, batch, n_graphs):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, ei, edge_attr)
        attn = torch.sigmoid(self.att(x))
        pooled = [self._mean(attn * x, batch, n_graphs),
                  self._max(attn * x, batch, n_graphs),
                  self._sum(attn * x, batch, n_graphs)]
        return self.head(torch.cat(pooled, dim=1))

    @staticmethod
    def _mean(x, batch, n):
        out = torch.zeros(n, x.size(1), device=x.device).index_add_(0, batch, x)
        c = torch.bincount(batch, minlength=n).clamp(min=1).unsqueeze(-1).float()
        return out / c

    @staticmethod
    def _max(x, batch, n):
        out = torch.full((n, x.size(1)), -1e9, device=x.device)
        return out.scatter_reduce(0, batch.unsqueeze(1).expand(-1, x.size(1)), x,
                                  reduce="amax", include_self=False)

    @staticmethod
    def _sum(x, batch, n):
        return torch.zeros(n, x.size(1), device=x.device).index_add_(0, batch, x)


def slice_graph(d, i):
    n0, n1 = int(d["node_ptr"][i]), int(d["node_ptr"][i + 1])
    e0, e1 = int(d["edge_ptr"][i]), int(d["edge_ptr"][i + 1])
    return dict(x=d["x"][n0:n1],
                edge_index=(d["edge_index"][:, e0:e1].long() - n0),
                y=d["y"][i],
                edge_attr=d["edge_attr"][e0:e1])


def collate(graphs):
    xs, eis, batch, ys, eas = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        eis.append(g["edge_index"] + offset)
        batch.append(torch.full((g["x"].size(0),), i, dtype=torch.long))
        ys.append(g["y"])
        eas.append(g["edge_attr"])
        offset += g["x"].size(0)
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(batch),
            torch.stack(ys), torch.cat(eas))


def get_batches(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]


@torch.no_grad()
def evaluate(model, d, indices, batch_size, device):
    model.eval()
    probs, ys = [], []
    for bidx in get_batches(indices, batch_size):
        chunk = [slice_graph(d, j) for j in bidx]
        x, ei, batch, y, ea = collate(chunk)
        x, ei, batch, ea = x.to(device), ei.to(device), batch.to(device), ea.to(device)
        out = model(x, ei, ea, batch, len(chunk)).squeeze(-1)
        p = torch.sigmoid(out)
        for j in range(len(chunk)):
            yv = float(y[j]) if not torch.isnan(y[j]).all() else None
            if yv is not None:
                probs.append(float(p[j]))
                ys.append(yv)
    if len(probs) < 2 or len(set(ys)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(ys, probs)


def train_epoch(model, d, indices, opt, batch_size, device):
    model.train()
    total_loss, n_samples = 0.0, 0
    for bidx in get_batches(indices, batch_size):
        chunk = [slice_graph(d, j) for j in bidx]
        x, ei, batch, y, ea = collate(chunk)
        x, ei, batch, y, ea = x.to(device), ei.to(device), batch.to(device), y.to(device), ea.to(device)
        logits = model(x, ei, ea, batch, len(chunk)).squeeze(-1)
        mask = ~torch.isnan(y).any(dim=1)
        if not mask.all():
            loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask].squeeze(-1))
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        n = int(mask.sum().item())
        total_loss += float(loss) * n
        n_samples += n
    return total_loss / max(1, n_samples)


@torch.no_grad()
def predict(model, d, indices, batch_size, device):
    model.eval()
    preds = []
    for bidx in get_batches(indices, batch_size):
        chunk = [slice_graph(d, j) for j in bidx]
        x, ei, batch, y, ea = collate(chunk)
        x, ei, batch, ea = x.to(device), ei.to(device), batch.to(device), ea.to(device)
        out = model(x, ei, ea, batch, len(chunk)).squeeze(-1)
        preds.append(torch.sigmoid(out))
    return torch.cat(preds) if preds else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    sizes = d["split_sizes"]
    tr = list(range(sizes["train"]))
    val_idx = list(range(sizes["train"], sizes["train"] + sizes["val"]))
    test_idx = list(range(sizes["train"] + sizes["val"], d["num_graphs"]))
    target = val_idx if a.split == "val" else test_idx

    IN_D = d["x"].size(1)
    EDGE_D = d["edge_attr"].size(1)
    HIDDEN = 128
    N_LAYERS = 4
    EPOCHS = 30
    BATCH_SIZE = 512
    LR = 3e-4
    WD = 1e-5
    DROPOUT = 0.0

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = GINModel(IN_D, EDGE_D, HIDDEN, N_LAYERS, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    g = torch.Generator().manual_seed(0)

    for ep in range(EPOCHS):
        perm = torch.randperm(len(tr), generator=g)
        idx = [tr[int(j)] for j in perm]
        loss = train_epoch(model, d, idx, opt, BATCH_SIZE, DEVICE)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            vauc = evaluate(model, d, val_idx, BATCH_SIZE, DEVICE)
            print(f"epoch {ep+1}/{EPOCHS} loss={loss:.4f} val_auc={vauc:.4f}", flush=True)

    preds = predict(model, d, target, BATCH_SIZE, DEVICE)
    preds = preds.reshape(-1, 1)
    torch.save(dict(pred=preds.cpu()), Path(a.out))
    print("wrote predictions for", len(target), a.split, "graphs ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
