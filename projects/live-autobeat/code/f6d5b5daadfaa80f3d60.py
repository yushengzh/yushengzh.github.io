#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GINConv(nn.Module):
    """GIN with edge-conditioned messages + residual + BN."""
    def __init__(self, dim, edge_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.edge = nn.Linear(edge_dim, dim) if edge_dim and edge_dim > 0 else None
        self.bn = nn.BatchNorm1d(dim)
        self.eps = nn.Parameter(torch.tensor([0.0]))
        self.drop = dropout

    def forward(self, x, ei, edge_attr):
        row, col = ei[0], ei[1]
        if self.edge is not None and edge_attr is not None:
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
        layers = [GINConv(hidden_dim, edge_dim, dropout)]
        for _ in range(1, n_layers):
            layers.append(GINConv(hidden_dim, edge_dim, dropout))
        self.convs = nn.ModuleList(layers)
        self.att = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))
        self.head = nn.Linear(hidden_dim * 3, 1)

    @staticmethod
    def _mean(x, batch, n):
        out = torch.zeros(n, x.size(1), device=x.device).index_add_(0, batch, x)
        c = torch.zeros(n, 1, device=x.device).index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
        c = c.clamp(min=1.0)
        return out / c

    @staticmethod
    def _max(x, batch, n):
        out = torch.full((n, x.size(1)), -1e9, device=x.device)
        return out.scatter_reduce(0, batch.unsqueeze(1).expand(-1, x.size(1)), x,
                                  reduce="amax", include_self=False)

    @staticmethod
    def _sum(x, batch, n):
        return torch.zeros(n, x.size(1), device=x.device).index_add_(0, batch, x)

    def forward(self, x, ei, edge_attr, batch, n_graphs):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, ei, edge_attr)
        attn = torch.sigmoid(self.att(x))
        pooled = [self._mean(attn * x, batch, n_graphs),
                  self._max(attn * x, batch, n_graphs),
                  self._sum(attn * x, batch, n_graphs)]
        return self.head(torch.cat(pooled, dim=1))


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
    logits, labels = [], []
    for bidx in get_batches(indices, batch_size):
        chunk = [slice_graph(d, j) for j in bidx]
        x, ei, batch, y, ea = collate(chunk)
        x, ei, batch, ea = x.to(device), ei.to(device), batch.to(device), ea.to(device)
        out = model(x, ei, ea, batch, len(chunk)).squeeze(-1)
        logits.append(out.cpu())
        labels.append(y)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    mask = ~torch.isnan(labels).any(dim=1)
    from sklearn.metrics import roc_auc_score
    if mask.sum() < 2:
        return 0.5
    return roc_auc_score(labels[mask].squeeze(-1).numpy(), logits[mask].numpy())


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


def train_epoch(model, d, indices, opt, batch_size, device, pos_weight):
    model.train()
    total_loss, n_samples = 0.0, 0
    for bidx in get_batches(indices, batch_size):
        chunk = [slice_graph(d, j) for j in bidx]
        x, ei, batch, y, ea = collate(chunk)
        x, ei, batch, y, ea = x.to(device), ei.to(device), batch.to(device), y.to(device), ea.to(device)
        logits = model(x, ei, ea, batch, len(chunk)).squeeze(-1)
        mask = ~torch.isnan(y).any(dim=1)
        if not mask.all():
            loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask].squeeze(-1),
                                                      pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1),
                                                      pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        n = int(mask.sum().item())
        total_loss += float(loss) * n
        n_samples += n
    return total_loss / max(1, n_samples)


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

    # One-hot encode the 9 atom-category features and 3 edge categories.
    x_raw = d["x"]
    per_col_max = x_raw.max(dim=0).values.long() + 1  # number of categories per column
    onehot_parts = []
    start = 0
    for c in range(x_raw.size(1)):
        nv = int(per_col_max[c].item())
        idx = x_raw[:, c].long()
        oh = torch.zeros(x_raw.size(0), nv)
        oh.scatter_(1, idx.unsqueeze(1), 1.0)
        onehot_parts.append(oh)
    d["x"] = torch.cat(onehot_parts, dim=1)
    IN_D = d["x"].size(1)
    # One-hot encode the 3 edge-category features (bond type, stereo, conjugation).
    ea_raw = d["edge_attr"]
    ea_per_col = ea_raw.max(dim=0).values.long() + 1
    ea_oh_parts = []
    for c in range(ea_raw.size(1)):
        nv = int(ea_per_col[c].item())
        idx = ea_raw[:, c].long()
        oh = torch.zeros(ea_raw.size(0), nv)
        oh.scatter_(1, idx.unsqueeze(1), 1.0)
        ea_oh_parts.append(oh)
    d["edge_attr"] = torch.cat(ea_oh_parts, dim=1)
    EDGE_D = d["edge_attr"].size(1)

    HIDDEN = 192
    N_LAYERS = 5
    EPOCHS = 15
    BATCH_SIZE = 512
    LR = 3e-4
    WD = 1e-5
    DROPOUT = 0.0
    # Class weighting from train positive rate.
    ytr = d["y"][:sizes["train"]]
    m = ~torch.isnan(ytr).any(dim=1)
    pos_rate = float(ytr[m].mean().item())
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], device=DEVICE)

    seeds = [0, 1, 2]
    all_preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = GINModel(IN_D, EDGE_D, HIDDEN, N_LAYERS, dropout=DROPOUT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        g = torch.Generator().manual_seed(seed)
        best_auc, best_state = -1.0, None
        for ep in range(EPOCHS):
            perm = torch.randperm(len(tr), generator=g)
            idx = [tr[int(k)] for k in perm]
            loss = train_epoch(model, d, idx, opt, BATCH_SIZE, DEVICE, pos_weight)
            vauc = evaluate(model, d, val_idx, BATCH_SIZE, DEVICE)
            print(f"seed {seed} epoch {ep+1}/{EPOCHS} loss={loss:.4f} val_auc={vauc:.4f}", flush=True)
            if vauc > best_auc:
                best_auc = vauc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        print(f"seed {seed} best val_auc={best_auc:.4f}", flush=True)
        preds = predict(model, d, target, BATCH_SIZE, DEVICE)
        all_preds.append(preds)
        del model, best_state, opt, g
        torch.cuda.empty_cache()
    preds = torch.stack(all_preds, dim=0).mean(dim=0)
    preds = preds.reshape(-1, 1)
    torch.save(dict(pred=preds.cpu()), Path(a.out))
    print(f"wrote predictions for {len(target)} {a.split} graphs -> {a.out} (ensemble of {len(seeds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
