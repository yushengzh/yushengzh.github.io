#!/usr/bin/env python3
"""Improved GIN for MoleculeBBBP (Blood-Brain Barrier Penetration)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def slice_graphs(d, indices):
    """Slice multiple graphs from the flat arrays, returning local edge_index."""
    xs, eis, eattrs = [], [], []
    batch = []
    offset = 0
    for i, idx in enumerate(indices):
        n0, n1 = int(d["node_ptr"][idx]), int(d["node_ptr"][idx + 1])
        e0, e1 = int(d["edge_ptr"][idx]), int(d["edge_ptr"][idx + 1])
        xs.append(d["x"][n0:n1])
        if e1 > e0:
            eis.append(d["edge_index"][:, e0:e1].long() - n0 + offset)
            eattrs.append(d["edge_attr"][e0:e1])
        else:
            eis.append(torch.zeros(2, 0, dtype=torch.long))
            eattrs.append(torch.zeros(0, 3))
        batch.append(torch.full((n1 - n0,), i, dtype=torch.long))
        offset += n1 - n0
    x = torch.cat(xs, dim=0)
    ei = torch.cat(eis, dim=1)
    ea = torch.cat(eattrs, dim=0)
    batch = torch.cat(batch)
    return x, ei, ea, batch


class GINLayer(torch.nn.Module):
    def __init__(self, hidden, edge_dim=3, dropout=0.1):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
        )
        self.edge_mlp = torch.nn.Linear(edge_dim, hidden)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.eps = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        if edge_attr.shape[0] > 0:
            msg = self.edge_mlp(edge_attr)
            h_col = h[col] + msg
        else:
            h_col = h[col]
        agg = torch.zeros_like(h).index_add_(0, row, h_col)
        out = self.mlp((1 + self.eps) * h + agg)
        out = self.dropout(out)
        return self.act(out)


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_layers, out_dim=1, edge_dim=3, dropout=0.1):
        super().__init__()
        self.input_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
        )
        self.layers = torch.nn.ModuleList([
            GINLayer(hidden, edge_dim, dropout) for _ in range(n_layers)
        ])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        h = self.input_mlp(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        pool_mean = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pool_mean = pool_mean / counts
        pool_max = torch.full((n_graphs, h.size(1)), -1e9, device=h.device)
        pool_max = pool_max.index_reduce_(0, batch, h, "amax", include_self=False)
        pooled = torch.cat([pool_mean, pool_max], dim=1)
        return self.head(pooled)


def evaluate(model, d, indices):
    model.eval()
    with torch.no_grad():
        x, ei, ea, batch = slice_graphs(d, indices)
        x, ei, ea, batch = x.to(DEVICE), ei.to(DEVICE), ea.to(DEVICE), batch.to(DEVICE)
        n_graphs = len(indices)
        logits = model(x, ei, ea, batch, n_graphs)
        return torch.sigmoid(logits).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    sizes = d["split_sizes"]
    n_train, n_val, n_test = sizes["train"], sizes["val"], sizes["test"]

    in_dim = d["x"].size(1)
    edge_dim = d["edge_attr"].size(1) if "edge_attr" in d else 0

    HIDDEN = 64
    N_LAYERS = 4
    DROPOUT = 0.2
    EPOCHS = 300
    LR = 0.001
    WEIGHT_DECAY = 1e-4

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, d["num_graphs"]))

    y_train = d["y"][:n_train].to(DEVICE)
    y_val = d["y"][n_train:n_train + n_val].cpu().numpy().flatten()

    x_tr, ei_tr, ea_tr, batch_tr = slice_graphs(d, train_indices)
    x_tr, ei_tr, ea_tr, batch_tr = x_tr.to(DEVICE), ei_tr.to(DEVICE), ea_tr.to(DEVICE), batch_tr.to(DEVICE)
    n_graphs_train = len(train_indices)

    torch.manual_seed(42)
    model = GIN(in_dim, HIDDEN, N_LAYERS, 1, edge_dim, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val_auc = 0.0
    best_state = None
    patience = 50
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x_tr, ei_tr, ea_tr, batch_tr, n_graphs_train)
        mask = ~torch.isnan(y_train)
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(logits[mask].view(-1), y_train[mask].view(-1).float())
        loss.backward()
        opt.step()
        scheduler.step()

        val_preds = evaluate(model, d, val_indices).numpy().flatten()
        mask_val = ~np.isnan(y_val)
        if mask_val.sum() > 0 and len(np.unique(y_val[mask_val])) > 1:
            val_auc = roc_auc_score(y_val[mask_val], val_preds[mask_val])
        else:
            val_auc = 0.5

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} loss={loss.item():.4f} val_auc={val_auc:.4f} best={best_val_auc:.4f}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val_auc={best_val_auc:.4f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    target_indices = val_indices if a.split == "val" else test_indices
    preds = evaluate(model, d, target_indices).numpy()
    # Ensure shape is [M, 1]
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    torch.save({"pred": torch.tensor(preds, dtype=torch.float32)}, Path(a.out))
    print(f"Wrote predictions for {len(target_indices)} {a.split} graphs -> {a.out}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
