#!/usr/bin/env python3
"""GIN graph classifier for ogbg-molbbbp.

5-layer GIN with batch norm, dropout, class-weighted BCE loss,
early stopping on validation, multi-seed ensemble.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph slicing utilities (flat arrays -> per-graph)
# ---------------------------------------------------------------------------

class FlatGraphDataset:
    """Wraps the flat arrays and lets us build batches efficiently."""

    def __init__(self, d: dict):
        self.d = d
        self.x = d["x"]
        self.edge_index = d["edge_index"]
        self.edge_attr = d.get("edge_attr", None)
        self.node_ptr = d["node_ptr"]
        self.edge_ptr = d["edge_ptr"]
        self.y = d["y"]
        self.num_graphs = d["num_graphs"]

    def build_batch(self, indices):
        """Build a batch tensor from a list of graph indices.
        Returns (x, edge_index, edge_attr, batch_ids, y_slice).
        """
        d = self.d
        xs, eis, eats, batches, ys = [], [], [], [], []
        offset = 0
        for i, gi in enumerate(indices):
            n0 = int(d["node_ptr"][gi])
            n1 = int(d["node_ptr"][gi + 1])
            e0 = int(d["edge_ptr"][gi])
            e1 = int(d["edge_ptr"][gi + 1])
            n = n1 - n0
            xs.append(self.x[n0:n1])
            eis.append(d["edge_index"][:, e0:e1] - n0 + offset)
            batches.append(torch.full((n,), i, dtype=torch.long))
            if self.edge_attr is not None:
                eats.append(self.edge_attr[e0:e1])
            ys.append(self.y[gi])
            offset += n
        x = torch.cat(xs, dim=0)
        edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
        batch_ids = torch.cat(batches) if batches else torch.zeros(0, dtype=torch.long)
        y = torch.stack(ys) if ys else torch.zeros(0)
        if self.edge_attr is not None:
            edge_attr = torch.cat(eats, dim=0) if eats else torch.zeros(0, self.edge_attr.shape[1])
        else:
            edge_attr = None
        return x, edge_index, edge_attr, batch_ids, y


class GINConv(nn.Module):
    """GIN convolution: (1+eps) * h + sum(neighbor_h), with MLP.
    Uses edge attributes to condition the message.
    """

    def __init__(self, in_dim, out_dim, edge_dim, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)

    def forward(self, h, edge_index, edge_attr, batch_ids, n_graphs):
        row, col = edge_index
        # Edge-conditioned message
        if edge_attr is not None and edge_attr.shape[0] > 0:
            msg = h[col] + self.edge_proj(edge_attr)
        else:
            msg = h[col]
        # Aggregate
        agg = torch.zeros_like(h).index_add_(0, row, msg)
        out = (1.0 + self.eps) * h + agg
        out = self.mlp(out)
        return out


class GIN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128, n_layers=5, dropout=0.2, eps=0.0):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList([
            GINConv(hidden, hidden, edge_dim, eps) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch_ids, n_graphs):
        h = self.input_proj(x)
        for conv in self.convs:
            h = self.dropout(conv(h, edge_index, edge_attr, batch_ids, n_graphs))
        # Mean pooling
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch_ids, h)
        counts = torch.bincount(batch_ids, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pooled = pooled / counts
        logits = self.head(pooled).squeeze(-1)
        return logits


def train_one_seed(seed, dataset, train_idx, val_idx, device, epochs=100, lr=0.001,
                   weight_pos=None, weight_neg=None, patience=15, batch_size=256):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GIN(
        node_dim=dataset.x.size(1),
        edge_dim=dataset.edge_attr.size(1) if dataset.edge_attr is not None else 0,
        hidden=128,
        n_layers=5,
        dropout=0.2,
        eps=0.0,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Class weights
    if weight_pos is not None and weight_neg is not None:
        pos_weight = torch.tensor([weight_pos / weight_neg], device=device)
    else:
        pos_weight = None

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_idx), batch_size):
            chunk = [int(train_idx[j]) for j in perm[i:i + batch_size]]
            x, ei, ea, batch_ids, y = dataset.build_batch(chunk)
            x, ei, batch_ids, y = x.to(device), ei.to(device), batch_ids.to(device), y.to(device)
            if ea is not None:
                ea = ea.to(device)

            opt.zero_grad()
            logits = model(x, ei, ea, batch_ids, len(chunk))
            if pos_weight is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, y.squeeze(-1).float(), pos_weight=pos_weight)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total_loss += loss.item() * len(chunk)
            n_batches += 1

        sched.step()
        avg_loss = total_loss / max(1, len(train_idx))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, len(val_idx), 512):
                chunk = [int(val_idx[j]) for j in range(i, min(i + 512, len(val_idx)))]
                x, ei, ea, batch_ids, y = dataset.build_batch(chunk)
                x, ei, batch_ids = x.to(device), ei.to(device), batch_ids.to(device)
                if ea is not None:
                    ea = ea.to(device)
                out = model(x, ei, ea, batch_ids, len(chunk))
                val_logits.append(out)
            val_logits = torch.cat(val_logits).cpu().numpy()
            val_y = dataset.y[val_idx].squeeze(-1).numpy()

            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(val_y, val_logits)
            except Exception:
                val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_val_auc


def predict_graphs(model, dataset, indices, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            chunk = [int(indices[j]) for j in range(i, min(i + batch_size, len(indices)))]
            x, ei, ea, batch_ids, _ = dataset.build_batch(chunk)
            x, ei, batch_ids = x.to(device), ei.to(device), batch_ids.to(device)
            if ea is not None:
                ea = ea.to(device)
            out = model(x, ei, ea, batch_ids, len(chunk))
            preds.append(torch.sigmoid(out).cpu())
    return torch.cat(preds) if preds else torch.zeros(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t_start = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    dataset = FlatGraphDataset(d)

    tr, va, te = d["split_sizes"]["train"], d["split_sizes"]["val"], d["split_sizes"]["test"]
    train_idx = torch.arange(tr)
    val_idx = torch.arange(tr, tr + va)

    target = (val_idx if a.split == "val"
              else torch.arange(tr + va, d["num_graphs"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class weights from training labels
    y_train = dataset.y[train_idx].squeeze(-1)
    pos_count = (y_train == 1).float().sum().item()
    neg_count = (y_train == 0).float().sum().item()
    # Inverse frequency weights (mild: square root)
    n_total = pos_count + neg_count
    w_pos = np.sqrt(n_total / (2 * pos_count))
    w_neg = np.sqrt(n_total / (2 * neg_count))
    print(f"Class weights: pos={w_pos:.3f} neg={w_neg:.3f} (pos={pos_count:.0f}, neg={neg_count:.0f})")

    # Multi-seed ensemble
    n_seeds = 5
    all_preds = []
    for s in range(n_seeds):
        seed = 42 + s * 1000
        print(f"\n--- Seed {seed} ---", flush=True)
        model, val_auc = train_one_seed(
            seed, dataset, train_idx, val_idx, device,
            epochs=100, lr=0.001,
            weight_pos=pos_count, weight_neg=neg_count,
            patience=15, batch_size=256,
        )
        print(f"  Best val AUC: {val_auc:.4f}")
        pred = predict_graphs(model, dataset, target, device)
        all_preds.append(pred)
        del model
        torch.cuda.empty_cache()

    # Average ensemble
    ensemble_pred = torch.stack(all_preds).mean(dim=0)

    out = {"pred": ensemble_pred.unsqueeze(-1)}
    torch.save(out, Path(a.out))
    elapsed = time.time() - t_start
    print(f"\nWrote predictions for {len(target)} {a.split} graphs -> {a.out}")
    print(f"Total time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
