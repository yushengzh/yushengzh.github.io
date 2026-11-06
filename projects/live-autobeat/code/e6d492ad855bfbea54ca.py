#!/usr/bin/env python3
"""GIN ensemble for MoleculeBBBP (Blood-Brain Barrier Penetration)."""
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
    xs, eis, eattrs, batch = [], [], [], []
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
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eattrs), torch.cat(batch)


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
        return torch.sigmoid(logits).cpu().numpy().flatten()


def train_gin(d, train_indices, val_indices, in_dim, edge_dim,
              hidden, n_layers, dropout, lr, epochs, seed, weight_decay=1e-4):
    """Train a single GIN model. Returns best model state and best val AUC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GIN(in_dim, hidden, n_layers, 1, edge_dim, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    y_train = d["y"][:len(train_indices)].to(DEVICE)
    y_val = d["y"][len(train_indices):len(train_indices) + len(val_indices)].cpu().numpy().flatten()

    x_tr, ei_tr, ea_tr, batch_tr = slice_graphs(d, train_indices)
    x_tr, ei_tr, ea_tr, batch_tr = x_tr.to(DEVICE), ei_tr.to(DEVICE), ea_tr.to(DEVICE), batch_tr.to(DEVICE)
    n_graphs_train = len(train_indices)

    best_val_auc = 0.0
    best_state = None
    patience = 60
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x_tr, ei_tr, ea_tr, batch_tr, n_graphs_train)
        mask = ~torch.isnan(y_train)
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(logits[mask].view(-1), y_train[mask].view(-1).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        val_preds = evaluate(model, d, val_indices)
        mask_val = ~np.isnan(y_val)
        if mask_val.sum() > 0 and len(np.unique(y_val[mask_val])) > 1:
            val_auc = roc_auc_score(y_val[mask_val], val_preds[mask_val])
        else:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model, best_val_auc


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

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, d["num_graphs"]))

    # Ensemble configurations: (hidden, n_layers, dropout, lr, seed)
    configs = [
        (64, 4, 0.2, 0.001, 42),
        (64, 4, 0.2, 0.001, 123),
        (64, 4, 0.2, 0.001, 456),
        (128, 4, 0.2, 0.001, 42),
        (128, 4, 0.2, 0.001, 123),
        (64, 6, 0.2, 0.001, 42),
        (64, 6, 0.2, 0.001, 123),
    ]

    all_val_preds = []
    all_models = []
    
    for cfg in configs:
        hidden, n_layers, dropout, lr, seed = cfg
        t1 = time.time()
        model, val_auc = train_gin(d, train_indices, val_indices, in_dim, edge_dim,
                                    hidden, n_layers, dropout, lr, 300, seed)
        all_models.append(model)
        
        # Get val predictions from this model
        val_preds = evaluate(model, d, val_indices)
        all_val_preds.append(val_preds)
        
        print(f"Config hidden={hidden} layers={n_layers} dropout={dropout} lr={lr} seed={seed}: "
              f"val_auc={val_auc:.4f} time={time.time()-t1:.1f}s", flush=True)

    # Combine val predictions (average) to check ensemble quality
    y_val = d["y"][n_train:n_train + n_val].cpu().numpy().flatten()
    mask_val = ~np.isnan(y_val)
    avg_val_preds = np.mean(all_val_preds, axis=0)
    if mask_val.sum() > 0 and len(np.unique(y_val[mask_val])) > 1:
        ens_val_auc = roc_auc_score(y_val[mask_val], avg_val_preds[mask_val])
        print(f"Ensemble val AUC: {ens_val_auc:.4f}", flush=True)

    # Predict on target split
    target_indices = val_indices if a.split == "val" else test_indices
    all_target_preds = []
    for model in all_models:
        preds = evaluate(model, d, target_indices)
        all_target_preds.append(preds)
    
    # Average predictions
    final_preds = np.mean(all_target_preds, axis=0)
    
    preds = final_preds.reshape(-1, 1)
    torch.save({"pred": torch.tensor(preds, dtype=torch.float32)}, Path(a.out))
    print(f"Wrote predictions for {len(target_indices)} {a.split} graphs -> {a.out}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
