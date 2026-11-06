#!/usr/bin/env python3
"""GIN ensemble for MoleculeBBBP with nn.Embedding input layers."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
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


class GINLayer(nn.Module):
    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_mlp = nn.Linear(hidden, hidden)  # edge hidden -> hidden
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.eps = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, edge_index, edge_emb):
        row, col = edge_index
        if edge_emb.shape[0] > 0:
            msg = self.edge_mlp(edge_emb)
            h_col = h[col] + msg
        else:
            h_col = h[col]
        agg = torch.zeros_like(h).index_add_(0, row, h_col)
        out = self.mlp((1 + self.eps) * h + agg)
        out = self.dropout(out)
        return self.act(out)


class GIN(nn.Module):
    def __init__(self, cat_sizes, edge_cat_sizes, hidden, n_layers, out_dim=1, dropout=0.2):
        """cat_sizes: list of 9 ints (num categories per atom feature).
           edge_cat_sizes: list of 3 ints (num categories per bond feature).
        """
        super().__init__()
        # Node embedding: per-column nn.Embedding, concat, project to hidden
        emb_dim = max(4, hidden // 4)
        self.node_embeds = nn.ModuleList([
            nn.Embedding(K, emb_dim) for K in cat_sizes
        ])
        self.node_proj = nn.Linear(emb_dim * len(cat_sizes), hidden)
        # Edge embedding: per-column nn.Embedding, concat, project to hidden
        self.edge_embeds = nn.ModuleList([
            nn.Embedding(K, emb_dim) for K in edge_cat_sizes
        ])
        self.edge_proj = nn.Linear(emb_dim * len(edge_cat_sizes), hidden)
        # GIN layers
        self.layers = nn.ModuleList([
            GINLayer(hidden, dropout) for _ in range(n_layers)
        ])
        # Head: mean+max pooling concat -> hidden -> out
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_id, edge_index, edge_attr_id, batch, n_graphs):
        # x_id: [N, 9] long, edge_attr_id: [E, 3] long
        # Embed nodes
        embs = [emb(x_id[:, c]) for c, emb in enumerate(self.node_embeds)]
        h = self.node_proj(torch.cat(embs, dim=1))
        # Embed edges
        if edge_attr_id.shape[0] > 0:
            e_embs = [emb(edge_attr_id[:, c]) for c, emb in enumerate(self.edge_embeds)]
            edge_emb = self.edge_proj(torch.cat(e_embs, dim=1))
        else:
            edge_emb = torch.zeros(0, self.edge_proj.out_features, device=h.device)
        # GIN layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_emb)
        # Mean + max pooling
        # batch: [N] long
        n = h.size(0)
        mean_pool = torch.zeros(n_graphs, h.size(1), device=h.device)
        mean_pool.index_add_(0, batch, h)
        mean_pool = mean_pool / torch.clamp(
            torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones_like(h[:, 0])).unsqueeze(1), min=1)
        max_pool = torch.full((n_graphs, h.size(1)), float('-inf'), device=h.device)
        max_pool = max_pool.scatter_reduce(0, batch.unsqueeze(1).expand_as(h), h, reduce='amax', include_self=False)
        out = self.head(torch.cat([mean_pool, max_pool], dim=1))
        return out.squeeze(-1)


def build_id_maps(d, train_indices):
    """Build per-column ID -> dense index maps from train data. Returns (node_maps, edge_maps, cat_sizes, edge_cat_sizes)."""
    n0, n1 = int(d["node_ptr"][train_indices[0]]), int(d["node_ptr"][train_indices[-1] + 1])
    e0, e1 = int(d["edge_ptr"][train_indices[0]]), int(d["edge_ptr"][train_indices[-1] + 1])
    x_train = d["x"][n0:n1].numpy().astype(np.int64)
    ea_train = d["edge_attr"][e0:e1].numpy().astype(np.int64)
    
    node_maps, cat_sizes = [], []
    for c in range(x_train.shape[1]):
        vals = np.unique(x_train[:, c])
        m = {int(v): i + 1 for i, v in enumerate(vals)}  # 0 = unknown
        node_maps.append(m)
        cat_sizes.append(len(m) + 1)  # +1 for unknown bucket
    
    edge_maps, edge_cat_sizes = [], []
    for c in range(ea_train.shape[1]):
        vals = np.unique(ea_train[:, c])
        m = {int(v): i + 1 for i, v in enumerate(vals)}
        edge_maps.append(m)
        edge_cat_sizes.append(len(m) + 1)
    
    return node_maps, edge_maps, cat_sizes, edge_cat_sizes


def to_ids(x, node_maps):
    """Convert float x [N, C] to long IDs [N, C] using maps."""
    x = x.numpy().astype(np.int64)
    out = np.zeros(x.shape, dtype=np.int64)  # 0 = unknown
    for c in range(x.shape[1]):
        for v, idx in node_maps[c].items():
            out[x[:, c] == v, c] = idx
    return torch.tensor(out, dtype=torch.long)


def to_edge_ids(ea, edge_maps):
    ea = ea.numpy().astype(np.int64)
    out = np.zeros(ea.shape, dtype=np.int64)
    for c in range(ea.shape[1]):
        for v, idx in edge_maps[c].items():
            out[ea[:, c] == v, c] = idx
    return torch.tensor(out, dtype=torch.long)


def train_gin(d, train_indices, val_indices, cat_sizes, edge_cat_sizes,
              node_maps, edge_maps, hidden, n_layers, dropout, lr, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x_tr, ei_tr, ea_tr, b_tr = slice_graphs(d, train_indices)
    y_tr = d["y"][train_indices].squeeze(-1)
    x_id_tr = to_ids(x_tr, node_maps).to(DEVICE)
    ea_id_tr = to_edge_ids(ea_tr, edge_maps).to(DEVICE)
    ei_tr = ei_tr.to(DEVICE)
    b_tr = b_tr.to(DEVICE)
    y_tr = y_tr.to(DEVICE)
    
    x_val, ei_val, ea_val, b_val = slice_graphs(d, val_indices)
    y_val = d["y"][val_indices].squeeze(-1)
    x_id_val = to_ids(x_val, node_maps).to(DEVICE)
    ea_id_val = to_edge_ids(ea_val, edge_maps).to(DEVICE)
    ei_val = ei_val.to(DEVICE)
    b_val = b_val.to(DEVICE)
    
    model = GIN(cat_sizes, edge_cat_sizes, hidden, n_layers, 1, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    best_state = None
    patience = 60
    bad_epochs = 0
    
    n_train_graphs = len(train_indices)
    n_val_graphs = len(val_indices)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_id_tr, ei_tr, ea_id_tr, b_tr, n_train_graphs)
        loss = criterion(logits, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            val_logits = model(x_id_val, ei_val, ea_id_val, b_val, n_val_graphs)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        mask = ~np.isnan(y_val.cpu().numpy())
        if mask.sum() > 0 and len(np.unique(y_val.cpu().numpy()[mask])) > 1:
            val_auc = roc_auc_score(y_val.cpu().numpy()[mask], val_probs[mask])
        else:
            val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return model, best_val_auc


def evaluate(model, d, indices, node_maps, edge_maps):
    model.eval()
    x, ei, ea, b = slice_graphs(d, indices)
    x_id = to_ids(x, node_maps).to(DEVICE)
    ea_id = to_edge_ids(ea, edge_maps).to(DEVICE)
    ei = ei.to(DEVICE)
    b = b.to(DEVICE)
    n_graphs = len(indices)
    with torch.no_grad():
        logits = model(x_id, ei, ea_id, b, n_graphs)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


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

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, d["num_graphs"]))

    # Build ID maps from train data
    node_maps, edge_maps, cat_sizes, edge_cat_sizes = build_id_maps(d, train_indices)
    print(f"cat_sizes={cat_sizes} edge_cat_sizes={edge_cat_sizes}", flush=True)

    configs = [
        (256, 4, 0.2, 0.001, 42),
        (256, 4, 0.2, 0.001, 123),
        (256, 4, 0.2, 0.001, 456),
        (256, 4, 0.2, 0.001, 789),
        (256, 4, 0.2, 0.001, 1000),
        (256, 4, 0.2, 0.001, 2024),
        (128, 4, 0.2, 0.001, 42),
        (128, 4, 0.2, 0.001, 123),
        (128, 4, 0.2, 0.001, 456),
        (64, 6, 0.2, 0.001, 42),
        (64, 6, 0.2, 0.001, 123),
        (64, 6, 0.2, 0.001, 456),
    ]

    all_val_preds = []
    all_models = []

    val_aucs = []
    for cfg in configs:
        hidden, n_layers, dropout, lr, seed = cfg
        t1 = time.time()
        model, val_auc = train_gin(d, train_indices, val_indices, cat_sizes, edge_cat_sizes,
                                    node_maps, edge_maps, hidden, n_layers, dropout, lr, 300, seed)
        all_models.append(model)
        val_aucs.append(val_auc)
        val_preds = evaluate(model, d, val_indices, node_maps, edge_maps)
        all_val_preds.append(val_preds)
        print(f"hidden={hidden} layers={n_layers} seed={seed}: val_auc={val_auc:.4f} t={time.time()-t1:.1f}s", flush=True)

    y_val = d["y"][n_train:n_train + n_val].cpu().numpy().flatten()
    mask_val = ~np.isnan(y_val)
    avg_val_preds = np.mean(all_val_preds, axis=0)
    if mask_val.sum() > 0 and len(np.unique(y_val[mask_val])) > 1:
        ens_val_auc = roc_auc_score(y_val[mask_val], avg_val_preds[mask_val])
        print(f"Ensemble val AUC: {ens_val_auc:.4f}", flush=True)

    target_indices = val_indices if a.split == "val" else test_indices
    all_target_preds = [evaluate(model, d, target_indices, node_maps, edge_maps) for model in all_models]
    w = np.array(val_aucs, dtype=float)
    w = np.clip(w, 1e-6, None)
    w = w / w.sum()
    final_preds = np.average(np.stack(all_target_preds, axis=0), axis=0, weights=w)

    preds = final_preds.reshape(-1, 1)
    torch.save({"pred": torch.tensor(preds, dtype=torch.float32)}, Path(a.out))
    print(f"Wrote {len(target_indices)} {a.split} preds -> {a.out} | total {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
