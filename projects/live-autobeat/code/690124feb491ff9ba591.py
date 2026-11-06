#!/usr/bin/env python3
"""Strong GNN for ogbg-molhiv: GIN with edge features, residual layers, sum pooling.

Optimized for CPU with pre-computed batch structures.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"
HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 128))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 8))
LR = float(os.environ.get("SOLVER_LR", 0.001))
BATCH = int(os.environ.get("SOLVER_BATCH", 512))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 4))
WEIGHT_DECAY = 1e-5
PATIENCE = 3


class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.edge_mlp = nn.Linear(edge_dim, dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        h = x[col]
        if edge_attr is not None:
            h = h + self.edge_mlp(edge_attr)
        msg = torch.zeros_like(x).index_add_(0, row, h)
        x = self.act(self.mlp(x + msg))
        return x


class GIN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden, n_layers, out_dim):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GINLayer(hidden, edge_dim) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        h = F.relu(self.input_lin(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        return self.head(pooled)


def prepare_data(d):
    """Pre-compute all data arrays on CPU for fast access."""
    x = d["x"].float()
    ei = d["edge_index"].long()
    ea = d["edge_attr"].float()
    node_ptr = d["node_ptr"].long()
    edge_ptr = d["edge_ptr"].long()
    y = d["y"].float()
    return x, ei, ea, node_ptr, edge_ptr, y


def collate(x, ei, ea, node_ptr, edge_ptr, y, indices):
    """Collate a batch of graphs into flat arrays."""
    n_batch = len(indices)
    if n_batch == 0:
        return None
    
    # Get node and edge ranges for each graph
    n_start = node_ptr[indices]
    n_end = node_ptr[indices + 1]
    e_start = edge_ptr[indices]
    e_end = edge_ptr[indices + 1]
    
    # Total nodes and edges in this batch
    n_nodes = (n_end - n_start).sum().item()
    n_edges = (e_end - e_start).sum().item()
    
    # Batch vector: which graph each node belongs to
    batch_sizes = (n_end - n_start)
    batch_vec = torch.repeat_interleave(torch.arange(n_batch), batch_sizes)
    
    # Gather node features
    node_indices = torch.cat([torch.arange(s, e) for s, e in zip(n_start, n_end)])
    x_batch = x[node_indices]
    
    # Gather edge indices (offset per graph)
    edge_node_offsets = torch.zeros(n_batch + 1, dtype=torch.long)
    edge_node_offsets[1:] = torch.cumsum(batch_sizes, 0)
    
    # For each edge, we need to remap node indices from global to local (batch-relative)
    # Global edge indices -> subtract the graph's node start -> add the batch offset
    e_global = torch.cat([torch.arange(s, e) for s, e in zip(e_start, e_end)])
    
    # Find which graph each edge belongs to
    edge_graph = torch.searchsorted(edge_node_offsets[1:], edge_node_offsets[:-1]) if False else None
    
    # Simpler: use searchsorted on n_start to find which graph each global edge belongs to
    # Actually, let's use a different approach: build edge_index by iterating
    # But for speed, let's use the fact that edges are ordered by graph
    
    # Edge source/dest in global space
    src_global = ei[0, e_global]
    dst_global = ei[1, e_global]
    
    # For each edge, find which graph it belongs to using edge_ptr
    # edge_graph_idx: which graph each edge in the batch belongs to
    edge_graph_idx = torch.zeros(n_edges, dtype=torch.long)
    cum_e = 0
    for i in range(n_batch):
        n_e = (e_end[i] - e_start[i]).item()
        edge_graph_idx[cum_e:cum_e+n_e] = i
        cum_e += n_e
    
    # Remap: local node index = global - n_start[graph] + batch_offset[graph]
    n_start_per_edge = n_start[edge_graph_idx]
    batch_offset_per_edge = edge_node_offsets[edge_graph_idx]
    
    src_local = src_global - n_start_per_edge + batch_offset_per_edge
    dst_local = dst_global - n_start_per_edge + batch_offset_per_edge
    
    ei_batch = torch.stack([src_local, dst_local], dim=0)
    
    # Edge attributes
    ea_batch = ea[e_global]
    
    # Labels
    y_batch = y[indices] if y is not None else None
    
    return x_batch, ei_batch, ea_batch, batch_vec, y_batch, n_batch


def predict_batch(model, x, ei, ea, node_ptr, edge_ptr, indices):
    """Predict for a set of graph indices."""
    data = collate(x, ei, ea, node_ptr, edge_ptr, None, indices)
    if data is None:
        return torch.zeros(0)
    x_b, ei_b, ea_b, b_vec, _, n_g = data
    with torch.no_grad():
        logits = model(x_b, ei_b, ea_b, b_vec, n_g)
    return torch.sigmoid(logits).view(-1)


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    print(f"Loading data...", flush=True)
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    sizes = d["split_sizes"]
    n_train = sizes["train"]
    n_val = sizes["val"]
    n_test = sizes["test"]
    num_graphs = d["num_graphs"]
    in_dim = d["x"].size(1)
    edge_dim = d["edge_attr"].size(1) if "edge_attr" in d else 0
    out_dim = d["y"].size(1)

    print(f"Preparing data...", flush=True)
    x, ei, ea, node_ptr, edge_ptr, y = prepare_data(d)
    del d
    
    # Class weights
    y_tr = y[:n_train]
    valid_mask = ~torch.isnan(y_tr)
    pos_count = (y_tr[valid_mask] == 1).sum().item()
    neg_count = (y_tr[valid_mask] == 0).sum().item()
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)])
    print(f"Classes: pos={pos_count}, neg={neg_count}, weight={pos_weight.item():.2f}", flush=True)
    print(f"Data ready. {n_train} train, {n_val} val, {n_test} test graphs", flush=True)

    model = GIN(in_dim, edge_dim, HIDDEN, LAYERS, out_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val_auc = 0.0
    best_state = None
    bad_epochs = 0
    
    torch.set_num_threads(5)
    
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_samples = 0
        t_epoch = time.time()
        
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            data = collate(x, ei, ea, node_ptr, edge_ptr, y, idx)
            if data is None:
                continue
            x_b, ei_b, ea_b, b_vec, y_b, n_g = data
            
            opt.zero_grad()
            logits = model(x_b, ei_b, ea_b, b_vec, n_g)
            
            nan_mask = torch.isnan(y_b)
            if nan_mask.any():
                logits_m = logits[~nan_mask]
                y_m = y_b[~nan_mask]
            else:
                logits_m = logits
                y_m = y_b
            
            if y_m.numel() > 0:
                loss = F.binary_cross_entropy_with_logits(
                    logits_m.view(-1), y_m.view(-1), pos_weight=pos_weight
                )
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
                n_samples += len(idx)
        
        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        t_ep = time.time() - t_epoch
        
        # Val evaluation
        model.eval()
        val_idx = torch.arange(n_train, n_train + n_val)
        val_pred = predict_batch(model, x, ei, ea, node_ptr, edge_ptr, val_idx)
        val_true = y[n_train:n_train + n_val].view(-1)
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_true.numpy(), val_pred.numpy())
        
        print(f"Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_auc={val_auc:.4f} t={t_ep:.1f}s total={time.time()-t0:.1f}s", flush=True)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE and epoch >= 3:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break
    
    # Load best
    model.load_state_dict(best_state)
    model.eval()
    
    # Predict on target split
    if a.split == "val":
        target_idx = torch.arange(n_train, n_train + n_val)
    else:
        target_idx = torch.arange(n_train + n_val, num_graphs)
    
    pred = predict_batch(model, x, ei, ea, node_ptr, edge_ptr, target_idx)
    torch.save({"pred": pred}, Path(a.out))
    print(f"Wrote {len(pred)} predictions -> {a.out}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
