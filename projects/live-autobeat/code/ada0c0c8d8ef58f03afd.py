#!/usr/bin/env python3
"""Strong GNN for ogbg-molhiv: GIN with edge features, learnable eps, virtual node, sum pooling."""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"
HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 256))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 30))
LR = float(os.environ.get("SOLVER_LR", 5e-4))
BATCH = int(os.environ.get("SOLVER_BATCH", 1024))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 6))
WEIGHT_DECAY = 1e-4
PATIENCE = 8
DROPOUT = float(os.environ.get("SOLVER_DROPOUT", 0.1))


class GINLayer(nn.Module):
    def __init__(self, dim, edge_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.edge_mlp = nn.Linear(edge_dim, dim)
        self.act = nn.ReLU()
        self.eps = nn.Parameter(torch.ones(1))  # learnable, init 1.0

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        h = x[col]
        if edge_attr is not None:
            h = h + self.edge_mlp(edge_attr)
        msg = torch.zeros_like(x).index_add_(0, row, h)
        x = self.act(self.mlp((1 + self.eps) * x + msg))
        return x


class GIN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden, n_layers, out_dim, dropout=0.0):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GINLayer(hidden, edge_dim, dropout) for _ in range(n_layers)])
        self.vnode = nn.Parameter(torch.randn(1, hidden) * 0.02)  # virtual node init
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch, n_graphs):
        h = F.relu(self.input_lin(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        # Sum pooling
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        # Virtual node interaction: mean node features + virtual node
        mean_h = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones(batch.size(0), device=h.device))
        counts = counts.clamp(min=1).unsqueeze(1)
        mean_h = mean_h / counts
        vn = mean_h + self.vnode
        # Virtual node message: virtual node broadcasts to all nodes (approximated at graph level)
        pooled = pooled + vn
        return self.head(pooled)


def prepare_data(d):
    x = d["x"].float()
    ei = d["edge_index"].long()
    ea = d["edge_attr"].float()
    node_ptr = d["node_ptr"].long()
    edge_ptr = d["edge_ptr"].long()
    y = d["y"].float()
    return x, ei, ea, node_ptr, edge_ptr, y


def collate(x, ei, ea, node_ptr, edge_ptr, y, indices):
    n_batch = len(indices)
    if n_batch == 0:
        return None
    n_start = node_ptr[indices]
    n_end = node_ptr[indices + 1]
    e_start = edge_ptr[indices]
    e_end = edge_ptr[indices + 1]
    n_nodes = (n_end - n_start).sum().item()
    n_edges = (e_end - e_start).sum().item()
    batch_sizes = (n_end - n_start)
    batch_vec = torch.repeat_interleave(torch.arange(n_batch), batch_sizes)
    node_indices = torch.cat([torch.arange(s, e) for s, e in zip(n_start, n_end)])
    x_batch = x[node_indices]
    edge_node_offsets = torch.zeros(n_batch + 1, dtype=torch.long)
    edge_node_offsets[1:] = torch.cumsum(batch_sizes, 0)
    e_global = torch.cat([torch.arange(s, e) for s, e in zip(e_start, e_end)])
    src_global = ei[0, e_global]
    dst_global = ei[1, e_global]
    edge_graph_idx = torch.zeros(n_edges, dtype=torch.long)
    cum_e = 0
    for i in range(n_batch):
        n_e = (e_end[i] - e_start[i]).item()
        edge_graph_idx[cum_e:cum_e+n_e] = i
        cum_e += n_e
    n_start_per_edge = n_start[edge_graph_idx]
    batch_offset_per_edge = edge_node_offsets[edge_graph_idx]
    src_local = src_global - n_start_per_edge + batch_offset_per_edge
    dst_local = dst_global - n_start_per_edge + batch_offset_per_edge
    ei_batch = torch.stack([src_local, dst_local], dim=0)
    ea_batch = ea[e_global]
    y_batch = y[indices] if y is not None else None
    return x_batch, ei_batch, ea_batch, batch_vec, y_batch, n_batch


def predict_batch(model, x, ei, ea, node_ptr, edge_ptr, indices):
    data = collate(x, ei, ea, node_ptr, edge_ptr, None, indices)
    if data is None:
        return torch.zeros(0)
    x_b, ei_b, ea_b, b_vec, _, n_g = data
    with torch.no_grad():
        logits = model(x_b, ei_b, ea_b, b_vec, n_g)
    return torch.sigmoid(logits).view(-1)


def train():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    a = parser.parse_args()

    torch.set_num_threads(5)

    print("Loading data...", flush=True)
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    print("Preparing data...", flush=True)
    x, ei, ea, node_ptr, edge_ptr, y = prepare_data(d)
    split_sizes = d["split_sizes"]
    n_train, n_val, n_test = split_sizes["train"], split_sizes["val"], split_sizes["test"]
    num_graphs = d["num_graphs"]

    # Class weights
    y_train = y[:n_train]
    valid_train = ~torch.isnan(y_train)
    y_train_flat = y_train[valid_train].view(-1)
    pos = (y_train_flat > 0.5).sum().item()
    neg = (y_train_flat < 0.5).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)])
    print(f"Classes: pos={pos}, neg={neg}, weight={pos_weight.item():.2f}", flush=True)
    print(f"Data ready. {n_train} train, {n_val} val, {n_test} test graphs", flush=True)

    model = GIN(9, 3, HIDDEN, LAYERS, 1, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Learning rate schedule: linear warmup for 2 epochs, then cosine decay
    warmup_epochs = 2
    total_epochs = EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    val_true = y[n_train:n_train + n_val].view(-1)

    best_val_auc = 0.0
    best_state = None
    bad_epochs = 0

    for epoch in range(EPOCHS):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH):
            end = min(start + BATCH, n_train)
            batch_indices = train_idx[perm[start:end]]
            data = collate(x, ei, ea, node_ptr, edge_ptr, y, batch_indices)
            if data is None:
                continue
            x_b, ei_b, ea_b, b_vec, y_b, n_g = data
            y_b = y_b.view(-1)
            valid = ~torch.isnan(y_b)
            if valid.sum() == 0:
                continue
            optimizer.zero_grad()
            logits = model(x_b, ei_b, ea_b, b_vec, n_g).view(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], y_b[valid], pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        t_ep = time.time() - t_ep

        # Val evaluation
        model.eval()
        val_pred = predict_batch(model, x, ei, ea, node_ptr, edge_ptr, val_idx)
        val_true = y[n_train:n_train + n_val].view(-1)
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_true.numpy(), val_pred.numpy())

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_auc={val_auc:.4f} lr={cur_lr:.6f} t={t_ep:.1f}s total={time.time()-t0:.1f}s", flush=True)

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
