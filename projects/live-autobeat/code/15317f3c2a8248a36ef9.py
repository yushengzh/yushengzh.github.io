#!/usr/bin/env python3
"""v4: Multi-seed ensemble of clamped embeddings-GIN for ogbg-molhiv.
Trains K independent models (different seeds), averages their sigmoid
predictions. Defensive clamping ensures no IndexError on unseen values.
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def _pick_device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda"
    except Exception:
        pass
    return "cpu"

device = _pick_device()

# Hyperparameters (env-overridable)
HIDDEN   = int(os.environ.get("SOLVER_HIDDEN", 256))
EPOCHS   = int(os.environ.get("SOLVER_EPOCHS", 30))
LR       = float(os.environ.get("SOLVER_LR", 5e-4))
BATCH    = int(os.environ.get("SOLVER_BATCH", 2048))
LAYERS   = int(os.environ.get("SOLVER_LAYERS", 6))
WEIGHT_DECAY = float(os.environ.get("SOLVER_WD", 1e-4))
PATIENCE = int(os.environ.get("SOLVER_PATIENCE", 8))
DROPOUT  = float(os.environ.get("SOLVER_DROPOUT", 0.1))
K_SEEDS  = int(os.environ.get("SOLVER_KSEEDS", 6))

# Defensive atom/edge sizes: use generous max so no clamp needed in practice,
# but clamp anyway for absolute safety.
ATOM_SIZES = [128, 4, 16, 8, 6, 4, 2, 2, 2]  # generous canonical upper bounds
EDGE_SIZES = [6, 2, 3]                       # generous for bond_type, conjugated, stereo

class EmbedInput(nn.Module):
    def __init__(self, atom_sizes, edge_sizes, hidden):
        super().__init__()
        self.atom_embs = nn.ModuleList([nn.Embedding(sz, hidden) for sz in atom_sizes])
        self.edge_embs = nn.ModuleList([nn.Embedding(sz, hidden) for sz in edge_sizes])
        self.proj = nn.Linear(len(atom_sizes) * hidden, hidden)
        self.edge_proj = nn.Linear(len(edge_sizes) * hidden, hidden)
    def forward(self, x, ea):
        x_l = x.long()
        embs = []
        for i, emb in enumerate(self.atom_embs):
            idx = x_l[:, i].clamp(0, emb.weight.size(0) - 1)
            embs.append(emb(idx))
        h = self.proj(torch.cat(embs, dim=-1))
        ea_l = ea.long()
        e_embs = []
        for i, emb in enumerate(self.edge_embs):
            idx = ea_l[:, i].clamp(0, emb.weight.size(0) - 1)
            e_embs.append(emb(idx))
        e = self.edge_proj(torch.cat(e_embs, dim=-1))
        return h, e

class GINLayer(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.eps = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        self.edge_mlp = nn.Linear(hidden, hidden)
    def forward(self, x, edge_index, e):
        src, dst = edge_index
        msg = x[src] + e[src]
        out = torch.zeros_like(x).index_add_(0, dst, msg)
        return F.relu(self.mlp((1 + self.eps) * x + out))

class GIN(nn.Module):
    def __init__(self, hidden, n_layers, dropout=0.1):
        super().__init__()
        self.embed = EmbedInput(ATOM_SIZES, EDGE_SIZES, hidden)
        self.layers = nn.ModuleList([GINLayer(hidden, dropout) for _ in range(n_layers)])
        self.vnode = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, edge_index, ea, batch, n_graphs):
        h, e = self.embed(x, ea)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones(batch.size(0), device=h.device))
        mean_h = pooled / counts.clamp(min=1).unsqueeze(1)
        pooled = pooled + mean_h + self.vnode
        return self.head(pooled)

def collate(x, ei, ea, node_ptr, edge_ptr, y, indices):
    n_batch = indices.size(0)
    if n_batch == 0:
        return None
    n_start = node_ptr[indices]; n_end = node_ptr[indices + 1]
    e_start = edge_ptr[indices]; e_end = edge_ptr[indices + 1]
    n_nodes = (n_end - n_start).sum().item()
    n_edges = (e_end - e_start).sum().item()
    if n_nodes == 0:
        return None
    batch_sizes = n_end - n_start
    batch_vec = torch.repeat_interleave(torch.arange(n_batch), batch_sizes)
    # Node gather
    n_global = torch.cat([torch.arange(s, e) for s, e in zip(n_start.tolist(), n_end.tolist())])
    x_batch = x[n_global]
    # Edge gather
    edge_sizes = e_end - e_start
    e_global = torch.cat([torch.arange(s, e) for s, e in zip(e_start.tolist(), e_end.tolist())])
    src_global = ei[0, e_global]
    dst_global = ei[1, e_global]
    ea_batch = ea[e_global]
    edge_graph_idx = torch.repeat_interleave(torch.arange(n_batch), edge_sizes)
    n_start_per_edge = n_start[edge_graph_idx]
    cum_nodes = torch.zeros(n_batch + 1, dtype=torch.long)
    cum_nodes[1:] = torch.cumsum(batch_sizes, 0)
    node_offset_per_edge = cum_nodes[edge_graph_idx]
    src_local = src_global - n_start_per_edge + node_offset_per_edge
    dst_local = dst_global - n_start_per_edge + node_offset_per_edge
    ei_batch = torch.stack([src_local, dst_local], dim=0)
    y_batch = y[indices] if y is not None else None
    return x_batch, ei_batch, ea_batch, batch_vec, y_batch, n_batch

def prepare_data(d):
    return (d["x"].float(), d["edge_index"].long(), d["edge_attr"].float(),
            d["node_ptr"].long(), d["edge_ptr"].long(), d["y"].float())

def predict(model, x, ei, ea, node_ptr, edge_ptr, indices, dev):
    model.eval()
    all_pred = []
    BS = 4096
    for start in range(0, len(indices), BS):
        idx = indices[start:start + BS]
        data = collate(x, ei, ea, node_ptr, edge_ptr, None, idx)
        if data is None:
            continue
        x_b, ei_b, ea_b, b_vec, _, n_g = data
        with torch.no_grad():
            logits = model(x_b.to(dev), ei_b.to(dev), ea_b.to(dev), b_vec.to(dev), n_g)
        all_pred.append(torch.sigmoid(logits).view(-1).cpu())
    return torch.cat(all_pred) if all_pred else torch.zeros(0)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def train_one(seed, x, ei, ea, node_ptr, edge_ptr, y, n_train, n_val, device, pos_weight):
    """Train one model with given seed, return model at best val epoch."""
    set_seed(seed)
    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    y_train = y[:n_train]
    valid_train = ~torch.isnan(y_train)
    y_flat = y_train[valid_train].view(-1)
    pos = (y_flat > 0.5).sum().item(); neg = (y_flat < 0.5).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    # Pre-collate val (small, do once)
    val_data = collate(x, ei, ea, node_ptr, edge_ptr, None, val_idx)
    val_x, val_ei, val_ea, val_bv, _, val_ng = val_data
    val_x = val_x.to(device); val_ei = val_ei.to(device); val_ea = val_ea.to(device); val_bv = val_bv.to(device)
    y_val = y[n_train:n_train+n_val]
    valid_val = ~torch.isnan(y_val)
    val_true_np = y_val[valid_val].view(-1).numpy()

    model = GIN(HIDDEN, LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(EPOCHS - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_auc = 0.0
    best_state = None
    bad_epochs = 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, BATCH):
            idx = perm[start:start + BATCH]
            data = collate(x, ei, ea, node_ptr, edge_ptr, y, idx)
            if data is None:
                continue
            x_b, ei_b, ea_b, b_vec, y_b, n_g = data
            x_b = x_b.to(device); ei_b = ei_b.to(device); ea_b = ea_b.to(device)
            b_vec = b_vec.to(device); y_b = y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b, ei_b, ea_b, b_vec, n_g)
            loss = F.binary_cross_entropy_with_logits(logits, y_b, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x, val_ei, val_ea, val_bv, val_ng)
        val_pred = torch.sigmoid(val_logits).view(-1).cpu().numpy()
        val_auc = roc_auc_score(val_true_np, val_pred)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE and epoch >= 3:
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, best_val_auc

def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    a = ap.parse_args()
    torch.set_num_threads(5)
    t0 = time.time()
    print(f"Device: {device}", flush=True)
    print("Loading data...", flush=True)
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu", weights_only=False)
    x, ei, ea, node_ptr, edge_ptr, y = prepare_data(d)
    n_train, n_val, n_test = d["split_sizes"]["train"], d["split_sizes"]["val"], d["split_sizes"]["test"]
    num_graphs = d["num_graphs"]

    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    test_idx = torch.arange(n_train + n_val, num_graphs)

    # Determine target
    if a.split == "val":
        target_idx = val_idx
    else:
        target_idx = test_idx

    # Seeds for ensemble
    base_seeds = [42, 1, 7, 13, 23, 31, 43, 53, 61, 71][:K_SEEDS]
    print(f"Ensemble with K={K_SEEDS} seeds: {base_seeds}", flush=True)

    all_preds = []
    for i, seed in enumerate(base_seeds):
        t_seed = time.time()
        model, best_auc = train_one(seed, x, ei, ea, node_ptr, edge_ptr, y, n_train, n_val, device, None)
        pred = predict(model, x, ei, ea, node_ptr, edge_ptr, target_idx, device)
        all_preds.append(pred)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Seed {seed} ({i+1}/{K_SEEDS}): best_val_auc={best_auc:.4f} t={time.time()-t_seed:.1f}s", flush=True)

    # Average predictions
    avg_pred = torch.stack(all_preds, dim=0).mean(dim=0)
    torch.save({"pred": avg_pred.cpu()}, Path(a.out))
    print(f"Wrote {len(avg_pred)} predictions -> {a.out}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(train())
