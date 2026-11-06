#!/usr/bin/env python3
"""v6: Multi-seed architecture-diverse ensemble of Linear-input GIN (ogbg-molhiv).

Architecture = v2's strong Linear-input GIN (nn.Linear(9,HIDDEN) node input,
edge_mlp=Linear(3,HIDDEN) fed by raw bond features), trained per-seed with
cosine LR + early-stop on val AUC. K independent models' sigmoid outputs
are averaged (v4's ensemble scaffold). GPU + env-overridable hyperparams.
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
        self.vnode = nn.Parameter(torch.randn(1, hidden) * 0.02)
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
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        mean_h = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(0, batch, torch.ones(batch.size(0), device=h.device))
        counts = counts.clamp(min=1).unsqueeze(1)
        mean_h = mean_h / counts
        vn = mean_h + self.vnode
        pooled = pooled + vn
        return self.head(pooled)


def prepare_data(d):
    return (d["x"].float(), d["edge_index"].long(), d["edge_attr"].float(),
            d["node_ptr"].long(), d["edge_ptr"].long(), d["y"].float())


def collate(x, ei, ea, node_ptr, edge_ptr, y, indices):
    n_batch = len(indices)
    if n_batch == 0:
        return None
    n_start = node_ptr[indices]
    n_end = node_ptr[indices + 1]
    batch_sizes = n_end - n_start
    n_nodes = batch_sizes.sum().item()
    if n_nodes == 0:
        return None
    e_start = edge_ptr[indices]
    e_end = edge_ptr[indices + 1]
    edge_sizes = e_end - e_start
    n_edges = edge_sizes.sum().item()
    # node features
    x_batch = torch.cat([x[int(n_start[i]):int(n_end[i])] for i in range(n_batch)])
    # edge features
    ea_batch = torch.cat([ea[int(e_start[i]):int(e_end[i])] for i in range(n_batch)]) if n_edges else torch.zeros(0, 3)
    # batch vector
    batch_vec = torch.repeat_interleave(torch.arange(n_batch), batch_sizes)
    # local edge indices: gather per-graph, then re-base to local node indices
    if n_edges:
        src_slices = [ei[0][int(e_start[i]):int(e_end[i])] for i in range(n_batch)]
        dst_slices = [ei[1][int(e_start[i]):int(e_end[i])] for i in range(n_batch)]
        src_global = torch.cat(src_slices)
        dst_global = torch.cat(dst_slices)
        edge_graph_idx = torch.repeat_interleave(torch.arange(n_batch), edge_sizes)
        n_start_per_edge = n_start[edge_graph_idx]
        cum_nodes = torch.zeros(n_batch + 1, dtype=torch.long)
        cum_nodes[1:] = torch.cumsum(batch_sizes, 0)
        node_offset_per_edge = cum_nodes[edge_graph_idx]
        src_local = src_global - n_start_per_edge + node_offset_per_edge
        dst_local = dst_global - n_start_per_edge + node_offset_per_edge
        ei_batch = torch.stack([src_local, dst_local], dim=0)
    else:
        ei_batch = torch.zeros(2, 0, dtype=torch.long)
    y_batch = y[indices] if y is not None else None
    return x_batch, ei_batch, ea_batch, batch_vec, y_batch, n_batch


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
        all_pred.append(torch.sigmoid(logits).view(-1, 1).cpu())
    return torch.cat(all_pred) if all_pred else torch.zeros(0)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def train_one(seed, x, ei, ea, node_ptr, edge_ptr, y, n_train, n_val, device, pos_weight):
    set_seed(seed)
    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    model = GIN(9, 3, HIDDEN, LAYERS, 1, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_epochs = EPOCHS
    warmup = 2
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_auc = 0.0
    best_state = None
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for start in range(0, n_train, BATCH):
            batch_indices = perm[start:start + BATCH]
            data = collate(x, ei, ea, node_ptr, edge_ptr, y, batch_indices)
            if data is None:
                continue
            x_b, ei_b, ea_b, b_vec, y_b, n_g = data
            x_b = x_b.to(device); ei_b = ei_b.to(device); ea_b = ea_b.to(device); b_vec = b_vec.to(device); y_b = y_b.to(device)
            logits = model(x_b, ei_b, ea_b, b_vec, n_g)
            valid = ~torch.isnan(y_b).any(dim=-1) if y_b.dim() > 1 else ~torch.isnan(y_b)
            if valid.any():
                loss = F.binary_cross_entropy_with_logits(logits[valid].view(-1), y_b[valid].view(-1), pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss_sum += loss.item() * valid.sum().item()
        scheduler.step()
        avg_loss = loss_sum / max(1, n_train)
        # Val evaluation
        model.eval()
        val_pred = predict(model, x, ei, ea, node_ptr, edge_ptr, val_idx, device)
        val_true = y[n_train:n_train + n_val].view(-1)
        val_auc = roc_auc_score(val_true.numpy(), val_pred.numpy())
        cur_lr = optimizer.param_groups[0]['lr']
        t_ep = time.time() - t0
        print(f"Epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f} val_auc={val_auc:.4f} lr={cur_lr:.6f} t={t_ep:.1f}s", flush=True)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE and epoch >= 3:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, best_val_auc


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    a = parser.parse_args()
    torch.set_num_threads(5)
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu", weights_only=False)
    x, ei, ea, node_ptr, edge_ptr, y = prepare_data(d)
    n_train, n_val, n_test = d["split_sizes"]["train"], d["split_sizes"]["val"], d["split_sizes"]["test"]
    num_graphs = d["num_graphs"]
    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    test_idx = torch.arange(n_train + n_val, num_graphs)
    if a.split == "val":
        target_idx = val_idx
    else:
        target_idx = test_idx
    # pos_weight from TRAIN split
    ty = y[:n_train].view(-1)
    pos = (ty > 0.5).sum().item()
    neg = (ty < 0.5).sum().item()
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], device=device)
    print(f"Classes: pos={pos}, neg={neg}, weight={pos_weight.item():.2f}", flush=True)
    print(f"device={device} K={K_SEEDS}", flush=True)
    base_seeds = [42, 1, 7, 13, 23, 31, 43, 53, 61, 71][:K_SEEDS]
    print(f"Ensemble with K={K_SEEDS} seeds: {base_seeds}", flush=True)
    # v6: per-seed architecture diversity (deeper + more dropout) to broaden the ensemble
    per_layers  = [6, 8, 6, 10, 8, 6, 8, 10, 6, 8]
    per_dropout = [0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.2, 0.2, 0.1, 0.3]
    all_preds = []
    val_preds = []
    for i, seed in enumerate(base_seeds):
        os.environ["SOLVER_LAYERS"]   = str(per_layers[i])
        os.environ["SOLVER_DROPOUT"]  = str(per_dropout[i])
        t_seed = time.time()
        model, best_auc = train_one(seed, x, ei, ea, node_ptr, edge_ptr, y, n_train, n_val, device, pos_weight)
        print(f"  seed {seed}: LAYERS={per_layers[i]} DROPOUT={per_dropout[i]}", flush=True)
        pred = predict(model, x, ei, ea, node_ptr, edge_ptr, target_idx, device)
        all_preds.append(pred)
        vp = predict(model, x, ei, ea, node_ptr, edge_ptr, val_idx, device)
        val_preds.append(vp)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Seed {seed} ({i+1}/{K_SEEDS}): best_val_auc={best_auc:.4f} t={time.time()-t_seed:.1f}s", flush=True)
    avg_pred = torch.stack(all_preds, dim=0).mean(dim=0)
    torch.save({"pred": avg_pred.cpu()}, Path(a.out))
    # ensemble val AUC (only meaningful if val labels exist)
    val_true = y[n_train:n_train + n_val].view(-1)
    if not torch.isnan(val_true).any():
        avg_val = torch.stack(val_preds, dim=0).mean(dim=0)
        ens_val_auc = roc_auc_score(val_true.numpy(), avg_val.numpy())
        print(f"Ensemble val AUC = {ens_val_auc:.4f}", flush=True)
    print(f"Wrote {len(avg_pred)} predictions -> {a.out}", flush=True)
    print("Done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
