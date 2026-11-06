#!/usr/bin/env python3
"""GIN for aqueous solubility regression (ogbg-molesol) - v2.

Improvements over v1:
- Uses ALL 9 atom feature dims via per-column embeddings (summed)
- Uses all 3 edge_attr dims via per-column embeddings (added to message)
- Correct edge_index handling (local per-graph indices offset into global)
- Clean full-batch training on GPU
- Cosine LR schedule with warmup, early stopping on val RMSE
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters ──────────────────────────────────────────────
HIDDEN = 128
LAYERS = 5
EPOCHS = 200
LR = 0.003
WD = 1e-4
PATIENCE = 25
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GINLayer(nn.Module):
    def __init__(self, dim, eps=1.0):
        super().__init__()
        self.eps = eps
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_attr, src, dst):
        # edge_attr: [E, dim] already computed
        msg = h[src] + edge_attr  # [E, dim]
        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        h_new = (1 + self.eps) * h + agg
        h_new = self.update(h_new)
        h_new = self.norm(h + h_new)  # residual + layer norm
        return h_new


class GINModel(nn.Module):
    def __init__(self, atom_dims, bond_dims, hidden=128, n_layers=5):
        """
        atom_dims: list of max_category+1 per column (9 cols, some may be 1)
        bond_dims: list of max_category+1 per column (3 cols)
        """
        super().__init__()
        self.hidden = hidden

        # Atom embeddings: per-column, summed
        self.atom_embeds = nn.ModuleList()
        for d in atom_dims:
            if d > 1:
                self.atom_embeds.append(nn.Embedding(d, hidden))
            else:
                self.atom_embeds.append(None)

        # Bond embeddings: per-column, summed
        self.bond_embeds = nn.ModuleList()
        for d in bond_dims:
            if d > 1:
                self.bond_embeds.append(nn.Embedding(d, hidden))
            else:
                self.bond_embeds.append(None)

        # GIN layers
        self.convs = nn.ModuleList([GINLayer(hidden) for _ in range(n_layers)])

        # Readout
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def embed_atoms(self, x):
        """x: [N, 9] long tensor. Returns [N, hidden]."""
        embs = []
        for c, emb in enumerate(self.atom_embeds):
            if emb is not None:
                embs.append(emb(x[:, c]))
            else:
                # constant feature - skip (adds nothing)
                pass
        if not embs:
            return torch.zeros(x.size(0), self.hidden, device=x.device)
        return sum(embs)

    def embed_bonds(self, edge_attr):
        """edge_attr: [E, 3] long tensor. Returns [E, hidden]."""
        embs = []
        for c, emb in enumerate(self.bond_embeds):
            if emb is not None:
                embs.append(emb(edge_attr[:, c]))
            else:
                pass
        if not embs:
            return torch.zeros(edge_attr.size(0), self.hidden, device=edge_attr.device)
        return sum(embs)

    def forward(self, x, edge_attr, edge_index, batch, n_graphs):
        """
        x: [N, 9] long
        edge_attr: [E, 3] long
        edge_index: [2, E] long (local to global batched nodes)
        batch: [N] long (graph id per node)
        n_graphs: int
        """
        h = self.embed_atoms(x)  # [N, hidden]
        bond_h = self.embed_bonds(edge_attr)  # [E, hidden]

        src = edge_index[1]  # column = source
        dst = edge_index[0]  # row = destination

        for conv in self.convs:
            h = conv(h, bond_h, src, dst)

        # Pooling: mean, max, sum
        pooled_mean = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_mean / counts

        pooled_max = torch.full((n_graphs, h.size(1)), -1e9, device=h.device)
        pooled_max.scatter_reduce_(0, batch.unsqueeze(-1).expand(-1, h.size(1)), h, reduce="amax", include_self=False)

        pooled_sum = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)

        pooled = torch.cat([pooled_mean, pooled_max, pooled_sum], dim=-1)
        return self.head(pooled)


def make_batch(d, indices):
    """
    Build a batched tensor for the given graph indices.
    Returns: x [N,9], edge_attr [E,3], edge_index [2,E], batch [N], n_graphs
    """
    atom_parts = []
    edge_parts = []
    bond_parts = []
    batch_parts = []

    for gi in range(len(indices)):
        orig_i = indices[gi]
        n0 = int(d["node_ptr"][orig_i])
        n1 = int(d["node_ptr"][orig_i + 1])
        e0 = int(d["edge_ptr"][orig_i])
        e1 = int(d["edge_ptr"][orig_i + 1])
        n_nodes = n1 - n0
        n_edges = e1 - e0

        if n_nodes > 0:
            # Atom features: [n_nodes, 9]
            atom_parts.append(d["x"][n0:n1].long())
            batch_parts.append(torch.full((n_nodes,), gi, dtype=torch.long))

        if n_edges > 0:
            # Edge index: [2, n_edges] - already global, offset to local batch
            ei = d["edge_index"][:, e0:e1].long()
            # These are global node indices. We need to offset them to be
            # local to the current batched node space.
            # The global node index for this graph starts at n0.
            # In the batched tensor, this graph's nodes start at current offset.
            # We'll handle offsetting after collecting all parts.
            edge_parts.append(ei)
            bond_parts.append(d["edge_attr"][e0:e1].long())

    # Concatenate atoms
    if atom_parts:
        x = torch.cat(atom_parts, dim=0)  # [N, 9]
        batch = torch.cat(batch_parts, dim=0)  # [N]
    else:
        x = torch.zeros(0, 9, dtype=torch.long)
        batch = torch.zeros(0, dtype=torch.long)

    # Handle edges with proper offsetting
    if edge_parts:
        # Build offset mapping: for each graph in the batch, what is the
        # global->local node offset?
        # We need to know the cumulative node offset for each graph.
        # Re-compute:
        offsets = []
        cur = 0
        for gi in range(len(indices)):
            orig_i = indices[gi]
            n0 = int(d["node_ptr"][orig_i])
            n1 = int(d["node_ptr"][orig_i + 1])
            offsets.append((cur, n0))
            cur += (n1 - n0)

        # Offset each graph's edge indices
        all_ei = []
        for gi in range(len(indices)):
            orig_i = indices[gi]
            e0 = int(d["edge_ptr"][orig_i])
            e1 = int(d["edge_ptr"][orig_i + 1])
            n_edges = e1 - e0
            if n_edges > 0:
                ei = d["edge_index"][:, e0:e1].long()
                local_start, global_start = offsets[gi]
                ei = ei - global_start + local_start
                all_ei.append(ei)
        if all_ei:
            edge_index = torch.cat(all_ei, dim=1)  # [2, E]
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    if bond_parts:
        edge_attr = torch.cat(bond_parts, dim=0)  # [E, 3]
    else:
        edge_attr = torch.zeros(0, 3, dtype=torch.long)

    return x, edge_attr, edge_index, batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    t0 = time.time()
    d = torch.load(Path(args.data_dir) / "dataset.pt", map_location="cpu")

    n_train = d["split_sizes"]["train"]
    n_val = d["split_sizes"]["val"]
    n_test = d["split_sizes"]["test"]

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_train + n_val + n_test))

    # Compute embedding sizes from the data
    x_all = d["x"]
    edge_attr_all = d["edge_attr"]

    # For each column, max value + 1 (to use as embedding size)
    atom_dims = []
    for c in range(x_all.shape[1]):
        mx = int(x_all[:, c].max().item()) + 1
        atom_dims.append(mx)

    bond_dims = []
    for c in range(edge_attr_all.shape[1]):
        mx = int(edge_attr_all[:, c].max().item()) + 1
        bond_dims.append(mx)

    print(f"Atom dims: {atom_dims}", flush=True)
    print(f"Bond dims: {bond_dims}", flush=True)

    # Build batches
    print("Building train batch...", flush=True)
    train_x, train_ea, train_ei, train_batch = make_batch(d, train_idx)
    print(f"  N={train_x.size(0)}, E={train_ea.size(0)}, graphs={n_train}", flush=True)

    print("Building val batch...", flush=True)
    val_x, val_ea, val_ei, val_batch = make_batch(d, val_idx)
    print(f"  N={val_x.size(0)}, E={val_ea.size(0)}, graphs={n_val}", flush=True)

    print("Building test batch...", flush=True)
    test_x, test_ea, test_ei, test_batch = make_batch(d, test_idx)
    print(f"  N={test_x.size(0)}, E={test_ea.size(0)}, graphs={n_test}", flush=True)

    # Train labels
    y_train = d["y"][:n_train].squeeze(-1).to(DEVICE)
    y_val = d["y"][n_train:n_train + n_val].squeeze(-1)

    # Move to device
    train_x = train_x.to(DEVICE)
    train_ea = train_ea.to(DEVICE)
    train_ei = train_ei.to(DEVICE)
    train_batch = train_batch.to(DEVICE)

    val_x = val_x.to(DEVICE)
    val_ea = val_ea.to(DEVICE)
    val_ei = val_ei.to(DEVICE)
    val_batch = val_batch.to(DEVICE)

    test_x = test_x.to(DEVICE)
    test_ea = test_ea.to(DEVICE)
    test_ei = test_ei.to(DEVICE)
    test_batch = test_batch.to(DEVICE)

    # Model
    model = GINModel(atom_dims, bond_dims, HIDDEN, LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Cosine schedule with linear warmup
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_rmse = float("inf")
    best_model_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x, train_ea, train_ei, train_batch, n_train).squeeze(-1)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x, val_ea, val_ei, val_batch, n_val).squeeze(-1)
            val_rmse = float(torch.sqrt(F.mse_loss(val_pred, y_val.to(DEVICE))))

        if (epoch + 1) % 10 == 0 or epoch < 3:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1}/{EPOCHS} loss={float(loss):.4f} val_rmse={val_rmse:.4f} t={elapsed:.1f}s", flush=True)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch+1}, best val_rmse={best_val_rmse:.4f}", flush=True)
                break

    print(f"Best val RMSE: {best_val_rmse:.4f} after {epoch+1} epochs", flush=True)

    # Restore best model
    model.load_state_dict(best_model_state)
    model.eval()

    # Predict
    with torch.no_grad():
        if args.split == "val":
            pred = model(val_x, val_ea, val_ei, val_batch, n_val)
        else:
            pred = model(test_x, test_ea, test_ei, test_batch, n_test)

    torch.save({"pred": pred.cpu()}, Path(args.out))
    print(f"Wrote predictions for {args.split} ({pred.numel()} graphs) -> {args.out}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
