#!/usr/bin/env python3
"""GraphSAGE link prediction for ogbl-citation2.

3-layer GraphSAGE with hidden=256, batched training, cosine-similarity decoder.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 256))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 3))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 25))
BATCH = int(os.environ.get("SOLVER_BATCH", 100_000))
LR = float(os.environ.get("SOLVER_LR", 0.005))


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, row, col, w):
        # Aggregate neighbor features
        nbr = x[col] * w.unsqueeze(-1)  # [E, in_dim]
        # Mean aggregation via scatter
        agg = torch.zeros(x.size(0), x.size(1), device=x.device)
        agg.index_add_(0, row, nbr)
        # Compute mean (degree-based)
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
        deg = deg.clamp(min=1)
        agg = agg / deg
        # Concat with self
        out = torch.cat([x, agg], dim=-1)
        return F.relu(self.lin(out))


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for i in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))

    def forward(self, x, row, col, w):
        for conv in self.convs:
            x = conv(x, row, col, w)
        return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    data_dir = Path(a.data_dir)
    print(f"Loading data...", flush=True)
    d = torch.load(data_dir / "dataset.pt", map_location="cpu")
    x = d["x"].float().to(device)
    edge_idx = d["edge_index"].long().to(device)
    num_nodes = d["num_nodes"]
    tr_pos = d["train_edges"].long().to(device)
    tr_neg = d["train_neg"].long().to(device)
    print(f"Data loaded in {time.time()-t0:.1f}s: {num_nodes} nodes, {edge_idx.size(1)} edges, feat dim {x.size(1)}", flush=True)

    # Normalize edge weights
    row, col = edge_idx
    deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
    deg.index_add_(0, row, torch.ones(row.numel(), dtype=torch.float, device=device))
    dinv = deg.clamp(min=1).pow(-0.5)
    w = dinv[row] * dinv[col]
    print(f"Graph normalized.", flush=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    enc = SAGEEncoder(x.size(1), HIDDEN, LAYERS).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    n_tr = tr_pos.size(1)
    print(f"Training: {n_tr} pos, {n_tr} neg pairs, {EPOCHS} epochs, batch={BATCH}", flush=True)

    best_val = 0.0
    for ep in range(EPOCHS):
        enc.train()
        opt.zero_grad()
        z = enc(x, row, col, w)
        z_norm = F.normalize(z, dim=-1)

        # Batched loss computation
        perm = torch.randperm(n_tr, device=device)
        total_loss = 0.0
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i+BATCH]
            pos_score = (z_norm[tr_pos[0, idx]] * z_norm[tr_pos[1, idx]]).sum(-1)
            neg_score = (z_norm[tr_neg[0, idx]] * z_norm[tr_neg[1, idx]]).sum(-1)
            logits = torch.cat([pos_score, neg_score])
            target = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            batch_loss = F.binary_cross_entropy_with_logits(logits, target)
            batch_loss.backward()
            total_loss += batch_loss.item()
        opt.step()
        avg_loss = total_loss / (n_tr // BATCH + 1)

        # Quick val check every 5 epochs
        if (ep + 1) % 5 == 0:
            enc.eval()
            with torch.no_grad():
                z = enc(x, row, col, w)
                z_norm = F.normalize(z, dim=-1)
                val_pos = d["val_edges"].long().to(device)
                val_neg = d["val_neg"].to(device)  # [V, K, 2]
                # Score first 1000 val examples
                n_check = min(1000, val_pos.size(1))
                vp = val_pos[:, :n_check]
                vn = val_neg[:n_check]  # [n_check, K, 2]
                pos_s = (z_norm[vp[0]] * z_norm[vp[1]]).sum(-1)
                # Reshape neg: [n_check, K, 2] -> [n_check*K, 2]
                vn_flat = vn.view(-1, 2)
                neg_s = (z_norm[vn_flat[0]] * z_norm[vn_flat[1]]).sum(-1)
                neg_s = neg_s.view(n_check, -1)
                # MRR: for each pos, count how many neg scores are >= pos score
                ranks = (neg_s >= pos_s.unsqueeze(1)).sum(1) + 1
                mrr = (1.0 / ranks.float()).mean().item()
            print(f"epoch {ep+1}/{EPOCHS} loss={avg_loss:.4f} val_mrr={mrr:.4f}", flush=True)
            scheduler.step(mrr)
        else:
            print(f"epoch {ep+1}/{EPOCHS} loss={avg_loss:.4f}", flush=True)

    enc.eval()
    with torch.no_grad():
        z = enc(x, row, col, w)
        z_norm = F.normalize(z, dim=-1)

        if a.split == "val":
            pos, neg = d["val_edges"], d["val_neg"]
            if neg.dim() == 3:
                n_v = pos.size(1)
                k = neg.size(1)
                cand = torch.zeros(2, n_v * (k + 1), dtype=torch.long)
                # Interleave: pos_i, neg_i_0, neg_i_1, ...
                for i in range(n_v):
                    base = i * (k + 1)
                    cand[0, base] = pos[0, i]
                    cand[1, base] = pos[1, i]
                    cand[0, base+1:base+k+1] = neg[i, :, 0]
                    cand[1, base+1:base+k+1] = neg[i, :, 1]
            else:
                cand = torch.cat([pos, neg], dim=1)
            cand = cand.long().to(device)
        else:
            ti_path = data_dir / "test_inputs.pt"
            if not ti_path.exists():
                raise SystemExit(f"missing {ti_path}")
            cand = torch.load(ti_path, map_location="cpu")["candidates"].long().to(device)

        # Score in chunks to save memory
        scores_list = []
        CHUNK = 1_000_000
        for i in range(0, cand.size(1), CHUNK):
            c = cand[:, i:i+CHUNK]
            s = (z_norm[c[0]] * z_norm[c[1]]).sum(-1)
            scores_list.append(s.cpu())
        scores = torch.cat(scores_list)

    torch.save({"scores": scores}, Path(a.out))
    print(f"Wrote {scores.numel()} {a.split} scores -> {a.out} (total {time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
