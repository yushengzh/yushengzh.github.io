#!/usr/bin/env python3
"""GIN for ogbg_molpcba: multi-task binary classification, AP metric.
v4: fast collate via prefix sums, sqrt pos_weight, larger batch."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 128
LAYERS = 5
HEAD_HIDDEN = 64
BATCH = 500
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42

ATOM_VOCAB = [120, 4, 7, 12, 8, 5, 6, 2, 2]
EDGE_VOCAB = [4, 3, 2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


def build_tensors(x_np, ei_np, ea_np, node_ptr, edge_ptr, g0, g1):
    """Pre-build contiguous tensors for graph range [g0, g1). Local node/edge ids.
    Returns dict with x, ei, ea, node_batch, edge_batch, and prefix sums."""
    n0 = int(node_ptr[g0]); n1 = int(node_ptr[g1])
    e0 = int(edge_ptr[g0]); e1 = int(edge_ptr[g1])
    Ntot = n1 - n0
    Etot = e1 - e0
    B = g1 - g0
    n_per = (node_ptr[g0+1:g1+1] - node_ptr[g0:g1]).astype(np.int64)
    node_batch = np.repeat(np.arange(B, dtype=np.int64), n_per)
    e_per = (edge_ptr[g0+1:g1+1] - edge_ptr[g0:g1]).astype(np.int64)
    edge_batch = np.repeat(np.arange(B, dtype=np.int64), e_per)
    x = x_np[n0:n1].astype(np.float32)
    eid_global = np.arange(e0, e1, dtype=np.int64)
    src_g = ei_np[0, eid_global]; dst_g = ei_np[1, eid_global]
    n0_rep = node_ptr[g0 + edge_batch]
    src_l = (src_g - n0_rep).astype(np.int64)
    dst_l = (dst_g - n0_rep).astype(np.int64)
    ei = np.stack([src_l, dst_l], axis=0).astype(np.int64)
    ea = ea_np[eid_global].astype(np.float32)
    # Prefix sums for fast batch range lookup
    node_csum = np.concatenate([[0], np.cumsum(n_per)])
    edge_csum = np.concatenate([[0], np.cumsum(e_per)])
    return {
        "x": torch.from_numpy(x),
        "ei": torch.from_numpy(ei),
        "ea": torch.from_numpy(ea),
        "node_batch": torch.from_numpy(node_batch.astype(np.int32)),
        "edge_batch": torch.from_numpy(edge_batch.astype(np.int32)),
        "N": Ntot, "E": Etot, "B": B,
        "node_csum": node_csum,  # (B+1,) int64
        "edge_csum": edge_csum,  # (B+1,) int64
    }


def collate_fast(T, indices, device):
    """Gather nodes/edges for given local graph ids. O(batch) not O(N)."""
    idx = np.asarray(indices, dtype=np.int64)
    # node ranges: [node_csum[idx[i]], node_csum[idx[i]+1])
    n_starts = T["node_csum"][idx]
    n_ends = T["node_csum"][idx + 1]
    e_starts = T["edge_csum"][idx]
    e_ends = T["edge_csum"][idx + 1]
    # node ranges
    n_total = int(n_ends[-1] - n_starts[0]) if len(idx) > 0 else 0
    # For simplicity: gather via boolean mask or direct concat
    # Direct concat per graph is O(batch * avg_size) which is fine
    if n_total == 0:
        x = T["x"][0:0]
        batch = torch.zeros(0, dtype=torch.long)
    else:
        x_list = []
        batch_list = []
        for i, g in enumerate(idx):
            ns = int(n_starts[i]); ne = int(n_ends[i])
            x_list.append(T["x"][ns:ne])
            batch_list.append(torch.full((ne - ns,), i, dtype=torch.long))
        x = torch.cat(x_list, dim=0)
        batch = torch.cat(batch_list, dim=0)
    # edge ranges
    e_total = int(e_ends[-1] - e_starts[0]) if len(idx) > 0 else 0
    if e_total == 0:
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = T["ea"][0:0]
        edge_batch_t = torch.zeros(0, dtype=torch.long)
    else:
        ei_list = []
        ea_list = []
        eb_list = []
        for i, g in enumerate(idx):
            es = int(e_starts[i]); ee = int(e_ends[i])
            ei_list.append(T["ei"][:, es:ee])
            ea_list.append(T["ea"][es:ee])
            eb_list.append(torch.full((ee - es,), i, dtype=torch.long))
        ei = torch.cat(ei_list, dim=1)
        ea = torch.cat(ea_list, dim=0)
        edge_batch_t = torch.cat(eb_list, dim=0)
    return x, ei, ea, batch


class GINConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_bond = nn.Embedding(EDGE_VOCAB[0], hidden)
        self.edge_stereo = nn.Embedding(EDGE_VOCAB[1], hidden)
        self.edge_ring = nn.Embedding(EDGE_VOCAB[2], hidden)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.size(1) > 0:
            row, col = edge_index[0], edge_index[1]
            bh = self.edge_bond(edge_attr[:, 0].long().clamp(0, EDGE_VOCAB[0] - 1))
            sh = self.edge_stereo(edge_attr[:, 1].long().clamp(0, EDGE_VOCAB[1] - 1))
            rh = self.edge_ring(edge_attr[:, 2].long().clamp(0, EDGE_VOCAB[2] - 1))
            edge_h = bh + sh + rh
        else:
            row = torch.zeros(0, dtype=torch.long, device=x.device)
            col = torch.zeros(0, dtype=torch.long, device=x.device)
            edge_h = torch.zeros(0, x.size(1), device=x.device)
        msg = x[col] + edge_h
        agg = torch.zeros_like(x).index_add_(0, row, msg)
        counts = torch.zeros(x.size(0), device=x.device).index_add_(
            0, row, torch.ones(row.size(0), device=x.device)
        ).clamp(min=1)
        agg = agg / counts.unsqueeze(-1)
        out = self.mlp((1 + self.eps) * x + agg)
        return out


class GIN(nn.Module):
    def __init__(self, hidden, layers, out_dim, head_hidden):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(v, hidden) for v in ATOM_VOCAB])
        self.input_proj = nn.Linear(hidden * len(ATOM_VOCAB), hidden)
        self.convs = nn.ModuleList([GINConv(hidden) for _ in range(layers)])
        self.pool_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr):
        h = self.input_proj(torch.cat([self.embeds[c](x[:, c].long().clamp(0, ATOM_VOCAB[c] - 1))
                       for c in range(len(ATOM_VOCAB))], dim=-1))
        for c in self.convs:
            h = F.relu(c(h, edge_index, edge_attr))
        p = torch.zeros(batch.max().item() + 1, h.size(1), device=h.device).index_add_(0, batch, h)
        p = self.pool_norm(p)
        return self.head(p)


def predict_batched(model, T, g0, device, bs, offset=0):
    """Predict for graphs in T starting at global graph index g0."""
    model.eval()
    B = T["B"]
    all_preds = []
    with torch.no_grad():
        for i in range(0, B, bs):
            idx = np.arange(i, min(i + bs, B), dtype=np.int64)
            x, ei, ea, batch = collate_fast(T, idx, device)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); batch = batch.to(device)
            logits = model(x, ei, batch, ea)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0)


def main():
    args = parse_args()
    t0 = time.time()
    data = torch.load(Path(args.data_dir) / "dataset.pt", map_location="cpu")
    x_all = data["x"].numpy()
    ei_all = data["edge_index"].numpy()
    ea_all = data["edge_attr"].numpy()
    node_ptr = data["node_ptr"].numpy()
    edge_ptr = data["edge_ptr"].numpy()
    y_all = data["y"].numpy()
    split_sizes = data["split_sizes"]
    tr = int(split_sizes["train"]); va = int(split_sizes["val"]); te = int(split_sizes["test"])
    T = data.get("num_tasks", y_all.shape[1])
    out_dim = y_all.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"G={tr+va+te} tr={tr} va={va} te={te} T={out_dim} N={x_all.shape[0]} E={ei_all.shape[1]}", flush=True)
    print(f"device={device}", flush=True)

    print("Building train tensors...", flush=True)
    Ttr = build_tensors(x_all, ei_all, ea_all, node_ptr, edge_ptr, 0, tr)
    print("Building val tensors...", flush=True)
    Tva = build_tensors(x_all, ei_all, ea_all, node_ptr, edge_ptr, tr, tr + va)
    if args.split == "test":
        print("Building test tensors...", flush=True)
        Tte = build_tensors(x_all, ei_all, ea_all, node_ptr, edge_ptr, tr + va, tr + va + te)

    ytr = torch.from_numpy(y_all[:tr]).float().to(device)
    yva = torch.from_numpy(y_all[tr:tr + va]).float().to(device)

    # pos_weight: sqrt(n_neg/n_pos), capped at 10
    mask = ~torch.isnan(ytr)
    pos = (ytr > 0.5) & mask
    n_pos = pos.float().sum(0)
    n_neg = mask.float().sum(0) - n_pos
    pos_weight = (n_neg / n_pos.clamp(min=1)).sqrt().clamp(max=10.0).to(device)
    print(f"pos_weight sqrt min/med/max: {pos_weight.min().item():.2f}/{pos_weight.median().item():.2f}/{pos_weight.max().item():.2f}", flush=True)

    torch.manual_seed(SEED)
    model = GIN(HIDDEN, LAYERS, out_dim, HEAD_HIDDEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = tr // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)

    best_val_ap = -1.0; patience = 0; best_state = None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(tr, generator=torch.Generator().manual_seed(SEED + epoch))
        ep_loss = 0.0; ep_n = 0; n_batches = 0
        for i in range(0, tr, BATCH):
            bidx = perm[i:i + BATCH]
            idx = bidx.cpu().numpy()
            x, ei, ea, batch = collate_fast(Ttr, idx, device)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); batch = batch.to(device)
            yb = ytr[bidx].clone().nan_to_num(0.0)
            mbl = ~torch.isnan(ytr[bidx])
            logits = model(x, ei, batch, ea)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            mb = mbl.to(logits.dtype)
            loss = (loss * mb).sum() / mb.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            ep_loss += loss.item() * mb.sum().item(); ep_n += mb.sum().item()
            n_batches += 1
        avg_loss = ep_loss / max(ep_n, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} t={time.time()-t0:.0f}s batches={n_batches}", flush=True)
        if epoch > 1 and (epoch + 1) % 2 == 0:
            vp = predict_batched(model, Tva, tr, device, 500).cpu().numpy()
            yv = yva.cpu().numpy()
            pv = vp
            from sklearn.metrics import average_precision_score
            valid = ~np.isnan(yv); aps = []
            for t in range(out_dim):
                m = valid[:, t]
                if m.sum() < 2: continue
                yt = yv[m, t].astype(int)
                if len(set(yt.tolist())) < 2: continue
                aps.append(average_precision_score(yt, pv[m, t]))
            vap = float(np.mean(aps)) if aps else 0.0
            print(f"  val AP={vap:.4f}", flush=True)
            if vap > best_val_ap + 1e-4:
                best_val_ap = vap; patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 4 and epoch > 3:
                    print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if args.split == "val":
        target = predict_batched(model, Tva, tr, device, 500).cpu().numpy()
    else:
        target = predict_batched(model, Tte, tr + va, device, 500).cpu().numpy()
    torch.save({"pred": torch.from_numpy(target)}, Path(args.out))
    print(f"Wrote {len(target)} preds -> {args.out} (t={time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
