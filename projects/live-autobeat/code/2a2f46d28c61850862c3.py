#!/usr/bin/env python3
"""Frozen-embedding 2-layer GCN + small learnable pair head for ogbl-vessel link prediction.

Design rationale:
- Full-graph autograd backprop OOMs the 11GB cap on 3.54M nodes / 8.55M edges.
- So: compute node embedding h2 ONCE in torch.no_grad() (memory-light forward only),
  then train ONLY a small linear head on sampled train pair-features (BCE).
- Head features: [h_u, h_v, h_u*h_v, |h_u-h_v|, logdeg_u, logdeg_v, xn_u, xn_v].
- Score candidates with the trained head (higher = more likely a true edge).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
import torch
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(0)


def compute_embedding(x, src, dst, n, H, device):
    """2-layer GCN, no_grad, D^-1/2 A D^-1/2 normalization + residual + L2 norm."""
    deg = torch.zeros(n, device=device).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    dinv = deg.clamp(min=1e-8).pow(-0.5)
    scale = (dinv[src] * dinv[dst]).float()

    W1 = torch.randn(x.shape[1], H, device=device) * 0.1
    b1 = torch.zeros(H, device=device)
    W2 = torch.randn(H, H, device=device) * (2.0 / H) ** 0.5
    b2 = torch.zeros(H, device=device)

    h1 = F.relu(x @ W1 + b1)                      # [n, H]
    msgs = h1[src] * scale.unsqueeze(-1)           # [E, H]
    agg = torch.zeros(n, H, device=device)
    agg.index_add_(0, dst, msgs)
    h2 = F.relu((agg + h1) @ W2 + b2)              # [n, H]
    del msgs, agg
    torch.cuda.empty_cache()
    h2 = F.normalize(h2, dim=-1)
    return h2


def build_pair_feats(h2, xn, logdeg, u, v, H):
    hu = h2[u]
    hv = h2[v]
    xu = xn[u]
    xv = xn[v]
    return torch.cat([hu, hv, hu * hv, (hu - hv).abs(),
                      logdeg[u].unsqueeze(-1), logdeg[v].unsqueeze(-1),
                      xu, xv], dim=-1)  # [B, 2H + 2H + 2 + 3 + 3] = [B, 4H+8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] Loading data...", flush=True)
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    x = d["x"].float()
    n = d["num_nodes"]
    ei = d["edge_index"].long()
    src, dst = ei[0], ei[1]
    tr_pos = d["train_edges"].long()
    tr_neg = d["train_neg"].long()
    print(f"[{time.time()-t0:.1f}s] Data: {n} nodes, {src.numel()} edges, F={x.shape[1]}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature normalization: per-dim mean/std then L2
    xm = x.mean(0, keepdim=True)
    xs = x.std(0, keepdim=True).clamp(min=1e-6)
    xn = F.normalize((x - xm) / xs, dim=-1)
    x_g = xn.to(device)
    src_g = src.to(device)
    dst_g = dst.to(device)
    print(f"[{time.time()-t0:.1f}s] Normalized, on {device}", flush=True)

    # Degree (for feature)
    deg = torch.zeros(n).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    logdeg = torch.log1p(deg)
    logdeg_g = logdeg.to(device)

    # Frozen embedding
    H = 64
    print(f"[{time.time()-t0:.1f}s] Computing GCN embedding H={H} (no_grad)...", flush=True)
    with torch.no_grad():
        h2 = compute_embedding(x_g, src_g, dst_g, n, H, device)
    xn_g = x_g  # [n, 3]
    print(f"[{time.time()-t0:.1f}s] Embedding done. h2={tuple(h2.shape)}", flush=True)

    # Move train pairs to GPU
    tr_pos_g = tr_pos.to(device)
    tr_neg_g = tr_neg.to(device)
    n_pos = tr_pos_g.shape[1]
    n_neg = tr_neg_g.shape[1]

    # Internal held-out slice of TRAIN pairs (2%) for generalization monitoring
    hold = int(0.02 * n_pos)
    perm = torch.randperm(n_pos, device=device)
    hold_idx = perm[:hold]
    keep_idx = perm[hold:]
    hp_u = tr_pos_g[0, hold_idx]; hp_v = tr_pos_g[1, hold_idx]
    hn_u = tr_neg_g[0, hold_idx]; hn_v = tr_neg_g[1, hold_idx]
    kp_u = tr_pos_g[0, keep_idx]; kp_v = tr_pos_g[1, keep_idx]
    kn_u = tr_neg_g[0, keep_idx]; kn_v = tr_neg_g[1, keep_idx]
    print(f"[{time.time()-t0:.1f}s] Train: {n_pos} pos / {n_neg} neg; held-out {hold}", flush=True)

    def bce_batch(u_p, v_p, u_n, v_n, head):
        fp = build_pair_feats(h2, xn_g, logdeg_g, u_p, v_p, H)
        fn = build_pair_feats(h2, xn_g, logdeg_g, u_n, v_n, H)
        logits = head(torch.cat([fp, fn], dim=0))
        target = torch.cat([torch.ones_like(fp[:, 0]), torch.zeros_like(fn[:, 0])])
        return F.binary_cross_entropy_with_logits(logits, target)

    # Head: linear on 4H+8 features
    feat_dim = 4 * H + 6 + 2 * x.shape[1]  # 2H + 2H + 2 + F + F
    head = torch.nn.Linear(feat_dim, 1, device=device)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-5)
    print(f"[{time.time()-t0:.1f}s] Head feat_dim={feat_dim}. Training...", flush=True)

    epochs = 25
    batch = 262144  # pairs per side (512K pairs total)
    for ep in range(epochs):
        perm_p = torch.randperm(n_pos, device=device)[hold:]
        perm_n = torch.randperm(n_neg, device=device)
        loss_sum = 0.0
        cnt = 0
        for i in range(0, n_pos - hold, batch):
            pi = perm_p[i:i + batch]
            ni = perm_n[i:i + batch]
            if ni.numel() == 0:
                continue
            opt.zero_grad()
            loss = bce_batch(kp_u[pi], kp_v[pi], kn_u[ni], kn_v[ni], head)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * pi.numel()
            cnt += pi.numel()
        # internal held-out val AUC-ish (quick)
        with torch.no_grad():
            lp = head(build_pair_feats(h2, xn_g, logdeg_g, hp_u, hp_v, H)).squeeze(-1)
            ln = head(build_pair_feats(h2, xn_g, logdeg_g, hn_u, hn_v, H)).squeeze(-1)
            lall = torch.cat([lp, ln])
            tall = torch.cat([torch.ones_like(lp), torch.zeros_like(ln)])
            # quick rank-based AUC
            rank = lall.argsort().argsort().float()
            n_pos_h = lp.numel(); n_neg_h = ln.numel()
            auc = ((rank[tall.bool()].sum() - n_pos_h * (n_pos_h + 1) / 2) / (n_pos_h * n_neg_h)).item()
        print(f"  ep{ep+1}/{epochs} loss={loss_sum/cnt:.4f} intval_auc={auc:.4f} [{time.time()-t0:.1f}s]", flush=True)

    # Score candidates
    if a.split == "val":
        pos_e, neg_e = d["val_edges"], d["val_neg"]
        if neg_e.dim() == 3:
            cand = torch.cat([pos_e.t().unsqueeze(1), neg_e], dim=1).reshape(-1, 2).t()
        else:
            cand = torch.cat([pos_e, neg_e], dim=1)
    else:
        cand = torch.load(Path(a.data_dir) / "test_inputs.pt")["candidates"]
    cand = cand.long().to(device)

    def score_pairs(u, v):
        return head(build_pair_feats(h2, xn_g, logdeg_g, u, v, H)).squeeze(-1)

    chunk = 262144
    scores = []
    with torch.no_grad():
        for i in range(0, cand.shape[1], chunk):
            e = cand[:, i:i + chunk]
            s = score_pairs(e[0], e[1])
            scores.append(s.cpu())
    scores = torch.cat(scores)
    torch.save({"scores": scores}, Path(a.out))
    print(f"[{time.time()-t0:.1f}s] Wrote {scores.numel()} scores -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
