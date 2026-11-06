#!/usr/bin/env python3
"""Frozen 2-layer GCN embedding + linear pair head for ogbl-vessel link prediction.
v3: no pre-built FP/FN; build pair feats on-the-fly per batch to stay under 11GB."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
import torch
import torch.nn.functional as F
from pathlib import Path

torch.manual_seed(0)


def compute_embedding(x, src, dst, n, H, device):
    """2-layer GCN, no_grad, D^-1/2 A D^-1/2 + residual + L2 norm."""
    deg = torch.zeros(n, device=device).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    dinv = deg.clamp(min=1e-8).pow(-0.5)
    scale = (dinv[src] * dinv[dst]).float()

    W1 = torch.randn(x.shape[1], H, device=device) * 0.1
    b1 = torch.zeros(H, device=device)
    W2 = torch.randn(H, H, device=device) * (2.0 / H) ** 0.5
    b2 = torch.zeros(H, device=device)

    h1 = F.relu(x @ W1 + b1)
    msgs = h1[src] * scale.unsqueeze(-1)
    agg = torch.zeros(n, H, device=device)
    agg.index_add_(0, dst, msgs)
    del msgs
    h2 = F.relu((agg + h1) @ W2 + b2)
    del agg
    torch.cuda.empty_cache()
    h2 = F.normalize(h2, dim=-1)
    return h2, deg


def build_pair_feats(h2, xn, logdeg, u, v, H):
    hu = h2[u]
    hv = h2[v]
    xu = xn[u]
    xv = xn[v]
    return torch.cat([hu, hv, hu * hv, (hu - hv).abs(),
                      logdeg[u].unsqueeze(-1), logdeg[v].unsqueeze(-1),
                      xu, xv], dim=-1)


def rank_auc(scores, is_pos):
    s = scores.float().cpu()
    order = s.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float32, device=s.device)
    np_ = is_pos.sum().item(); nn_ = (1 - is_pos.sum()).item()
    if np_ == 0 or nn_ == 0:
        return 0.5
    return ((ranks[is_pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.time()-t0:.1f}s] device={device}. Loading data...", flush=True)
    d = torch.load(Path(a.data_dir) / "dataset.pt")
    x = d["x"].float()
    n = d["num_nodes"]
    ei = d["edge_index"].long()
    src, dst = ei[0], ei[1]
    tr_pos = d["train_edges"].long()
    tr_neg = d["train_neg"].long()
    print(f"[{time.time()-t0:.1f}s] Data: {n} nodes, {src.numel()} edges, F={x.shape[1]}", flush=True)

    # Standardize + L2 normalize node features
    xm = x.mean(dim=0, keepdim=True)
    xs = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    xn = F.normalize((x - xm) / xs, dim=-1).to(device)
    del x

    H = 64
    print(f"[{time.time()-t0:.1f}s] Computing GCN embedding H={H}", flush=True)
    h2, deg = compute_embedding(xn, src.to(device), dst.to(device), n, H, device)
    logdeg = torch.log1p(deg).to(device)
    print(f"[{time.time()-t0:.1f}s] Embedding done. h2={tuple(h2.shape)} mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True)

    n_p = tr_pos.shape[1]
    n_n = tr_neg.shape[1]

    # Move train edge indices to GPU
    tp_u = tr_pos[0].to(device)
    tp_v = tr_pos[1].to(device)
    tn_u = tr_neg[0].to(device)
    tn_v = tr_neg[1].to(device)
    del tr_pos, tr_neg

    # Deterministic 2% holdout
    perm = torch.randperm(n_p, device="cpu")
    hold = int(0.02 * n_p)
    hold_idx = perm[:hold].to(device)
    keep_idx = perm[hold:].to(device)

    tp_u_k = tp_u[keep_idx]
    tp_v_k = tp_v[keep_idx]
    tp_u_h = tp_u[hold_idx]
    tp_v_h = tp_v[hold_idx]

    feat_dim = 4 * H + 2 + 2 * xn.shape[1]
    print(f"[{time.time()-t0:.1f}s] Train: {n_p} pos / {n_n} neg; holdout={hold}; feat_dim={feat_dim}", flush=True)

    head = torch.nn.Linear(feat_dim, 1, device=device)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-5)

    epochs = 25
    batch = 262144
    nk = tp_u_k.shape[0]

    for ep in range(epochs):
        perm_k = torch.randperm(nk, device="cpu").to(device)
        loss_sum, cnt = 0.0, 0
        for i in range(0, nk, batch):
            pi = perm_k[i:i + batch]
            ni = torch.randint(0, n_n, (pi.numel(),), device=device)
            # Build pair feats on-the-fly (stays on GPU, small per batch)
            fp = build_pair_feats(h2, xn, logdeg, tp_u_k[pi], tp_v_k[pi], H)
            fn = build_pair_feats(h2, xn, logdeg, tn_u[ni], tn_v[ni], H)
            logits = head(torch.cat([fp, fn], dim=0)).squeeze(-1)
            target = torch.cat([torch.ones_like(fp[:, 0]), torch.zeros_like(fn[:, 0])])
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * pi.numel()
            cnt += pi.numel()
            del fp, fn, logits, target, loss
        with torch.no_grad():
            s_p = head(build_pair_feats(h2, xn, logdeg, tp_u_h, tp_v_h, H)).squeeze(-1)
            s_n = head(build_pair_feats(h2, xn, logdeg, tn_u[:hold], tn_v[:hold], H)).squeeze(-1)
            auc = rank_auc(torch.cat([s_p, s_n]),
                           torch.cat([torch.ones(s_p.numel()), torch.zeros(s_n.numel())]).bool())
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
    cand = cand.long()

    def score_pairs(u, v):
        return head(build_pair_feats(h2, xn, logdeg, u.to(device), v.to(device), H)).squeeze(-1)

    chunk = 262144
    scores = []
    with torch.no_grad():
        for i in range(0, cand.shape[1], chunk):
            e = cand[:, i:i + chunk]
            s = score_pairs(e[0], e[1])
            scores.append(s.cpu())
    scores = torch.cat(scores)
    torch.save({"scores": scores}, Path(a.out))
    print(f"[{time.time()-t0:.1f}s] Wrote {scores.numel()} scores -> {a.out} (total mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
