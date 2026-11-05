import os
import argparse
import torch
import torch.nn.functional as F


def score_pairs(z, a, b, deg, w, chunk=2_000_000):
    """Score = cos + w*log1p(deg_a*deg_b+1) for pairs (a[i],b[i])."""
    n = a.numel()
    out = torch.empty(n, dtype=torch.float32, device=z.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        cos = (z[a[s:e]] * z[b[s:e]]).sum(dim=-1)
        dp = (deg[a[s:e]] * deg[b[s:e]] + 1.0).log1p()
        out[s:e] = cos + w * dp
    return out


def build_val_candidates(val_edges, val_neg):
    """Grouped MRR layout: per positive i, the positive pair then its K negatives."""
    V, K, _ = val_neg.shape
    cand = torch.empty(2, V * (K + 1), dtype=val_neg.dtype)
    pos_idx = torch.arange(V) * (K + 1)
    cand[0, pos_idx] = val_edges[0]
    cand[1, pos_idx] = val_edges[1]
    flat = val_neg.reshape(-1, 2)
    j = torch.arange(flat.size(0))
    slots = (j // K) * (K + 1) + 1 + (j % K)
    cand[0, slots] = flat[:, 0]
    cand[1, slots] = flat[:, 1]
    return cand


def tune_w(val_edges, val_neg, z, deg, w_candidates, n_tune=5000):
    """Pick best w from a small held-out slice of val positives."""
    V = val_edges.shape[1]
    idx = torch.randperm(V)[:min(n_tune, V)]
    sub_pos = val_edges[:, idx]
    sub_neg = val_neg[idx]  # [n_tune, K, 2]
    from ogb.linkproppred import Evaluator
    import numpy as np
    ev = Evaluator(name="ogbl-citation2")
    K = sub_neg.shape[1]
    best_w, best_mrr = 0.0, -1.0
    for w in w_candidates:
        cand = build_val_candidates(sub_pos, sub_neg)
        a, b = cand[0], cand[1]
        scores = score_pairs(z, a, b, deg, w).cpu()
        mat = scores.view(-1, K + 1)
        r = ev.eval({"y_pred_pos": mat[:, 0], "y_pred_neg": mat[:, 1:]})
        mrr = float(np.mean(np.asarray(r["mrr_list"])))
        print(f"  w={w:.4f} -> MRR={mrr:.6f}", flush=True)
        if mrr > best_mrr:
            best_mrr, best_w = mrr, w
    print(f"  Best w={best_w:.4f} MRR={best_mrr:.6f}", flush=True)
    return best_w, best_mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    torch.manual_seed(0)

    d = torch.load(os.path.join(args.data_dir, "dataset.pt"), map_location="cpu")
    x = d["x"].float().contiguous()
    print(f"[solve] x={tuple(x.shape)}", flush=True)
    z = F.normalize(x, dim=-1)

    # Compute undirected degree
    e = d["edge_index"].long()
    deg = torch.zeros(d["num_nodes"], dtype=torch.float32)
    deg.index_add_(0, e[0], torch.ones_like(e[0], dtype=torch.float32))
    deg.index_add_(0, e[1], torch.ones_like(e[1], dtype=torch.float32))
    print(f"[solve] degree: mean={deg.mean():.2f} max={deg.max():.0f}", flush=True)

    if args.split == "val":
        val_edges = d["val_edges"].long()
        val_neg = d["val_neg"].long()
        w_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        print("[solve] tuning w on 5000 val positives...", flush=True)
        best_w, best_mrr = tune_w(val_edges, val_neg, z, deg, w_candidates)
        cand = build_val_candidates(val_edges, val_neg)
        print(f"[solve] val candidates C={cand.shape[1]} using w={best_w:.4f}", flush=True)
        a, b = cand[0], cand[1]
        with torch.no_grad():
            scores = score_pairs(z, a, b, deg, best_w)
    else:
        # Use a fixed w (tuned offline on val)
        w = 0.01  # tuned offline on val (full-val MRR 0.3077 vs v3 0.2550)
        tp = os.path.join(args.data_dir, "test_inputs.pt")
        td = torch.load(tp, map_location="cpu")
        cand = td["candidates"].long()
        print(f"[solve] test candidates C={cand.shape[1]} using w={w:.4f}", flush=True)
        a, b = cand[0], cand[1]
        with torch.no_grad():
            scores = score_pairs(z, a, b, deg, w)

    torch.save({"scores": scores.float()}, args.out)
    print(f"[solve] wrote {scores.shape[0]} scores to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
