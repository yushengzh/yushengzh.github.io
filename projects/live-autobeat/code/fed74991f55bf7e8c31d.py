import os
import argparse
import torch
import torch.nn.functional as F


def score_pairs(z, a, b, chunk=2_000_000):
    """Dot product of normalized embeddings for pairs (a[i], b[i])."""
    n = a.numel()
    out = torch.empty(n, dtype=torch.float32, device=z.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = (z[a[s:e]] * z[b[s:e]]).sum(dim=-1)
    return out


def build_val_candidates(val_edges, val_neg):
    """Grouped MRR layout: per positive i, the positive pair then its K negatives.
    Vectorized, no per-i python loop."""
    V, K, _ = val_neg.shape
    K2 = K * V
    # repeat each positive pair K+1 times -> block of K+1
    rep = torch.repeat_interleave(torch.arange(V), K + 1)  # [V*(K+1)]
    cand = torch.empty(2, V * (K + 1), dtype=val_neg.dtype)
    # positives at i*(K+1)
    pos_idx = torch.arange(V) * (K + 1)
    cand[0, pos_idx] = val_edges[0]
    cand[1, pos_idx] = val_edges[1]
    # negatives: flatten to [V*K,2]; slot j (0-based) goes into position j//K*(K+1) + 1 + j%K
    flat = val_neg.reshape(-1, 2)  # [V*K,2]
    j = torch.arange(flat.size(0))
    slots = (j // K) * (K + 1) + 1 + (j % K)
    cand[0, slots] = flat[:, 0]
    cand[1, slots] = flat[:, 1]
    return cand


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

    if args.split == "test":
        tp = os.path.join(args.data_dir, "test_inputs.pt")
        td = torch.load(tp, map_location="cpu")
        cand = td["candidates"].long()
    else:
        val_edges = d["val_edges"].long()
        val_neg = d["val_neg"].long()
        if val_neg.dim() == 3:
            cand = build_val_candidates(val_edges, val_neg)
        else:
            cand = torch.cat([val_edges, val_neg], dim=1)
        print(f"[solve] val candidates C={cand.shape[1]}", flush=True)

    a = cand[0].cpu()
    b = cand[1].cpu()
    with torch.no_grad():
        scores = score_pairs(z, a, b, chunk=2_000_000)
    torch.save({"scores": scores.float()}, args.out)
    print(f"[solve] wrote {scores.shape[0]} scores to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
