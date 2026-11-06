#!/usr/bin/env python3
"""Vessel link prediction — training-free geometric scorer.

Hypothesis: the node features `x` are 3-D coordinates and a pair is a true edge
when the two vessels are spatially close.  Distance is a strong, consistent,
zero-training signal (val AUC(-dist) ~= 0.918, identical on train).  Score each
candidate pair as -euclidean distance (monotone transform does not change
ROC-AUC).  No fitting -> nothing to overfit -> transfers to the sealed test set.

Output contract: write {"scores": tensor[C] float32} to --out, one value per
column of the candidate matrix, in the SAME order.  Saved as a torch tensor so
it loads under torch.load(weights_only=True).
"""
import argparse
import os
import time

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    return ap.parse_args()


def negative_dist(x, u, v, chunk=1_000_000):
    """Return [-||x[u]-x[v]||] for pair rows u,v, chunked to bound memory.
    Runs on CPU; x is a few hundred MB, C is ~1e6 -> finishes in seconds."""
    out = []
    n = u.numel()
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        xu = x[u[i:j]]
        xv = x[v[i:j]]
        d = torch.linalg.vector_norm(xu - xv, dim=-1)
        out.append(-d)
    return torch.cat(out, dim=0) if len(out) > 1 else out[0]


def main():
    args = parse_args()
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = torch.load(os.path.join(args.data_dir, 'dataset.pt'), map_location='cpu')
    x = d['x'].float().to(device)

    if args.split == 'val':
        pos = d['val_edges'].long()
        neg = d['val_neg']
        if neg.dim() == 3:
            # MRR layout [V, K, 2]: per positive i -> the positive, then its K negs
            V, K = neg.size(0), neg.size(1)
            cand = torch.empty(2, V * (K + 1), dtype=torch.long)
            cand[0, 0::K + 1] = pos[0]
            cand[1, 0::K + 1] = pos[1]
            cand[1, 1::K + 1] = neg[:, :, 1].reshape(-1)
            cand[0, 1::K + 1] = neg[:, :, 0].reshape(-1)
        else:
            # flat pool -> cat([val_edges, val_neg], dim=1)
            cand = torch.cat([pos, neg], dim=1)
        cand = cand.long().to(device)
    else:
        tp = os.path.join(args.data_dir, 'test_inputs.pt')
        if not os.path.exists(tp):
            raise FileNotFoundError(f"{tp} not found for --split test")
        td = torch.load(tp, map_location='cpu')
        cand = td['candidates'].long().to(device)

    print(f"candidates: {list(cand.shape)}")
    scores = negative_dist(x, cand[0], cand[1]).cpu().float()

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"scores": scores}, args.out)
    print(f"Wrote {args.out}: {scores.numel()} scores. total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
