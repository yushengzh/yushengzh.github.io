#!/usr/bin/env python3
"""Link prediction on ogbl-vessel. Distance-dominant signal + rich features.
Contract: python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
Output: {"scores": [C] float}, higher = more likely true edge.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch

def build_features(x, deg, pairs):
    # pairs: [2, P] long tensor (or np int64 array)
    x = x
    a = x[pairs[0]]
    b = x[pairs[1]]
    diff = a - b
    sq = (diff * diff).sum(-1)              # [P]
    dist = sq.sqrt()
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    cos = (a * b).sum(-1) / (na * nb + 1e-8)
    F = []
    F.append(dist.view(-1, 1).log1p())
    F.append(dist.view(-1, 1) / 10.0)
    F.append(cos.view(-1, 1))
    F.append(na.view(-1, 1))
    F.append(nb.view(-1, 1))
    F.append((na - nb).view(-1, 1))
    F.append(diff.abs())
    F.append(deg[pairs[0]].float().log1p().view(-1, 1))
    F.append(deg[pairs[1]].float().log1p().view(-1, 1))
    F.append((deg[pairs[0]].float() + deg[pairs[1]].float()).log1p().view(-1, 1))
    return torch.cat(F, dim=-1)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    torch.manual_seed(0)
    data_dir = Path(a.data_dir)
    t0 = time.time()
    d = torch.load(data_dir / "dataset.pt")
    x = d["x"].float()
    row, col = d["edge_index"]
    N = d["num_nodes"]
    deg = torch.zeros(N, dtype=torch.int64).index_add_(0, row.long(), torch.ones(row.numel(), dtype=torch.int64))

    tr_pos, tr_neg = d["train_edges"], d["train_neg"]
    npos = tr_pos.size(1); nneg = tr_neg.size(1)
    TP = torch.cat([tr_pos, tr_neg], 1)
    y = np.concatenate([np.ones(npos), np.zeros(nneg)]).astype(np.float32)
    print(f"train pos={npos} neg={nneg} x={tuple(x.shape)}", flush=True)

    # Build train features chunked (CPU)
    def feat(np_):
        return build_features(x, deg, np_)
    Xtr_np = feat(TP).numpy()  # [T, F]
    print("train feat shape", Xtr_np.shape, f"t={time.time()-t0:.0f}s", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
    clf.fit(Xtr_np, y)

    def score(pairs):
        return clf.predict_proba(feat(pairs).numpy())[:, 1]

    if a.split == "val":
        vpos, vneg = d["val_edges"], d["val_neg"]
        VP = torch.cat([vpos, vneg], 1)
        s = score(VP)
        torch.save({"scores": torch.from_numpy(s.astype(np.float32))}, Path(a.out))
    else:
        ti = torch.load(data_dir / "test_inputs.pt")["candidates"].long()
        s = score(ti)
        torch.save({"scores": torch.from_numpy(s.astype(np.float32))}, Path(a.out))
    print(f"wrote {a.split} scores={len(s)} -> {a.out} t={time.time()-t0:.0f}s", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
