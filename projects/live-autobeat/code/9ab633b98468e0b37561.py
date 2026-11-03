#!/usr/bin/env python3
"""Link prediction on ogbl-vessel (v3).
Contract: python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
Output: {"scores": [C] float}, higher = more likely true edge. Metric: ROC-AUC.

Method: LogisticRegression on geometric + degree edge features, incl.
norm-normalized distance ratios (the dominant signal is Euclidean distance
relative to node position magnitude). No kNN (measured near-useless).
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

def build_features(x, deg, pairs):
    """x: [N,F] np float32, deg: [N] np int, pairs: [2,P] np int64.
    Returns [P, K] float32 feature matrix."""
    a = x[pairs[0]].astype(np.float64)
    b = x[pairs[1]].astype(np.float64)
    diff = a - b                                  # [P,F]
    dist = np.sqrt((diff * diff).sum(-1))         # [P]
    na = np.linalg.norm(a, axis=-1)               # [P]
    nb = np.linalg.norm(b, axis=-1)               # [P]
    cos = (a * b).sum(-1) / (na * nb + 1e-8)      # [P]
    dga = np.log1p(deg[pairs[0]].astype(np.float64))
    dgb = np.log1p(deg[pairs[1]].astype(np.float64))
    dgs = np.log1p(deg[pairs[0]].astype(np.float64) + deg[pairs[1]].astype(np.float64))
    F = np.column_stack([
        np.log1p(dist),                           # 0
        dist / 10.0,                              # 1
        cos,                                       # 2
        na / 1000.0,                              # 3  (scale ~0-5)
        nb / 1000.0,                              # 4
        (na - nb) / 1000.0,                       # 5
        np.abs(diff[:, 0]) / 1000.0,              # 6
        np.abs(diff[:, 1]) / 1000.0,              # 7
        np.abs(diff[:, 2]) / 1000.0,              # 8
        dga,                                       # 9
        dgb,                                       # 10
        dgs,                                       # 11
        dist / (na + 1e-8),                       # 12  norm-normalized
        dist / (nb + 1e-8),                       # 13
        dist / (na + nb + 1e-8),                  # 14
        dist,                                      # 15  raw
        dist * na / 1e6,                          # 16
        dist * nb / 1e6,                          # 17
        np.log1p(dist / (na + 1e-8)),             # 18
        np.log1p(dist / (nb + 1e-8)),             # 19
        deg[pairs[0]] - deg[pairs[1]],            # 20
        (deg[pairs[0]] * deg[pairs[1]]).astype(np.float64),  # 21
        diff[:, 0] / 1000.0,                      # 22  signed
        diff[:, 1] / 1000.0,                      # 23
        diff[:, 2] / 1000.0,                      # 24
    ])
    return F.astype(np.float32)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    data_dir = Path(a.data_dir)
    t0 = time.time()
    d = torch.load(data_dir / "dataset.pt")
    x = d["x"].numpy().astype(np.float32)
    row = d["edge_index"][0].long().numpy()
    N = int(d["num_nodes"])
    deg = np.bincount(row, minlength=N).astype(np.int64)
    print(f"t={time.time()-t0:.0f}s data loaded N={N} deg_max={deg.max()}", flush=True)

    tp = d["train_edges"].numpy()
    tn = d["train_neg"].numpy()
    Xtr = build_features(x, deg, np.concatenate([tp, tn], axis=1))
    y = np.concatenate([np.ones(tp.shape[1]), np.zeros(tn.shape[1])]).astype(np.int64)
    print(f"t={time.time()-t0:.0f}s train feats {Xtr.shape}", flush=True)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=0)
    clf.fit(Xtr, y)
    print(f"t={time.time()-t0:.0f}s fit done", flush=True)

    def score(pairs):
        return clf.predict_proba(build_features(x, deg, pairs))[:, 1]

    if a.split == "val":
        VP = np.concatenate([d["val_edges"].numpy(), d["val_neg"].numpy()], axis=1)
    else:
        VP = torch.load(data_dir / "test_inputs.pt")["candidates"].long().numpy()
    s = score(VP)
    torch.save({"scores": torch.from_numpy(s.astype(np.float32))}, Path(a.out))
    print(f"t={time.time()-t0:.0f}s wrote {a.split} scores={len(s)} -> {a.out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
