#!/usr/bin/env python3
"""Reference solution, node classification: SIGN features + label feature + MLP.

Decoupled propagation (SIGN, Frasca et al. 2020; SGC): precompute A^k X for
k = 0..3 on the symmetric-normalised undirected graph, then train an MLP on the
concatenation. Add one label feature: the mean one-hot training label over each
node's neighbours (self excluded, so a training node never sees its own label).

Runs on any task with x / edge_index / train_idx / val_idx / y. Selection on the
visible validation split only.

    python solve.py --data-dir DATA --out OUT.pt [--split val|test]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

T0 = time.time()
# Run seed. 0 is the default; the anchor is the mean over REF_SEED = 0, 1, 2.
RUN_SEED = int(os.environ.get("REF_SEED", 0))
HOPS = 3
HIDDEN = 512
EPOCHS = int(os.environ.get("REF_EPOCHS", 300))
SEEDS = int(os.environ.get("REF_SEEDS", 5))


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def adjacency(edge_index: torch.Tensor, n: int, self_loops: bool, norm: str):
    ei = edge_index.long()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    if self_loops:
        ar = torch.arange(n)
        ei = torch.cat([ei, torch.stack([ar, ar])], dim=1)
    a = torch.sparse_coo_tensor(ei, torch.ones(ei.size(1)), (n, n)).coalesce()
    idx = a.indices()
    v = torch.ones(idx.size(1))                       # dedupe multi-edges
    deg = torch.zeros(n).index_add_(0, idx[0], v).clamp(min=1)
    if norm == "sym":
        v = deg[idx[0]].rsqrt() * deg[idx[1]].rsqrt()
    else:                                             # row: neighbour mean
        v = 1.0 / deg[idx[0]]
    return torch.sparse_coo_tensor(idx, v, (n, n)).coalesce().to_sparse_csr()


class MLP(torch.nn.Module):
    def __init__(self, i, h, o, drop=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(i, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
            torch.nn.Linear(h, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
            torch.nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    d = torch.load(Path(a.data_dir) / "dataset.pt", weights_only=False)
    x, y = d["x"].float(), d["y"].long()
    n, c = x.size(0), int(d["num_classes"])
    tr, va = d["train_idx"], d["val_idx"]
    target = va if a.split == "val" else d["test_idx"]

    a_sym = adjacency(d["edge_index"], n, True, "sym")
    feats, h = [x], x
    for _ in range(HOPS):
        h = a_sym @ h
        feats.append(h)
    onehot = torch.zeros(n, c)
    onehot[tr, y[tr]] = 1.0
    feats.append(adjacency(d["edge_index"], n, False, "row") @ onehot)
    X = torch.cat(feats, dim=1)
    X = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
    log(f"features {tuple(X.shape)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xd = X.to(dev)
    ytr = y[tr].to(dev)
    probs = torch.zeros(target.numel(), c)
    for seed in range(RUN_SEED * SEEDS, (RUN_SEED + 1) * SEEDS):
        torch.manual_seed(seed)
        model = MLP(X.size(1), HIDDEN, c).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_p = -1.0, None
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(tr.numel(), device=dev)
            for i in range(0, perm.numel(), 50_000):
                b = perm[i:i + 50_000]
                opt.zero_grad()
                F.cross_entropy(model(Xd[tr.to(dev)[b]]), ytr[b]).backward()
                opt.step()
            if ep % 5 == 4:
                model.eval()
                with torch.no_grad():
                    acc = (model(Xd[va.to(dev)]).argmax(1).cpu() == y[va]).float().mean().item()
                    if acc > best:
                        best = acc
                        best_p = model(Xd[target.to(dev)]).softmax(1).cpu()
        log(f"seed {seed}: best val acc {best:.4f}")
        probs += best_p
    torch.save({"pred": probs.argmax(1)}, Path(a.out))
    log(f"wrote {target.numel()} {a.split} predictions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
