#!/usr/bin/env python3
"""Link prediction: SIGN node features + pair heuristics + MLP.

v2: adds degree features (log1p deg_u, log1p deg_v, log1p |deg_u - deg_v|)
to the pair-side heuristic vector. hz dim: 5 -> 8.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

T0 = time.time()
RUN_SEED = int(os.environ.get("REF_SEED", 0))
HOPS = 2
HIDDEN = 256
TARGET_FRAC = 0.1
MAX_TARGETS = int(os.environ.get("REF_MAX_TARGETS", 3_000_000))
EPOCHS = int(os.environ.get("REF_EPOCHS", 30))
TRAIN_BUDGET = float(os.environ.get("REF_TRAIN_BUDGET_SEC", 1200))
BATCH = 8192
VAL_QUERIES = 2000
VAL_PAIRS = 400_000
CHUNK = 200_000

# hz dimension: 3 (heuristics) + 2 (spatial) + 3 (degree) = 8
HZ_DIM = 8


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sym_adj(ei: np.ndarray, n: int) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float32)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def sign_features(a: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    a_hat = a + sp.eye(n, dtype=np.float32, format="csr")
    dinv = 1.0 / np.sqrt(np.asarray(a_hat.sum(1)).ravel())
    a_hat = sp.diags(dinv) @ a_hat @ sp.diags(dinv)
    feats, h = [x], x
    for _ in range(HOPS):
        h = a_hat @ h
        feats.append(h)
    return np.concatenate(feats, axis=1).astype(np.float32)


class Heuristics:
    def __init__(self, a: sp.csr_matrix):
        deg = np.asarray(a.sum(1)).ravel()
        self.deg = deg  # store for pair-side degree features
        self.a = a
        self.a_ra = (a @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()
        self.a_aa = (a @ sp.diags(1.0 / np.log(np.maximum(deg, 2)))).tocsr()

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        au, av = self.a[u], self.a[v]
        cn = np.asarray(au.multiply(av).sum(1)).ravel()
        ra = np.asarray(au.multiply(self.a_ra[v]).sum(1)).ravel()
        aa = np.asarray(au.multiply(self.a_aa[v]).sum(1)).ravel()
        deg_u = np.log1p(self.deg[u])
        deg_v = np.log1p(self.deg[v])
        deg_diff = np.log1p(np.abs(self.deg[u] - self.deg[v]))
        return np.stack([np.log1p(cn), ra, aa, deg_u, deg_v, deg_diff], axis=1).astype(np.float32)


def mlp(i, h, o, drop=0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(i, h), torch.nn.LayerNorm(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
        torch.nn.Linear(h, h), torch.nn.LayerNorm(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
        torch.nn.Linear(h, o))


class LinkModel(torch.nn.Module):
    def __init__(self, f_dim: int):
        super().__init__()
        self.enc = mlp(f_dim, HIDDEN, HIDDEN)
        # hz is now 6-dim: [log1p(cn), ra, aa, log1p(deg_u), log1p(deg_v), log1p(|deg_u-deg_v|)]
        self.pred = mlp(2 * HIDDEN + HZ_DIM, HIDDEN, 1)

    def forward(self, fu, fv, hz):
        hu, hv = self.enc(fu), self.enc(fv)
        return self.pred(torch.cat([hu * hv, (hu - hv).abs(), hz], 1)).view(-1)


class PairScorer:
    """Scores (u, v) pairs on one graph view: SIGN features + heuristics (incl. degree)."""

    def __init__(self, feats: np.ndarray, raw: np.ndarray, heur: Heuristics, dev):
        self.f = torch.from_numpy(feats)
        self.raw = torch.from_numpy(raw)
        self.h = heur
        self.dev = dev

    def batch(self, u: np.ndarray, v: np.ndarray):
        fu = self.f[torch.from_numpy(u)].to(self.dev, non_blocking=True)
        fv = self.f[torch.from_numpy(v)].to(self.dev, non_blocking=True)
        ru, rv = self.raw[torch.from_numpy(u)], self.raw[torch.from_numpy(v)]
        # spatial: log dist and cosine (F-agnostic)
        dist = torch.log((ru - rv).norm(dim=1, keepdim=True) + 1e-6)
        cos = F.cosine_similarity(ru, rv, dim=1).unsqueeze(1)
        # heuristics already include degree features (6-dim)
        hz = torch.cat([torch.from_numpy(self.h(u, v)), dist, cos], 1).to(self.dev)
        return fu, fv, hz

    @torch.no_grad()
    def score(self, model, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
        model.eval()
        out = []
        for i in range(0, len(u), CHUNK):
            out.append(model(*self.batch(u[i:i + CHUNK], v[i:i + CHUNK])).float().cpu())
        return torch.cat(out) if out else torch.zeros(0)


def val_pairs(d, rng):
    """A fixed validation subset and a scorer for it (MRR or ROC-AUC)."""
    pos, neg = d["val_edges"].numpy(), d["val_neg"].numpy()
    if neg.ndim == 3:  # grouped: MRR
        q = rng.choice(pos.shape[1], min(VAL_QUERIES, pos.shape[1]), replace=False)
        k = neg.shape[1]
        cand = np.concatenate([pos[:, q].T[:, None, :], neg[q]], axis=1).reshape(-1, 2)

        def metric(s):
            s = s.view(len(q), k + 1)
            rank = (s[:, 1:] >= s[:, :1]).sum(1) + 1
            return (1.0 / rank.float()).mean().item()
        return cand[:, 0], cand[:, 1], metric
    from sklearn.metrics import roc_auc_score
    m = min(VAL_PAIRS // 2, pos.shape[1], neg.shape[1])
    ip = rng.choice(pos.shape[1], m, replace=False)
    ineg = rng.choice(neg.shape[1], m, replace=False)
    u = np.concatenate([pos[0, ip], neg[0, ineg]])
    v = np.concatenate([pos[1, ip], neg[1, ineg]])
    y = np.r_[np.ones(m), np.zeros(m)]
    return u, v, lambda s: float(roc_auc_score(y, s.numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    a = ap.parse_args()

    d = torch.load(a.data_dir / "dataset.pt", weights_only=False)
    x = d["x"].numpy().astype(np.float32)
    ei = d["edge_index"].numpy()
    n = d["num_nodes"]
    log(f"N={n} F={x.shape[1]} E={ei.shape[1]}")

    mean = x.mean(0, keepdims=True)
    std = np.maximum(x.std(0, keepdims=True), 1e-6)
    x = ((x - mean) / std).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(RUN_SEED)

    # --- target edges: 10% of train pos held out, removed from graph ---
    tr_pos = d["train_edges"].numpy()
    n_t = min(int(TARGET_FRAC * tr_pos.shape[1]), MAX_TARGETS)
    tgt = rng.choice(tr_pos.shape[1], n_t, replace=False)
    tgt_edges = tr_pos[:, tgt]
    keys = np.minimum(tgt_edges[0], tgt_edges[1]).astype(np.int64) * n + np.maximum(tgt_edges[0], tgt_edges[1]).astype(np.int64)
    ei_keys = np.minimum(ei[0], ei[1]).astype(np.int64) * n + np.maximum(ei[0], ei[1]).astype(np.int64)
    mask = ~np.isin(ei_keys, keys)
    ei_train = ei[:, mask]
    log(f"targets={n_t}  ei_train={ei_train.shape[1]}")

    a_full = sym_adj(ei, n)
    a_train = sym_adj(ei_train, n)

    # --- build graph views ---
    t0 = time.time()
    feats_full = sign_features(a_full, x)
    heur_full = Heuristics(a_full)
    full_view = PairScorer(feats_full, x, heur_full, dev)
    log(f"full view built {time.time() - t0:.1f}s")

    t0 = time.time()
    feats_train = sign_features(a_train, x)
    heur_train = Heuristics(a_train)
    train_view = PairScorer(feats_train, x, heur_train, dev)
    log(f"train view built {time.time() - t0:.1f}")

    # --- training pairs ---
    neg_pool = d["train_neg"].numpy()
    n_tr = tr_pos.shape[1] - n_t  # non-target train pos
    pos_idx = np.concatenate([np.arange(tr_pos.shape[1]), [tr_pos.shape[1]]])[:0]  # placeholder
    # train pos = all tr_pos except tgt (the non-target subset)
    keep = np.ones(tr_pos.shape[1], dtype=bool)
    keep[tgt] = False
    tr_pos_kept = tr_pos[:, keep]
    neg_idx = rng.choice(neg_pool.shape[1], n_tr, replace=False)
    tr_neg_kept = neg_pool[:, neg_idx]
    U = np.concatenate([tr_pos_kept[0], tgt_edges[0]])
    V = np.concatenate([tr_pos_kept[1], tgt_edges[1]])
    Y = np.concatenate([np.ones(n_tr), np.ones(n_t)])
    log(f"train pairs: pos={n_tr}+{n_t} neg={n_tr}")

    # Actually: positives = tr_pos_kept (non-target) + tgt_edges (held-out)
    # Negatives = sampled from neg_pool
    # Total training positives: n_tr + n_t = full tr_pos count
    # But we only have n_tr negatives. Let's balance:
    # Use all tr_pos_kept as pos with n_tr negatives, and tgt_edges as extra pos
    # Simpler: train on tr_pos_kept (pos) vs tr_neg_kept (neg), plus tgt (pos) vs sampled neg
    # For simplicity keep original structure: U = all tr_pos, V = all tr_pos[1], negatives balanced
    # Re-reading reference: U = cat([tr_pos_kept[0], tgt[0]]), V = cat([tr_pos_kept[1], tgt[1]])
    # Y = cat([ones(n_tr), zeros(n_t)])  -- NO, tgt are positives too
    # Actually in the reference: train on tr_pos (all) as pos, tr_neg as neg.
    # The tgt are a SUBSET of tr_pos used for early-stopping validation.
    # Let me re-read... The reference uses train_view for training on U,V,Y.
    # U = cat([non-tgt pos, tgt pos]), Y = cat([ones(n_tr), zeros(n_t)])
    # That means tgt edges are used as NEGATIVES in training?? No...
    # Actually looking at the code again: the tgt are the HELD OUT positives
    # for validation. The training set is the remaining tr_pos (pos) + tr_neg (neg).
    # Let me just follow the reference exactly.

    # Following reference structure:
    # tr_pos_kept: non-target positives (used for training)
    # tgt_edges: held-out targets (removed from graph, used as val slice)
    # Training: pos = tr_pos_kept, neg = sampled from neg_pool
    n_train_pos = tr_pos_kept.shape[1]
    n_train_neg = n_train_pos
    neg_idx = rng.choice(neg_pool.shape[1], n_train_neg, replace=False)
    U = np.concatenate([tr_pos_kept[0], neg_pool[0, neg_idx]])
    V = np.concatenate([tr_pos_kept[1], neg_pool[1, neg_idx]])
    Y = np.concatenate([np.ones(n_train_pos), np.zeros(n_train_neg)])
    log(f"train: pos={n_train_pos} neg={n_train_neg}")

    # val slice: 400k pairs from the visible val set
    vu, vv, vmetric = val_pairs(d, rng)
    log(f"val slice: {len(vu)} pairs")

    # --- train ---
    torch.manual_seed(RUN_SEED)
    model = LinkModel(f_dim=feats_full.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, t_train = None, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(U))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            out = model(*train_view.batch(U[b], V[b]))
            loss = F.binary_cross_entropy_with_logits(out, torch.from_numpy(Y[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sc = vmetric(full_view.score(model, vu, vv))
        log(f"epoch {ep + 1}: loss={loss.item():.4f} val={sc:.4f}")
        if best is None or sc > best:
            best = sc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if time.time() - t_train > TRAIN_BUDGET:
            log("training budget reached")
            break
    model.load_state_dict(best_state)
    log(f"best val {best:.4f}")

    if a.split == "val":
        pos, neg = d["val_edges"].numpy(), d["val_neg"].numpy()
        if neg.ndim == 3:
            cand = np.concatenate([pos.T[:, None, :], neg], axis=1).reshape(-1, 2).T
        else:
            cand = np.concatenate([pos, neg], axis=1)
    else:
        cand = torch.load(a.data_dir / "test_inputs.pt", weights_only=False)["candidates"].numpy()
    scores = full_view.score(model, cand[0].astype(np.int64), cand[1].astype(np.int64))
    torch.save({"scores": scores}, a.out)
    log(f"wrote {scores.numel()} {a.split} scores -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
