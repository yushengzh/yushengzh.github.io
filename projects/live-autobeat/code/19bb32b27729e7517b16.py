#!/usr/bin/env python3
"""Link prediction: SIGN node features + pair heuristics + MLP, best-state 3-seed ensemble.

Node side: z-scored features, then decoupled propagation (SIGN / SGC):
[X, ÂX, Â²X] on the symmetric-normalised undirected graph with self loops,
encoded by a shared MLP.

Pair side: [h_u * h_v, |h_u - h_v|, log1p(CN), RA, AA, log dist(x_u, x_v),
cos(x_u, x_v)] -> MLP -> logit. The two raw-feature terms are computed before
standardisation: on vessel the endpoints of a true edge are spatially close, a
signal far smaller than the feature scale that an encoder washes out.
CN / RA / AA are the common-neighbour, resource-allocation and Adamic-Adar
counts (Liben-Nowell & Kleinberg 2007; Zhou et al. 2009).

Target-edge leakage. Training edges are in the message-passing graph but test
edges are not, so training uses a held-out 10% of training edges as positives
and removes them from the graph; inference uses the full graph (sacrosanct
asymmetry). Model selection: best-state by a fixed val subset scored on the
full view. Multiple seeds are trained and their score vectors averaged.

    python solve.py --data-dir DATA --out OUT.pt [--split val|test]
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
VAL_QUERIES = 2000          # grouped (MRR) validation subset
VAL_PAIRS = 400_000         # flat (AUC) validation subset
CHUNK = 200_000
VAL_SEL_SEED = 12345        # fixed, shared across seeds, for best-state selection


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# graph side
# --------------------------------------------------------------------------
def sym_adj(ei: np.ndarray, n: int) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float32)
    a.setdiag(0)
    return a


def sign_features(a: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """Decoupled SIGN / SGC features: [X, ÂX, Â²X] on sym-normalised adjacency."""
    n = a.shape[0]
    deg = np.asarray(a.sum(1)).ravel()
    deg[deg == 0] = 1.0
    d = (1.0 / np.sqrt(deg))
    An = (a.multiply(d[:, None]).multiply(d[None, :])).astype(np.float32)
    feats = [x.astype(np.float32)]
    h = x.astype(np.float32)
    for _ in range(HOPS):
        h = (An @ h).astype(np.float32)
        feats.append(h)
    return np.concatenate(feats, axis=1)


class Heuristics:
    """CN, RA (log1p) and AA (log1p) for endpoint pairs, computed on the view graph."""

    def __init__(self, a: sp.csr_matrix):
        self.deg = np.asarray(a.sum(1)).ravel()
        self.logdeg = np.log1p(np.maximum(self.deg, 1.0))
        self.A = a
        self.nn = 2 ** 8  # neighbour cap for AA/RA accumulation

    def _nbrs(self, u: int) -> np.ndarray:
        row = self.A.indices[self.A.indptr[u]:self.A.indptr[u + 1]]
        return row if row.size <= self.nn else row[:self.nn]

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.empty((len(u), 3), dtype=np.float32)
        for i in range(len(u)):
            nu, nv = self._nbrs(u[i]), self._nbrs(v[i])
            cn = np.intersect1d(nu, nv, assume_unique=True).size
            ra = 0.0 if cn == 0 else float(np.sum(1.0 / (self.deg[np.intersect1d(nu, nv, assume_unique=True)] + 1.0)))
            aa = 0.0 if cn == 0 else float(np.sum(1.0 / self.logdeg[np.intersect1d(nu, nv, assume_unique=True)]))
            out[i] = (cn, np.log1p(ra), np.log1p(aa))
        return out


def mlp(*dims: int) -> torch.nn.Sequential:
    layers, it = [], 0
    while it < len(dims) - 1:
        layers.append(torch.nn.Linear(dims[it], dims[it + 1]))
        if it < len(dims) - 2:
            layers.append(torch.nn.ReLU())
        it += 1
    return torch.nn.Sequential(*layers)


class LinkModel(torch.nn.Module):
    def __init__(self, f_dim: int):
        super().__init__()
        self.enc = mlp(f_dim, HIDDEN, HIDDEN)
        self.pred = mlp(2 * HIDDEN + 5, HIDDEN, 1)

    def forward(self, fu, fv, hz):
        hu, hv = self.enc(fu), self.enc(fv)
        return self.pred(torch.cat([hu * hv, (hu - hv).abs(), hz], 1)).view(-1)


class PairScorer:
    """Scores (u, v) pairs on one graph view: features + heuristics."""

    def __init__(self, feats: np.ndarray, raw: np.ndarray, heur: Heuristics, dev):
        self.f = torch.from_numpy(feats)
        self.raw = torch.from_numpy(raw)
        self.h = heur
        self.dev = dev

    def batch(self, u: np.ndarray, v: np.ndarray):
        fu = self.f[torch.from_numpy(u)].to(self.dev, non_blocking=True)
        fv = self.f[torch.from_numpy(v)].to(self.dev, non_blocking=True)
        ru, rv = self.raw[torch.from_numpy(u)], self.raw[torch.from_numpy(v)]
        dist = torch.log((ru - rv).norm(dim=1, keepdim=True) + 1e-6)
        cos = F.cosine_similarity(ru, rv, dim=1).unsqueeze(1)
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
    """Flat (AUC) validation subset: m pos + m neg. Grouped (MRR) branch for [V,K,2]."""
    pos, neg = d["val_edges"].numpy(), d["val_neg"].numpy()
    if neg.ndim == 3:
        m = min(VAL_QUERIES, pos.shape[1])
        sel = rng.choice(pos.shape[1], m, replace=False)
        U = np.concatenate([pos[0][sel], neg[:, sel].reshape(-1, 2)[0]])
        V = np.concatenate([pos[1][sel], neg[:, sel].reshape(-1, 2)[1]])
        Y = np.r_[np.ones(m), np.zeros(m * (neg.shape[1] - 1))].astype(np.float32)
        return U, V, Y, "mrr"
    m = min(VAL_PAIRS // 2, pos.shape[1], neg.shape[1])
    ps = rng.choice(pos.shape[1], m, replace=False)
    ns = rng.choice(neg.shape[1], m, replace=neg.shape[1] < m)
    U = np.concatenate([pos[0][ps], neg[0][ns]])
    V = np.concatenate([pos[1][ps], neg[1][ns]])
    Y = np.r_[np.ones(m), np.zeros(m)].astype(np.float32)
    return U, V, Y, "auc"


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def _run_one(d, n, ei, tr_pos, tr_neg, x, dev, seed, budget, full_view, vU, vV, vY):
    """Train one LinkModel; return the best-state model by fixed val AUC (full view)."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    n_t = min(int(TARGET_FRAC * tr_pos.shape[1]), MAX_TARGETS)
    tgt = rng.choice(tr_pos.shape[1], n_t, replace=False)
    keep = np.ones(tr_pos.shape[1], bool)
    keep[tgt] = False
    t_pos = tr_pos[:, tgt]
    key = lambda s, t: np.minimum(s, t) * n + np.maximum(s, t)
    ei_train = ei[:, ~np.isin(key(ei[0], ei[1]), key(t_pos[0], t_pos[1]))]
    a_tr = sym_adj(ei_train, n)
    train_view = PairScorer(sign_features(a_tr, x), d["x"].numpy().astype(np.float32), Heuristics(a_tr), dev)
    neg_idx = rng.choice(tr_neg.shape[1], n_t, replace=tr_neg.shape[1] < n_t)
    t_neg = tr_neg[:, neg_idx]
    U = np.concatenate([t_pos[0], t_neg[0]])
    V = np.concatenate([t_pos[1], t_neg[1]])
    Y = np.r_[np.ones(n_t, np.float32), np.zeros(n_t, np.float32)]
    torch.manual_seed(seed)
    model = LinkModel(train_view.f.size(1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, t_train = -1.0, None, time.time()
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
        log(f"  seed {seed} epoch {ep + 1}: loss={loss.item():.4f}")
        # best-state selection: score the fixed val subset on the FULL view
        vs = full_view.score(model, vU, vV).numpy()
        m = roc_auc(vY, vs)
        if m > best:
            best = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"    seed {seed} val AUC={m:.6f} (best so far)")
        if time.time() - t_train > budget or time.time() - t_start > 3300.0:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    log(f"  seed {seed} final best val AUC={best:.6f}")
    return model


N_SEEDS = int(os.environ.get("REF_N_SEEDS", 3))
TOTAL_BUDGET = float(os.environ.get("REF_TOTAL_BUDGET_SEC", 3300.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    d = torch.load(data_dir / "dataset.pt", weights_only=False)
    n = int(d["num_nodes"])
    raw = d["x"].numpy().astype(np.float32)
    x = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-6)
    ei = d["edge_index"].numpy().astype(np.int64)
    tr_pos = d["train_edges"].numpy()
    tr_neg = d["train_neg"].numpy()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_total = time.time()
    a_full = sym_adj(ei, n)
    full_view = PairScorer(sign_features(a_full, x), raw, Heuristics(a_full), dev)
    log(f"full view ready (nodes={n} edges={ei.shape[1]})")

    # fixed val subset for best-state selection, shared across seeds
    sel_rng = np.random.default_rng(VAL_SEL_SEED)
    vU, vV, vY, _kind = val_pairs(d, sel_rng)

    seeds = list(range(N_SEEDS))
    per_budget = min(TRAIN_BUDGET, max(60.0, (TOTAL_BUDGET - 300.0) / max(1, len(seeds))))
    log(f"training {len(seeds)} seeds, per-seed budget={per_budget:.0f}s total={TOTAL_BUDGET:.0f}s")

    models = []
    for seed in seeds:
        reserved = 60.0 * (len(models) + 1)
        remain = TOTAL_BUDGET - (time.time() - t_total)
        if remain - reserved < 120.0 and models:
            log(f"deadline guard: stopping after {len(models)} seed(s)")
            break
        budget = min(per_budget, max(120.0, remain - reserved - 60.0))
        t0 = time.time()
        model = _run_one(d, n, ei, tr_pos, tr_neg, x, dev, seed, budget, full_view, vU, vV, vY)
        models.append((seed, model))
        log(f"seed {seed} trained in {time.time() - t0:.0f}s")

    if a.split == "val":
        pos, neg = d["val_edges"].numpy(), d["val_neg"].numpy()
        if neg.ndim == 3:
            cand = np.concatenate([pos.T[:, None, :], neg], axis=1).reshape(-1, 2).T
        else:
            cand = np.concatenate([pos, neg], axis=1)
    else:
        cand = torch.load(data_dir / "test_inputs.pt", weights_only=False)["candidates"].numpy()
    cu, cv = cand[0].astype(np.int64), cand[1].astype(np.int64)

    acc = None
    for seed, model in models:
        sc = full_view.score(model, cu, cv)
        acc = sc if acc is None else acc + sc
        log(f"scored seed {seed} ({sc.numel()} pairs)")
    acc = acc / len(models)
    torch.save({"scores": acc}, Path(a.out))
    log(f"wrote {acc.numel()} {a.split} scores ({len(models)} seed(s)) -> {a.out}")
    log(f"total wall {time.time() - t_total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
