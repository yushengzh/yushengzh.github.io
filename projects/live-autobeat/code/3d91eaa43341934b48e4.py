#!/usr/bin/env python3
"""Reference solution, link prediction: SIGN node features + pair heuristics + MLP.

Node side: z-scored features, then decoupled propagation (SIGN / SGC):
[X, ÂX, Â²X] on the symmetric-normalised undirected graph with self loops,
encoded by a shared MLP.

Pair side: [h_u * h_v, |h_u - h_v|, log1p(CN), RA, AA, log dist(x_u, x_v),
cos(x_u, x_v)] -> MLP -> logit. The two raw-feature terms are computed before
standardisation: on vessel the endpoints of a true edge are spatially close, a
signal far smaller than the feature scale that an encoder washes out.
CN / RA / AA are the common-neighbour, resource-allocation and Adamic-Adar
counts (Liben-Nowell & Kleinberg 2007; Zhou et al. 2009). The difference term
is the vessel baseline's DiffPredictor; the heuristics carry citation2.

Target-edge leakage. Test edges are not in the message-passing graph, but
training edges are. So training uses a held-out 10% of the training edges as
positives and removes them from the graph the features and heuristics are
computed on; inference uses the full graph. Selection on a validation subset.

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
# Run seed. 0 is the default; the anchor is the mean over REF_SEED = 0, 1, 2.
RUN_SEED = int(os.environ.get("REF_SEED", 0))
HOPS = 3
HIDDEN = 256
TARGET_FRAC = 0.1
MAX_TARGETS = int(os.environ.get("REF_MAX_TARGETS", 3_000_000))
EPOCHS = int(os.environ.get("REF_EPOCHS", 30))
TRAIN_BUDGET = float(os.environ.get("REF_TRAIN_BUDGET_SEC", 1200))
BATCH = 8192
VAL_QUERIES = 2000          # grouped (MRR) validation subset
VAL_PAIRS = 400_000         # flat (AUC) validation subset
CHUNK = 200_000


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# graph side
# --------------------------------------------------------------------------
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
        self.a = a
        self.a_ra = (a @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()
        self.a_aa = (a @ sp.diags(1.0 / np.log(np.maximum(deg, 2)))).tocsr()

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        au, av = self.a[u], self.a[v]
        cn = np.asarray(au.multiply(av).sum(1)).ravel()
        ra = np.asarray(au.multiply(self.a_ra[v]).sum(1)).ravel()
        aa = np.asarray(au.multiply(self.a_aa[v]).sum(1)).ravel()
        return np.stack([np.log1p(cn), ra, aa], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def mlp(i, h, o, drop=0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(i, h), torch.nn.LayerNorm(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
        torch.nn.Linear(h, h), torch.nn.LayerNorm(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
        torch.nn.Linear(h, o))


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
    def _score_chunk(self, model, u: np.ndarray, v: np.ndarray, c: int) -> torch.Tensor:
        if c <= 0 or len(u) == 0:
            return torch.zeros(0)
        try:
            return model(*self.batch(u, v)).float().cpu()
        except torch.OutOfMemoryError:
            if c <= 8192:
                torch.cuda.empty_cache()
                raise
            torch.cuda.empty_cache()
            h = c // 2
            return torch.cat([self._score_chunk(model, u[:h], v[:h], h),
                              self._score_chunk(model, u[h:], v[h:], c - h)])

    @torch.no_grad()
    def score(self, model, u: np.ndarray, v: np.ndarray) -> torch.Tensor:
        model.eval()
        out = []
        for i in range(0, len(u), CHUNK):
            out.append(self._score_chunk(model, u[i:i + CHUNK], v[i:i + CHUNK], CHUNK))
        return torch.cat(out) if out else torch.zeros(0)


def val_pairs(d, rng):
    """A fixed validation subset and a scorer for it (MRR or ROC-AUC)."""
    pos, neg = d["val_edges"].numpy(), d["val_neg"].numpy()
    if neg.ndim == 3:                                    # grouped: MRR
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    data_dir = Path(a.data_dir)
    d = torch.load(data_dir / "dataset.pt", weights_only=False)
    n = int(d["num_nodes"])
    raw = d["x"].numpy().astype(np.float32)
    x = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-6)
    ei = d["edge_index"].numpy().astype(np.int64)
    tr_pos = d["train_edges"].numpy()
    tr_neg = d["train_neg"].numpy()
    rng = np.random.default_rng(RUN_SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # held-out target edges, removed from the training view of the graph
    n_t = min(int(TARGET_FRAC * tr_pos.shape[1]), MAX_TARGETS)
    tgt = rng.choice(tr_pos.shape[1], n_t, replace=False)
    keep = np.ones(tr_pos.shape[1], bool)
    keep[tgt] = False
    t_pos = tr_pos[:, tgt]
    # drop the targets (either direction) from the message-passing edges
    key = lambda s, t: np.minimum(s, t) * n + np.maximum(s, t)
    ei_train = ei[:, ~np.isin(key(ei[0], ei[1]), key(t_pos[0], t_pos[1]))]
    log(f"nodes={n} edges={ei.shape[1]} targets={n_t} train-view edges={ei_train.shape[1]}")

    a_tr = sym_adj(ei_train, n)
    train_view = PairScorer(sign_features(a_tr, x), raw, Heuristics(a_tr), dev)
    log("training view ready")

    neg_idx = rng.choice(tr_neg.shape[1], n_t, replace=tr_neg.shape[1] < n_t)
    t_neg = tr_neg[:, neg_idx]
    U = np.concatenate([t_pos[0], t_neg[0]])
    V = np.concatenate([t_pos[1], t_neg[1]])
    Y = np.r_[np.ones(n_t, np.float32), np.zeros(n_t, np.float32)]

    # full view: used for validation and for the final predictions
    a_full = sym_adj(ei, n)
    full_view = PairScorer(sign_features(a_full, x), raw, Heuristics(a_full), dev)
    del a_tr
    vu, vv, vmetric = val_pairs(d, rng)
    log("full view ready")

    torch.manual_seed(RUN_SEED)
    model = LinkModel(train_view.f.size(1)).to(dev)
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
        if neg.ndim == 3:   # per positive: the positive, then its K negatives
            cand = np.concatenate([pos.T[:, None, :], neg], axis=1).reshape(-1, 2).T
        else:
            cand = np.concatenate([pos, neg], axis=1)
    else:
        cand = torch.load(data_dir / "test_inputs.pt", weights_only=False)["candidates"].numpy()
    scores = full_view.score(model, cand[0].astype(np.int64), cand[1].astype(np.int64))
    torch.save({"scores": scores}, Path(a.out))
    log(f"wrote {scores.numel()} {a.split} scores -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
