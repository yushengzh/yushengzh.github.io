#!/usr/bin/env python3
"""v4: full_view features for training + scoring, target pairs, early stopping.

Key change from v1: use full_view (not train_view) for training node features.
This eliminates the train/test distribution mismatch in node embeddings.
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

T0 = time.time()
RUN_SEED = int(os.environ.get("REF_SEED", 0))
HOPS = 2
HIDDEN = int(os.environ.get("REF_HIDDEN", 256))
TARGET_FRAC = 0.1
MAX_TARGETS = int(os.environ.get("REF_MAX_TARGETS", 3_000_000))
TRAIN_BUDGET = float(os.environ.get("REF_TRAIN_BUDGET_SEC", 3000))
BATCH = int(os.environ.get("REF_BATCH", 8192))
VAL_PAIRS = int(os.environ.get("REF_VAL_PAIRS", 400_000))
CHUNK = 200_000
HZ_DIM = 5  # 3 heuristics + 2 spatial (logdist, cosine)

def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

def sym_adj(ei, n):
    a = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float32)
    a.setdiag(0); a.eliminate_zeros()
    return a.tocsr()

def sign_features(a, x):
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
    def __init__(self, a):
        self.a = a
        deg = np.asarray(a.sum(1)).ravel()
        self.a_ra = (a @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()
        self.a_aa = (a @ sp.diags(1.0 / np.log(np.maximum(deg, 2)))).tocsr()
    def __call__(self, u, v):
        au, av = self.a[u], self.a[v]
        cn = np.asarray(au.multiply(av).sum(1)).ravel()
        ra = np.asarray(au.multiply(self.a_ra[v]).sum(1)).ravel()
        aa = np.asarray(au.multiply(self.a_aa[v]).sum(1)).ravel()
        return np.stack([np.log1p(cn), ra, aa], axis=1).astype(np.float32)

def mlp(i, h, o, drop=0.1):
    return torch.nn.Sequential(
        torch.nn.Linear(i, h), torch.nn.LayerNorm(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
        torch.nn.Linear(h, h), torch.nn.LayerNorm(h), torch.nn.ReLU(), torch.nn.Dropout(drop),
        torch.nn.Linear(h, o))

class LinkModel(torch.nn.Module):
    def __init__(self, f_dim):
        super().__init__()
        self.enc = mlp(f_dim, HIDDEN, HIDDEN)
        self.pred = mlp(2 * HIDDEN + HZ_DIM, HIDDEN, 1)
    def forward(self, fu, fv, hz):
        hu, hv = self.enc(fu), self.enc(fv)
        return self.pred(torch.cat([hu * hv, (hu - hv).abs(), hz], 1)).view(-1)

class PairScorer:
    def __init__(self, feats, raw, heur, dev):
        self.f = torch.from_numpy(feats)
        self.raw = torch.from_numpy(raw)
        self.h = heur
        self.dev = dev
    def batch(self, u, v):
        fu = self.f[torch.from_numpy(u)].to(self.dev, non_blocking=True)
        fv = self.f[torch.from_numpy(v)].to(self.dev, non_blocking=True)
        ru, rv = self.raw[torch.from_numpy(u)], self.raw[torch.from_numpy(v)]
        dist = torch.log((ru - rv).norm(dim=1, keepdim=True) + 1e-6)
        cos = F.cosine_similarity(ru, rv, dim=1).unsqueeze(1)
        hz = torch.cat([torch.from_numpy(self.h(u, v)), dist, cos], 1).to(self.dev)
        return fu, fv, hz
    @torch.no_grad()
    def score(self, model, u, v):
        model.eval()
        out = []
        for i in range(0, len(u), CHUNK):
            fu, fv, hz = self.batch(u[i:i+CHUNK], v[i:i+CHUNK])
            out.append(torch.sigmoid(model(fu, fv, hz)).cpu())
        return torch.cat(out)

def val_pairs(d, rng):
    m = VAL_PAIRS // 2
    pos, neg = d["val_edges"].numpy(), d["val_neg"].numpy()
    if neg.ndim == 3:
        # grouped: use first group's negatives as pool
        neg = neg.reshape(-1, 2)
    ip = rng.choice(pos.shape[1], m, replace=False)
    ineg = rng.choice(neg.shape[1], m, replace=False)
    u = np.concatenate([pos[0, ip], neg[0, ineg]])
    v = np.concatenate([pos[1, ip], neg[1, ineg]])
    y = np.r_[np.ones(m), np.zeros(m)]
    return u, v, lambda s: float(roc_auc_score(y, s.numpy()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    d = torch.load(Path(a.data_dir) / "dataset.pt", weights_only=False)
    n = int(d["num_nodes"])
    raw = d["x"].numpy().astype(np.float32)
    x = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-6)
    ei = d["edge_index"].numpy().astype(np.int64)
    tr_pos = d["train_edges"].numpy()
    tr_neg = d["train_neg"].numpy()
    rng = np.random.default_rng(RUN_SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"N={n} F={raw.shape[1]} E={ei.shape[1]}")

    # Build full graph view (used for BOTH training and scoring)
    a_full = sym_adj(ei, n)
    full_view = PairScorer(sign_features(a_full, x), raw, Heuristics(a_full), dev)
    log(f"full view built {time.time()-T0:.1f}s")

    # Select target edges (held out for training supervision)
    n_t = min(int(TARGET_FRAC * tr_pos.shape[1]), MAX_TARGETS)
    tgt = rng.choice(tr_pos.shape[1], n_t, replace=False)
    t_pos = tr_pos[:, tgt]
    neg_idx = rng.choice(tr_neg.shape[1], n_t, replace=tr_neg.shape[1] < n_t)
    t_neg = tr_neg[:, neg_idx]

    U = np.concatenate([t_pos[0], t_neg[0]])
    V = np.concatenate([t_pos[1], t_neg[1]])
    Y = np.r_[np.ones(n_t, np.float32), np.zeros(n_t, np.float32)]
    log(f"train pairs: pos={n_t} neg={n_t}")

    # Val slice for early stopping
    vu, vv, vmetric = val_pairs(d, rng)
    log(f"val slice: {len(vu)} pairs")

    torch.manual_seed(RUN_SEED)
    model = LinkModel(full_view.f.size(1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state = None, None
    t_train = time.time()
    patience = 3
    bad_epochs = 0
    for ep in range(200):
        model.train()
        perm = rng.permutation(len(U))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i+BATCH]
            out = model(*full_view.batch(U[b], V[b]))
            loss = F.binary_cross_entropy_with_logits(out, torch.from_numpy(Y[b]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sc = vmetric(full_view.score(model, vu, vv))
        log(f"epoch {ep+1}: loss={loss.item():.4f} val={sc:.4f}")
        if best is None or sc > best:
            best = sc
            bad_epochs = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            log("early stop")
            break
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
        cand = torch.load(Path(a.data_dir) / "test_inputs.pt", weights_only=False)["candidates"].numpy()
    scores = full_view.score(model, cand[0].astype(np.int64), cand[1].astype(np.int64))
    torch.save({"scores": scores}, a.out)
    log(f"wrote {scores.numel()} {a.split} scores -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
