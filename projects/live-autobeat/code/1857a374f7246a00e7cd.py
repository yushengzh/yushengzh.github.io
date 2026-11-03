from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    a = ap.parse_args()
    t0 = time.time()
    d = torch.load(Path(a.data_dir)/"dataset.pt")
    x = d["x"].numpy().astype(np.float64)

    def pdist(P):
        P = P.numpy() if torch.is_tensor(P) else np.asarray(P)
        a1 = x[P[0]]; b1 = x[P[1]]
        return np.sqrt(((a1-b1)**2).sum(-1))

    if a.split == "val":
        cand = torch.cat([d["val_edges"], d["val_neg"]], dim=1)
    else:
        cand = torch.load(Path(a.data_dir)/"test_inputs.pt")["candidates"]
    dist = pdist(cand)

    tr_e = d["train_edges"]; tr_n = d["train_neg"]
    rng = np.random.RandomState(42)
    N = min(400_000, tr_e.shape[1], tr_n.shape[1])
    ti = rng.choice(tr_e.shape[1], N, replace=False)
    nj = rng.choice(tr_n.shape[1], N, replace=False)
    tp = pdist(tr_e[:, ti]); tn = pdist(tr_n[:, nj])
    hi = int(max(tp.max(), tn.max())) + 2
    edges = np.arange(0, hi+1, 1.0)
    bw = 0.5
    ph,_ = np.histogram(tp, bins=edges); ph = ph.astype(float) / tp.size
    nh,_ = np.histogram(tn, bins=edges); nh = nh.astype(float) / tn.size
    ps = gaussian_filter1d(ph, sigma=bw, mode="nearest")
    ns = gaussian_filter1d(nh, sigma=bw, mode="nearest")
    eps = 1e-9
    lr = np.log(ps+eps) - np.log(ns+eps)
    s = lr[np.clip(dist//1.0, 0, len(edges)-2).astype(int)]
    if not np.all(np.isfinite(s)):
        s = -np.log1p(dist)
    s = s - s.mean()
    print("v9 hist t=%.0fs n=%d" % (time.time()-t0, len(s)))
    out = dict()
    out["scores"] = torch.from_numpy(s.astype(np.float32))
    torch.save(out, Path(a.out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
