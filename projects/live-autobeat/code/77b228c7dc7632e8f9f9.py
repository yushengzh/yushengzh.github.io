from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    t0 = time.time()
    d = torch.load(data_dir / "dataset.pt")
    x = d["x"].numpy().astype(np.float64)
    tr_e = d["train_edges"].numpy(); tr_n = d["train_neg"].numpy()
    if a.split == "val":
        VP = np.concatenate([d["val_edges"].numpy(), d["val_neg"].numpy()], axis=1)
    else:
        VP = torch.load(data_dir / "test_inputs.pt")["candidates"].long().numpy()

    # Compute candidate distances
    a_ = x[VP[0]]; b_ = x[VP[1]]
    dist = np.sqrt(((a_ - b_)**2).sum(-1))

    # Try to fit 2-Gaussian params on train subsample
    rng = np.random.RandomState(42)
    N_sub = 100_000
    tp_idx = rng.choice(tr_e.shape[1], N_sub, replace=False)
    tn_idx = rng.choice(tr_n.shape[1], N_sub, replace=False)
    a_ = x[tr_e[0, tp_idx]]; b_ = x[tr_e[1, tp_idx]]
    tp_d = np.sqrt(((a_ - b_)**2).sum(-1))
    a_ = x[tr_n[0, tn_idx]]; b_ = x[tr_n[1, tn_idx]]
    tn_d = np.sqrt(((a_ - b_)**2).sum(-1))
    fit_d = np.concatenate([tp_d, tn_d])
    fit_y = np.concatenate([np.ones(N_sub), np.zeros(N_sub)])

    def g(d, sp, mu_p, sn, mu_n):
        return np.exp(-((d - mu_p)**2) / (2.0 * sp * sp)) - np.exp(-((d - mu_n)**2) / (2.0 * sn * sn))

    best_auc = 0.0
    best_params = None
    for sp in [2.5, 3.0, 4.0, 5.0]:
        for mu_p in [3, 4, 5, 6]:
            for sn in [10, 15, 20, 25, 30]:
                for mu_n in [24, 28, 32, 36, 40]:
                    auc = roc_auc_score(fit_y, g(fit_d, sp, mu_p, sn, mu_n))
                    if auc > best_auc:
                        best_auc = auc
                        best_params = (sp, mu_p, sn, mu_n)

    t_fit = time.time() - t0
    print(f"fit t={t_fit:.0f}s best_auc={best_auc:.5f} params={best_params}")

    # Use gaussian if fit AUC is clearly better than monotone (0.91865 on train)
    if best_params is not None and best_auc > 0.922:
        sp, mu_p, sn, mu_n = best_params
        s = g(dist, sp, mu_p, sn, mu_n)
        print(f"v7 gaussian t={time.time()-t0:.0f}s n={len(s)} p={best_params}")
    else:
        s = -np.log1p(dist)
        print(f"v7 monotone-fallback t={time.time()-t0:.0f}s n={len(s)}")

    torch.save({"scores": torch.from_numpy(s.astype(np.float32))}, Path(a.out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
