#!/usr/bin/env python3
"""Molecular Activity Prediction (ROC-AUC).
CPU-safe: OGB-style hashed circular fingerprint (radius 2, 2048 bits,
all 9 atom features, bond col0) + RandomForest(500).
"""
import argparse, os, time, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

M = np.uint64(0x9E3779B97F4A7C15)

def mix(a):
    a = a.astype(np.uint64)
    with np.errstate(over='ignore'):
        a ^= a >> np.uint64(31); a *= M; a ^= a >> np.uint64(29)
    return a

def build_fp(x, ei, ea, nptr, radius=2, FP_BITS=2048):
    x = x.long().numpy()
    ei = ei.long().numpy()
    ea = ea.long().numpy()
    G = int(nptr.shape[0] - 1)
    gon = np.repeat(np.arange(G), np.diff(nptr))
    h = np.zeros(x.shape[0], dtype=np.uint64)
    with np.errstate(over='ignore'):
        for j in range(x.shape[1]):
            h = mix(h * np.uint64(1000003) + x[:, j].astype(np.uint64) + np.uint64(j + 1))
    bond = ea[:, 0].astype(np.uint64) + np.uint64(7)
    layers = [h]
    src, dst = ei[0], ei[1]
    for r in range(radius):
        with np.errstate(over='ignore'):
            msg = mix(layers[-1][src] * np.uint64(31) + bond)
            agg = np.zeros_like(h)
            np.add.at(agg, dst, msg)
            layers.append(mix(layers[-1] * np.uint64(1000003) + agg + np.uint64(r + 101)))
    fp = np.zeros((G, FP_BITS), dtype=np.float32)
    for hl in layers:
        np.add.at(fp, (gon, (hl % np.uint64(FP_BITS)).astype(np.int64)), 1.0)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    a = ap.parse_args()
    t0 = time.time()
    import torch
    data = torch.load(os.path.join(a.data_dir, 'dataset.pt'), map_location='cpu')
    x = data['x']; ei = data['edge_index']; ea = data['edge_attr']
    nptr = data['node_ptr']; ep = data['edge_ptr']; y = data['y']
    ss = data['split_sizes']; G = int(data.get('num_graphs', nptr.shape[0] - 1))
    tr = int(ss['train']); va = tr + int(ss['val']); te = va + int(ss['test'])

    fp = build_fp(x, ei, ea, nptr.numpy())

    y = y.numpy()
    if a.split == 'test':
        ytr = y[:tr, 0]
        ok = ~np.isnan(ytr); ytrc = ytr[ok].astype(int)
        Xtr = fp[:tr][ok]
        Xte = fp[va:te]
    else:
        ytr = y[:tr, 0]
        ok = ~np.isnan(ytr); ytrc = ytr[ok].astype(int)
        Xtr = fp[:tr][ok]
        Xte = fp[tr:va]

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                class_weight='balanced_subsample', n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytrc)
    pred = rf.predict_proba(Xte)[:, 1]

    if a.split == 'test':
        assert pred.shape[0] == int(ss['test']), (pred.shape, ss['test'])
    out = {'pred': torch.from_numpy(pred.reshape(-1, 1).astype(np.float64))}
    torch.save(out, a.out)
    print(f'[done] split={a.split} n={pred.shape[0]} elapsed={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
