#!/usr/bin/env python3
import argparse, os, sys, time
import torch
import torch.nn as nn

def pair_features(x, u, v, chunk=1_000_000):
    n = u.numel()
    dists = torch.empty(n)
    nds = torch.empty(n)
    ang = torch.empty(n)
    m = x.shape[0]
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        uc = u[i:j].long().clamp(0, m - 1)
        vc = v[i:j].long().clamp(0, m - 1)
        xu = x[uc]
        xv = x[vc]
        d = torch.linalg.vector_norm(xu - xv, dim=-1)
        nu = torch.linalg.vector_norm(xu, dim=-1)
        nv = torch.linalg.vector_norm(xv, dim=-1)
        nd = (nu - nv).abs()
        cos = (xu * xv).sum(-1) / (nu * nv + 1e-8)
        a = 1.0 - cos
        # out-of-range pairs -> mark dist as +inf-equivalent (very large)
        bad = ~(((u[i:j] >= 0) & (u[i:j] < m)) & ((v[i:j] >= 0) & (v[i:j] < m)))
        if bad.any():
            d[bad] = 1e9
            nd[bad] = 1e9
            a[bad] = 1e9
        dists[i:j] = d
        nds[i:j] = nd
        ang[i:j] = a
    return torch.stack([dists, nds, ang], dim=0)

def negative_dist(x, u, v, chunk=1_000_000):
    n = u.numel()
    out = torch.empty(n)
    m = x.shape[0]
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        uc = u[i:j].long().clamp(0, m - 1)
        vc = v[i:j].long().clamp(0, m - 1)
        d = torch.linalg.vector_norm(x[uc] - x[vc], dim=-1)
        bad = ~(((u[i:j] >= 0) & (u[i:j] < m)) & ((v[i:j] >= 0) & (v[i:j] < m)))
        if bad.any():
            d[bad] = 1e9
        out[i:j] = -d
    return out

def train_mlp(x, train_edges, train_neg, device, epochs=100, lr=5e-4, wd=0.0):
    try:
        t0 = time.time()
        pos_u = train_edges[0].to(device)
        pos_v = train_edges[1].to(device)
        neg_u = train_neg[0].to(device)
        neg_v = train_neg[1].to(device)
        T = pos_u.numel()
        x_dev = x.to(device)
        feat_pos = pair_features(x_dev, pos_u, pos_v, 2_000_000).cpu()
        feat_neg = pair_features(x_dev, neg_u, neg_v, 2_000_000).cpu()
        all_feats = torch.cat([feat_pos, feat_neg], dim=1)
        mu = all_feats.mean(1)
        sd = all_feats.std(1) + 1e-8
        feat_pos_z = (feat_pos - mu.unsqueeze(1)) / sd.unsqueeze(1)
        feat_neg_z = (feat_neg - mu.unsqueeze(1)) / sd.unsqueeze(1)
        X = torch.cat([feat_pos_z.t(), feat_neg_z.t()], dim=0).to(device)
        y = torch.cat([torch.ones(T), torch.zeros(T)]).to(device)
        model = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.BCEWithLogitsLoss()
        chunk = 65536
        for ep in range(epochs):
            model.train()
            p2 = torch.randperm(X.shape[0], device=device)
            for i in range(0, X.shape[0], chunk):
                idx = p2[i:i + chunk]
                opt.zero_grad()
                loss = crit(model(X[idx]).squeeze(-1), y[idx])
                loss.backward()
                opt.step()
            if (ep + 1) % 20 == 0:
                print(f"  ep{ep+1}: train_loss={loss.item():.5f}", flush=True)
        model.eval()
        print(f"MLP trained in {time.time()-t0:.1f}s", flush=True)
        return model, mu, sd
    except Exception as e:
        print(f"MLP training failed: {e}", flush=True)
        return None, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(os.path.join(args.data_dir, "dataset.pt"), map_location="cpu")
    x = d["x"].float().to(device)
    if args.split == "val":
        pos = d["val_edges"].long()
        neg = d["val_neg"]
        if neg.dim() == 3:
            V, K = neg.size(0), neg.size(1)
            cand = torch.empty(2, V * (K + 1), dtype=torch.long)
            cand[0, 0::K + 1] = pos[0]
            cand[1, 0::K + 1] = pos[1]
            cand[0, 1::K + 1] = neg[:, :, 0].reshape(-1)
            cand[1, 1::K + 1] = neg[:, :, 1].reshape(-1)
        else:
            cand = torch.cat([pos, neg], dim=1)
        cand = cand.long().to(device)
    else:
        tp = os.path.join(args.data_dir, "test_inputs.pt")
        if not os.path.exists(tp):
            raise FileNotFoundError(f"{tp} not found for --split test")
        td = torch.load(tp, map_location="cpu")
        cand = td["candidates"].long().to(device)
    C = cand.shape[1]
    print(f"candidates: {list(cand.shape)}", flush=True)
    train_edges = d["train_edges"].long()
    train_neg = d["train_neg"].long()
    model, mu, sd = train_mlp(x, train_edges, train_neg, device)
    if model is not None and mu is not None:
        out = []
        sc = 500_000
        for i in range(0, C, sc):
            j = min(i + sc, C)
            feats = pair_features(x, cand[0, i:j], cand[1, i:j], 500_000).cpu()
            feats_z = (feats - mu.unsqueeze(1)) / sd.unsqueeze(1)
            model.to(device)
            with torch.no_grad():
                s = model(feats_z.t().to(device)).squeeze(-1)
            out.append(s.cpu().float())
        scores = torch.cat(out, dim=0)
    else:
        scores = negative_dist(x, cand[0], cand[1]).cpu().float()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"scores": scores}, args.out)
    print(f"Wrote {args.out}: {scores.numel()} scores. total {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
