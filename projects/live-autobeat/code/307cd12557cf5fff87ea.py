#!/usr/bin/env python3
"""Improved node classification: SIGN features + 1-hop label feature + MLP head.

    python solve.py --data-dir DATA --out OUT.pt [--split val|test]

Method (v10): v9 pipeline + APPNP GNN head (extra ensemble members).
  * Standardise raw features X.
  * Build symmetric-normalised undirected adjacency A (self-loops added).
  * SIGN propagation: [X, A X, A^2 X, ..., A^H X] concat  (H hops).
  * 1-hop label feature: row-normalised mean one-hot of TRAIN labels over each
    node's neighbours (self EXCLUDED so a train node never sees its own label).
  * Standardise the whole concat.
  * MLP head (Linear->BN->ReLU->Dropout x2 -> Linear), batched CE training.
  * APPNP GNN head: residual + diffusion on sym-norm adjacency over 128-dim
    standardized features, small MLP head on top. Trained independently and
    added as extra ensemble members.
  * AdamW + CosineAnnealing + label smoothing; ensemble SEEDS_MLP + SEEDS_GNN
    runs by averaging softmax probabilities; argmax for the final prediction.
"""
from __future__ import annotations
import argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
SEEDS   = int(os.environ.get("SEEDS", 5))       # MLP seeds
GNN_SEEDS = int(os.environ.get("GNN_SEEDS", 0))  # GNN seeds (0=off)
H       = int(os.environ.get("H", 9))          # SIGN hops (k=1..H)
HIDDEN  = int(os.environ.get("HIDDEN", 512))
MLP_LAYERS = int(os.environ.get("MLP_LAYERS", 3))
EPOCHS  = int(os.environ.get("EPOCHS", 300))
LR      = float(os.environ.get("LR", 1e-3))
WD      = float(os.environ.get("WD", 1e-4))
DROPOUT = float(os.environ.get("DROP", 0.5))
BATCH   = int(os.environ.get("BATCH", 50_000))
LS      = float(os.environ.get("LS", 0.0))     # label smoothing
PATIENCE= int(os.environ.get("PATIENCE", 100))
VAL_EVERY = int(os.environ.get("VAL_EVERY", 5))

# GNN hyperparams
GNN_LAYERS = int(os.environ.get("GNN_LAYERS", 2))   # num GCN/APPNP layers
GNN_K      = int(os.environ.get("GNN_K", 10))       # APPNP diffusion steps
GNN_ALPHA  = float(os.environ.get("GNN_ALPHA", 0.1)) # residual mix (1-alpha)*X + alpha*A*X
GNN_HIDDEN = int(os.environ.get("GNN_HIDDEN", 256))
GNN_DROPOUT = float(os.environ.get("GNN_DROP", 0.2))


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def build_adj(edge_index, n, self_loops, norm):
    """Sym or row normalised sparse CSR adjacency."""
    ei = edge_index.long()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    if self_loops:
        ar = torch.arange(n)
        ei = torch.cat([ei, torch.stack([ar, ar])], dim=1)
    a = torch.sparse_coo_tensor(ei, torch.ones(ei.size(1), device=ei.device), (n, n)).coalesce()
    idx = a.indices()
    v = torch.ones(idx.size(1), device=idx.device)
    deg = torch.zeros(n, device=idx.device).index_add_(0, idx[0], v).clamp(min=1)
    if norm == "sym":
        v = deg[idx[0]].rsqrt() * deg[idx[1]].rsqrt()
    else:  # row: neighbour mean
        v = 1.0 / deg[idx[0]]
    return torch.sparse_coo_tensor(idx, v, (n, n)).coalesce().to_sparse_csr()


class MLPHead(nn.Module):
    def __init__(self, in_c, hidden, out_c, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_c
        for i in range(n_layers):
            nxt = hidden if i < n_layers - 1 else out_c
            layers.append(nn.Linear(prev, nxt))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(nxt))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev = nxt
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GCNConv(nn.Module):
    """Simple GCN layer: X @ W, then sym-normalised aggregation."""
    def __init__(self, in_c, out_c, a_sym, alpha, k):
        super().__init__()
        self.lin = nn.Linear(in_c, out_c)
        self.a = a_sym
        self.alpha = alpha
        self.k = k

    def forward(self, x):
        x = self.lin(x)
        # APPNP-style: residual + k steps of diffusion
        h = x
        for _ in range(self.k):
            h = self.alpha * x + (1 - self.alpha) * (self.a @ h)
        return h


class GNNHead(nn.Module):
    def __init__(self, in_c, out_c, a_sym, n_layers=2, hidden=256,
                 alpha=0.1, k=10, dropout=0.2):
        super().__init__()
        convs = []
        prev = in_c
        for i in range(n_layers):
            nxt = hidden if i < n_layers - 1 else out_c
            convs.append(GCNConv(prev, nxt, a_sym, alpha, k))
            prev = nxt
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


def train_gnn(Xtr, Xva, Xtg, ytr, yva, c, a_sym, seed, epochs, lr, hidden,
              n_layers, alpha, k, dropout, batch, patience, val_every):
    """Train one GNN head, return best softmax probs for target."""
    torch.manual_seed(seed)
    model = GNNHead(128, c, a_sym, n_layers=n_layers, hidden=hidden,
                    alpha=alpha, k=k, dropout=dropout).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    best, best_p = -1.0, None
    best_ep, patience_cnt = 0, 0
    n_tr = Xtr.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device='cuda')
        for i in range(0, perm.numel(), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[b]), ytr[b], label_smoothing=0.0).backward()
            opt.step()
        sched.step()
        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(Xva).argmax(1) == yva).float().mean().item()
                if acc > best:
                    best = acc; best_ep = ep; patience_cnt = 0
                    best_p = model(Xtg).softmax(1)
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience // val_every and ep > 30:
                        break
    log(f"gnn seed {seed}: best val acc {best:.4f} @ ep {best_ep}")
    del model, opt, sched
    torch.cuda.empty_cache()
    return best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(f"{a.data_dir}/dataset.pt", weights_only=False)
    x, y = d["x"].float(), d["y"].long()
    n, c = x.size(0), int(d["num_classes"])
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    if a.split == "val":
        target = va
        tr_idx = tr
        va_idx = va
    else:
        target = te
        # Train on train+val (val labels are known), validate on 10% holdout of train
        all_train = torch.cat([tr, va], dim=0)
        n_all = len(all_train)
        _gen = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_all, generator=_gen)
        n_hv = n_all // 10  # 10% holdout for validation
        va_idx = all_train[perm[:n_hv]]
        tr_idx = all_train[perm[n_hv:]]
    log(f"graph: N={n}, E={d['edge_index'].size(1)}, F={x.size(1)}, C={c}")
    log(f"train={len(tr)}, val={len(va)}, test={len(te)} | SEEDS={SEEDS} GNN_SEEDS={GNN_SEEDS} H={H} HIDDEN={HIDDEN} LAYERS={MLP_LAYERS} EPOCHS={EPOCHS}")

    # --- Standardise raw features ---
    mu, sd = x.mean(0), x.std(0).clamp(min=1e-6)
    xs = (x - mu) / sd
    del x

    # --- SIGN propagation on sym-norm adjacency ---
    a_sym = build_adj(d["edge_index"], n, True, "sym").to(dev)
    feats = [xs.to(dev)]
    cur = feats[0].contiguous()
    for k in range(H):
        cur = a_sym @ cur
        feats.append(cur)

    # --- 1-hop label feature (self excluded, row-norm) ---
    onehot = torch.zeros(n, c)
    onehot[tr_idx, y[tr_idx]] = 1.0
    a_row = build_adj(d["edge_index"], n, False, "row").to(dev)
    lab = a_row @ onehot.to(dev)
    del a_row, onehot
    feats.append(lab)

    # --- Standardise the whole concat (on CPU to cut GPU peak) ---
    X = torch.cat(feats, dim=1)
    del feats, cur, xs
    X = X.cpu()
    xm = X.mean(0); xstd = X.std(0).clamp(min=1e-6)
    X = (X - xm) / xstd
    del xm, xstd
    log(f"features {tuple(X.shape)}")
    X = X.to(dev)

    ytr = y[tr_idx].to(dev); yva = y[va_idx].to(dev)

    # Move split feature views to GPU
    Xtr = X[tr_idx.to(dev)].contiguous()
    Xva = X[va_idx.to(dev)].contiguous()
    Xtg = X[target.to(dev)].contiguous()
    del X
    torch.cuda.empty_cache()
    in_dim = Xtr.size(1)
    log(f"train MLP: in={in_dim} hidden={HIDDEN} out={c} layers={MLP_LAYERS} batch={BATCH}")

    total_seeds = SEEDS + GNN_SEEDS
    probs = torch.zeros(len(target), c, device=dev)

    # --- MLP ensemble ---
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        model = MLPHead(in_dim, HIDDEN, c, MLP_LAYERS, DROPOUT).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
        best, best_p = -1.0, None
        best_ep, patience = 0, 0
        n_tr = Xtr.size(0)
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n_tr, device=dev)
            for i in range(0, perm.numel(), BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad()
                F.cross_entropy(model(Xtr[b]), ytr[b], label_smoothing=LS).backward()
                opt.step()
            sched.step()
            if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
                model.eval()
                with torch.no_grad():
                    acc = (model(Xva).argmax(1) == yva).float().mean().item()
                    if acc > best:
                        best = acc; best_ep = ep; patience = 0
                        best_p = model(Xtg).softmax(1)
                    else:
                        patience += 1
                        if patience >= PATIENCE // VAL_EVERY and ep > 30:
                            break
        log(f"seed {seed}: best val acc {best:.4f} @ ep {best_ep}")
        probs += best_p
        del model, opt, sched
        torch.cuda.empty_cache()

    # --- GNN ensemble (APPNP) ---
    if GNN_SEEDS > 0:
        # Use sym-norm adjacency for GNN (a_sym still in memory from SIGN)
        gnn_xtr = xs_cuda = None
        # Re-standardize raw X on GPU for GNN input
        # We already have xs on CPU before del; rebuild
        # Actually xs was deleted. Use the standardized raw features from the concat:
        # The first 128 dims of Xtr are the standardized raw features.
        gnn_in_dim = 128
        log(f"train GNN: in={gnn_in_dim} hidden={GNN_HIDDEN} out={c} layers={GNN_LAYERS} k={GNN_K} alpha={GNN_ALPHA}")
        for gseed in range(GNN_SEEDS):
            gp = train_gnn(
                Xtr[:, :gnn_in_dim].contiguous(),
                Xva[:, :gnn_in_dim].contiguous(),
                Xtg[:, :gnn_in_dim].contiguous(),
                ytr, yva, c, a_sym,
                seed=100 + gseed, epochs=EPOCHS, lr=LR,
                hidden=GNN_HIDDEN, n_layers=GNN_LAYERS,
                alpha=GNN_ALPHA, k=GNN_K, dropout=GNN_DROPOUT,
                batch=BATCH, patience=PATIENCE, val_every=VAL_EVERY)
            probs += gp

    if dev.type == 'cuda':
        log(f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    pred = (probs / total_seeds).argmax(1)
    torch.save({"pred": pred.cpu()}, a.out)
    log(f"wrote {len(target)} {a.split} predictions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
