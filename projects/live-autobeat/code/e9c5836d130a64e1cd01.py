#!/usr/bin/env python3
"""Improved GIN-VN ensemble + fingerprint forest for OGB molecule tasks.

v1 changes vs reference:
- Fingerprint: radius 3 (was 2), 4096 bits (was 2048), all 3 edge_attr cols (was bond only)
- GIN: 4 seeds (was 3), dropout 0.3 (was 0.5), dim 384 (was 300), layers 6 (was 5)
- Ensemble: finer weight grid {0.0,0.1,...,1.0} (was {0,0.25,0.5,0.75,1.0})
- LR schedule: cosine decay within each seed's budget

    python solve.py --data-dir DATA --out OUT.pt [--split val|test]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

T0 = time.time()
RUN_SEED = int(os.environ.get("REF_SEED", 0))
GNN_BUDGET = float(os.environ.get("REF_GNN_BUDGET_SEC", 2400))
SEEDS = int(os.environ.get("REF_SEEDS", 4))
FP_BITS = 4096
FP_RADIUS = 3
GNN_DIM = 384
GNN_LAYERS = 6
GNN_DROP = 0.3


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def to_pyg(d) -> list:
    x = d["x"].long()
    ei = d["edge_index"].long()
    ea = d["edge_attr"].long()
    y = d["y"]
    nptr, eptr = d["node_ptr"].tolist(), d["edge_ptr"].tolist()
    out = []
    for i in range(int(d["num_graphs"])):
        n0, n1, e0, e1 = nptr[i], nptr[i + 1], eptr[i], eptr[i + 1]
        out.append(Data(x=x[n0:n1], edge_index=ei[:, e0:e1] - n0,
                        edge_attr=ea[e0:e1], y=y[i:i + 1]))
    return out


def fingerprints(d) -> np.ndarray:
    """Hashed circular fingerprint counts, [G, FP_BITS] float32."""
    x = d["x"].long().numpy()
    ei = d["edge_index"].long().numpy()
    ea = d["edge_attr"].long().numpy()
    nptr = d["node_ptr"].numpy()
    G = int(d["num_graphs"])
    graph_of_node = np.repeat(np.arange(G), np.diff(nptr))

    M = np.uint64(0x9E3779B97F4A7C15)

    def mix(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.uint64)
        with np.errstate(over="ignore"):
            a ^= a >> np.uint64(31)
            a *= M
            a ^= a >> np.uint64(29)
        return a

    # radius 0: the atom invariant (all 9 OGB atom features)
    h = np.zeros(x.shape[0], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(x.shape[1]):
            h = mix(h * np.uint64(1000003) + x[:, j].astype(np.uint64) + np.uint64(j + 1))

    # bond feature: hash all 3 edge_attr columns
    bond = np.zeros(ea.shape[0], dtype=np.uint64)
    for j in range(ea.shape[1]):
        bond = mix(bond * np.uint64(1000003) + ea[:, j].astype(np.uint64) + np.uint64(j + 7))

    layers = [h]
    src, dst = ei[0], ei[1]
    for r in range(FP_RADIUS):
        with np.errstate(over="ignore"):
            msg = mix(layers[-1][src] * np.uint64(31) + bond)
            agg = np.zeros_like(h)
            np.add.at(agg, dst, msg)
            layers.append(mix(layers[-1] * np.uint64(1000003) + agg + np.uint64(r + 101)))
    fp = np.zeros((G, FP_BITS), dtype=np.float32)
    for hl in layers:
        np.add.at(fp, (graph_of_node, (hl % np.uint64(FP_BITS)).astype(np.int64)), 1.0)
    return fp


# --------------------------------------------------------------------------
# GIN with virtual node
# --------------------------------------------------------------------------
class GINConv(MessagePassing):
    def __init__(self, dim: int):
        super().__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim), torch.nn.BatchNorm1d(2 * dim),
            torch.nn.ReLU(), torch.nn.Linear(2 * dim, dim))
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.bond = BondEncoder(emb_dim=dim)

    def forward(self, x, edge_index, edge_attr):
        e = self.bond(edge_attr)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=e))

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class GINVN(torch.nn.Module):
    def __init__(self, out_dim: int, dim: int = GNN_DIM, layers: int = GNN_LAYERS, drop: float = GNN_DROP):
        super().__init__()
        self.layers, self.drop = layers, drop
        self.atom = AtomEncoder(dim)
        self.vn = torch.nn.Embedding(1, dim)
        torch.nn.init.constant_(self.vn.weight, 0)
        self.convs = torch.nn.ModuleList(GINConv(dim) for _ in range(layers))
        self.bns = torch.nn.ModuleList(torch.nn.BatchNorm1d(dim) for _ in range(layers))
        self.vn_mlps = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(dim, 2 * dim), torch.nn.BatchNorm1d(2 * dim), torch.nn.ReLU(),
                torch.nn.Linear(2 * dim, dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU())
            for _ in range(layers - 1))
        self.head = torch.nn.Linear(dim, out_dim)

    def forward(self, b):
        vn = self.vn(torch.zeros(int(b.num_graphs), dtype=torch.long, device=b.x.device))
        h = self.atom(b.x)
        for l in range(self.layers):
            h = h + vn[b.batch]
            h_new = self.bns[l](self.convs[l](h, b.edge_index, b.edge_attr))
            if l < self.layers - 1:
                h_new = F.relu(h_new)
            h_new = F.dropout(h_new, self.drop, self.training)
            if l < self.layers - 1:
                vn_in = vn + torch.zeros_like(vn).index_add_(0, b.batch, h)
                vn = F.dropout(self.vn_mlps[l](vn_in), self.drop, self.training)
            h = h_new
        return self.head(global_mean_pool(h, b.batch, int(b.num_graphs)))


def loss_fn(out, y, regression: bool):
    mask = ~torch.isnan(y)
    if regression:
        return F.mse_loss(out[mask], y[mask])
    return F.binary_cross_entropy_with_logits(out[mask], y[mask])


@torch.no_grad()
def predict(model, loader, dev, regression: bool) -> torch.Tensor:
    model.eval()
    outs = [model(b.to(dev)).cpu() for b in loader]
    out = torch.cat(outs)
    return out if regression else torch.sigmoid(out)


def evaluate(ev: Evaluator, y_true: torch.Tensor, pred: torch.Tensor) -> float:
    return float(next(iter(ev.eval({"y_true": y_true, "y_pred": pred}).values())))


def better(a: float, b: float, regression: bool) -> bool:
    return a < b if regression else a > b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    d = torch.load(Path(a.data_dir) / "dataset.pt", weights_only=False)
    ev = Evaluator(name=d["dataset"])
    regression = ev.eval_metric == "rmse"
    s = d["split_sizes"]
    tr = np.arange(s["train"])
    va = np.arange(s["train"], s["train"] + s["val"])
    te = np.arange(s["train"] + s["val"], int(d["num_graphs"]))
    target = va if a.split == "val" else te
    y_va = d["y"][va]
    n_tasks = d["y"].size(1)
    big = int(d["num_graphs"]) > 100_000
    log(f"{d['dataset']} metric={ev.eval_metric} graphs={d['num_graphs']} tasks={n_tasks}")

    # ---------------- GNN ensemble ----------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = to_pyg(d)
    bs = 256 if big else 32
    kw = dict(batch_size=bs, num_workers=0)
    tr_loader = DataLoader([graphs[i] for i in tr], shuffle=True, **kw)
    va_loader = DataLoader([graphs[i] for i in va], **kw)
    tg_loader = DataLoader([graphs[i] for i in target], **kw)
    max_epochs = 200
    n_seeds = 1 if big else SEEDS
    per_seed = GNN_BUDGET / n_seeds
    gnn_va, gnn_tg = [], []
    for seed in range(RUN_SEED * n_seeds, (RUN_SEED + 1) * n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GINVN(n_tasks).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # cosine LR schedule
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
        t_seed, best, best_va, best_tg, ep = time.time(), None, None, None, 0
        while ep < max_epochs and time.time() - t_seed < per_seed:
            model.train()
            for b in tr_loader:
                b = b.to(dev)
                opt.zero_grad()
                loss_fn(model(b), b.y.float(), regression).backward()
                opt.step()
            sched.step()
            ep += 1
            p_va = predict(model, va_loader, dev, regression)
            sc = evaluate(ev, y_va, p_va)
            if best is None or better(sc, best, regression):
                best, best_va = sc, p_va
                best_tg = predict(model, tg_loader, dev, regression)
        log(f"seed {seed}: {ep} epochs, best val {ev.eval_metric}={best:.4f}")
        gnn_va.append(best_va)
        gnn_tg.append(best_tg)
    gnn_va = torch.stack(gnn_va).mean(0)
    gnn_tg = torch.stack(gnn_tg).mean(0)
    log(f"GNN ensemble val {ev.eval_metric}={evaluate(ev, y_va, gnn_va):.4f}")

    # ---------------- fingerprint forest ----------------
    final_tg = gnn_tg
    if not big:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        fp = fingerprints(d)
        y = d["y"].numpy()
        rf_va = np.zeros((len(va), n_tasks), dtype=np.float32)
        rf_tg = np.zeros((len(target), n_tasks), dtype=np.float32)
        for t in range(n_tasks):
            ok = ~np.isnan(y[tr, t])
            Xt, yt = fp[tr][ok], y[tr, t][ok]
            if regression:
                rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=RUN_SEED)
                rf.fit(Xt, yt)
                rf_va[:, t], rf_tg[:, t] = rf.predict(fp[va]), rf.predict(fp[target])
            else:
                rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=RUN_SEED)
                rf.fit(Xt, yt.astype(int))
                rf_va[:, t] = rf.predict_proba(fp[va])[:, 1]
                rf_tg[:, t] = rf.predict_proba(fp[target])[:, 1]
        rf_va, rf_tg = torch.from_numpy(rf_va), torch.from_numpy(rf_tg)
        log(f"forest val {ev.eval_metric}={evaluate(ev, y_va, rf_va):.4f}")

        # finer weight grid
        best_w, best_sc = 0.0, None
        for w10 in range(11):
            w = w10 / 10.0
            sc = evaluate(ev, y_va, w * rf_va + (1 - w) * gnn_va)
            if best_sc is None or better(sc, best_sc, regression):
                best_w, best_sc = w, sc
        log(f"ensemble weight on forest={best_w:.1f} val {ev.eval_metric}={best_sc:.4f}")
        final_tg = best_w * rf_tg + (1 - best_w) * gnn_tg

    pred = final_tg if n_tasks > 1 else final_tg.view(-1, 1)
    torch.save({"pred": pred}, Path(a.out))
    log(f"wrote {pred.shape[0]} {a.split} predictions -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
