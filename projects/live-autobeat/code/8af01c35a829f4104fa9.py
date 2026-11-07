#!/usr/bin/env python3
"""GIN-VN for OGB mol tasks. Fast config for shared GPU + 30-min selfcheck."""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def to_pyg(d):
    n = d["node_ptr"].numpy(); e = d["edge_ptr"].numpy()
    x = d["x"].long(); ei = d["edge_index"].long(); ea = d["edge_attr"].long(); y = d["y"]
    G = []
    for i in range(len(n)-1):
        a,b = int(n[i]),int(n[i+1]); c,d_ = int(e[i]),int(e[i+1])
        ei_g = ei[:, c:d_].clone()
        if ei_g.numel()>0:
            ei_g[0]-=a; ei_g[1]-=a
        G.append(Data(x=x[a:b], edge_index=ei_g, edge_attr=ea[c:d_],
                      y=y[i] if not torch.isnan(y[i]).any() else y[i]))
    return G

class GINConv(MessagePassing):
    def __init__(self, dim):
        super().__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim,2*dim), torch.nn.BatchNorm1d(2*dim),
            torch.nn.ReLU(), torch.nn.Linear(2*dim,dim))
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.bond = BondEncoder(emb_dim=dim)
    def forward(self,x,edge_index,edge_attr):
        e = self.bond(edge_attr)
        return self.mlp((1+self.eps)*x + self.propagate(edge_index, x=x, edge_attr=e))
    def message(self,x_j,edge_attr):
        return F.relu(x_j+edge_attr)

class GINVN(torch.nn.Module):
    def __init__(self,out_dim,dim=None,layers=None,drop=None):
        import os
        if dim is None: dim = int(os.environ.get("REF_DIM", 300))
        if layers is None: layers = int(os.environ.get("REF_LAYERS", 5))
        if drop is None: drop = float(os.environ.get("REF_DROP", 0.5))
        super().__init__()
        self.layers,self.drop = layers,drop
        self.atom = AtomEncoder(dim)
        self.vn = torch.nn.Embedding(1,dim)
        torch.nn.init.constant_(self.vn.weight,0)
        self.convs = torch.nn.ModuleList(GINConv(dim) for _ in range(layers))
        self.bns = torch.nn.ModuleList(torch.nn.BatchNorm1d(dim) for _ in range(layers))
        self.vn_mlps = torch.nn.ModuleList(torch.nn.Sequential(
            torch.nn.Linear(dim,2*dim), torch.nn.BatchNorm1d(2*dim), torch.nn.ReLU(),
            torch.nn.Linear(2*dim,dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU()) for _ in range(layers-1))
        self.head = torch.nn.Linear(dim,out_dim)
    def forward(self,b):
        vn = self.vn(torch.zeros(int(b.num_graphs),dtype=torch.long,device=b.x.device))
        h = self.atom(b.x)
        for l in range(self.layers):
            h = h + vn[b.batch]
            h_new = self.bns[l](self.convs[l](h, b.edge_index, b.edge_attr))
            if l < self.layers-1: h_new = F.relu(h_new)
            h_new = F.dropout(h_new, self.drop, self.training)
            if l < self.layers-1:
                vn_in = vn + torch.zeros_like(vn).index_add_(0, b.batch, h)
                vn = F.dropout(self.vn_mlps[l](vn_in), self.drop, self.training)
            h = h_new
        return self.head(global_mean_pool(h, b.batch, int(b.num_graphs)))

def make_loss(pos_w):
    def loss_fn(out, y, regression):
        if regression: return F.mse_loss(out, y)
        if y.dim() == 1: y = y.reshape(out.shape)
        mask = ~torch.isnan(y)
        if pos_w is None:
            o = out[mask]; t = y[mask]
            return F.binary_cross_entropy_with_logits(o, t)
        # pos_w is [T]; use 2D broadcasting to avoid shape mismatch
        loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_w.to(out.device).view(1,-1), reduction='none')
        mask_f = mask.float()
        return (loss * mask_f).sum() / mask_f.sum().clamp(min=1)
    return loss_fn

@torch.no_grad()
def predict(model, loader, dev, regression):
    model.eval(); outs=[model(b.to(dev)).cpu() for b in loader]
    out = torch.cat(outs); return out if regression else torch.sigmoid(out)

def evaluate(ev, y_true, pred): return float(next(iter(ev.eval({"y_true":y_true,"y_pred":pred}).values())))

def cosine_lr(opt, base_lr, epoch, max_epoch, warmup=2):
    if epoch < warmup:
        lr = base_lr * (epoch+1) / warmup
    else:
        prog = (epoch - warmup) / max(1, max_epoch - warmup)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * prog))
    for g in opt.param_groups: g["lr"] = lr
    return lr

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    a = ap.parse_args()

    d = torch.load(Path(a.data_dir)/"dataset.pt", weights_only=False)
    ev = Evaluator(name=d["dataset"])
    regression = ev.eval_metric == "rmse"
    s = d["split_sizes"]
    tr = np.arange(s["train"])
    va = np.arange(s["train"], s["train"]+s["val"])
    te = np.arange(s["train"]+s["val"], int(d["num_graphs"]))
    target = va if a.split == "val" else te
    y_va = d["y"][va]
    n_tasks = d["y"].size(1)
    big = int(d["num_graphs"]) > 100_000
    log(f"{d['dataset']} metric={ev.eval_metric} graphs={d['num_graphs']} tasks={n_tasks} big={big}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"dev={dev} {time.time()-T0:.1f}s")
    graphs = to_pyg(d)
    log(f"graphs loaded {len(graphs)} {time.time()-T0:.1f}s")

    BS = 1024 if big else 32
    GNN_BUDGET = float(os.environ.get("REF_GNN_BUDGET_SEC", 3300))
    MAX_EPOCHS = int(os.environ.get("REF_MAX_EPOCHS", 200))
    LR = float(os.environ.get("REF_LR", 1e-3))
    WD = float(os.environ.get("REF_WD", 1e-4))
    CLIP = float(os.environ.get("REF_CLIP", 1.0))
    WARMUP = int(os.environ.get("REF_WARMUP", 2))
    POS_W = float(os.environ.get("REF_POS_W", 0.0))
    EVAL_EVERY = int(os.environ.get("REF_EVAL_EVERY", 2))
    RUN_SEED = int(os.environ.get("REF_SEED", 0))
    log(f"bs={BS} lr={LR} wd={WD} clip={CLIP} warmup={WARMUP} pos_w={'on' if POS_W>0 else 'off'} eval_every={EVAL_EVERY} budget={GNN_BUDGET:.0f}s")

    kw = dict(batch_size=BS, num_workers=0)
    tr_loader = DataLoader([graphs[i] for i in tr], shuffle=True, **kw)
    va_loader = DataLoader([graphs[i] for i in va], **kw)
    tg_loader = DataLoader([graphs[i] for i in target], **kw)

    # per-task pos_weight from train split
    pos_w = None
    if POS_W > 0:
        ytr = d["y"][tr].numpy()
        mask = ~np.isnan(ytr)
        pos = (ytr==1).sum(0)
        tot = np.maximum(mask.sum(0), 1)
        pw = np.minimum((tot-pos)/np.maximum(pos,1), POS_W)
        pos_w = torch.tensor(pw, dtype=torch.float32).to(dev)
        log(f"pos_weight min={pw.min():.2f} med={np.median(pw):.2f} max={pw.max():.2f}")

    loss_fn = make_loss(pos_w)
    n_seeds = 1 if big else 3
    per_seed = GNN_BUDGET / n_seeds
    gnn_va, gnn_tg = [], []

    for seed in range(RUN_SEED*n_seeds, (RUN_SEED+1)*n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = GINVN(n_tasks).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        t_seed, best, best_va, best_tg, ep = time.time(), None, None, None, 0
        while ep < MAX_EPOCHS and time.time() - t_seed < per_seed:
            model.train()
            t_ep = time.time()
            for b in tr_loader:
                b = b.to(dev)
                opt.zero_grad()
                loss_fn(model(b), b.y.float(), regression).backward()
                if CLIP > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
            ep += 1
            cosine_lr(opt, LR, ep, MAX_EPOCHS, WARMUP)
            if ep % EVAL_EVERY == 0 or ep == MAX_EPOCHS:
                p_va = predict(model, va_loader, dev, regression)
                sc = evaluate(ev, y_va, p_va)
                log(f"seed {seed} ep{ep} val={sc:.4f} t={time.time()-t_ep:.1f}s")
                if best is None or (sc > best if not regression else sc < best):
                    best, best_va = sc, p_va
                    best_tg = predict(model, tg_loader, dev, regression)
            else:
                log(f"seed {seed} ep{ep} train t={time.time()-t_ep:.1f}s (no eval)")
        log(f"seed {seed}: {ep} epochs, best val={best:.4f}")
        gnn_va.append(best_va); gnn_tg.append(best_tg)

    gnn_va = torch.stack(gnn_va).mean(0)
    gnn_tg = torch.stack(gnn_tg).mean(0)
    log(f"GNN ensemble val {ev.eval_metric}={evaluate(ev, y_va, gnn_va):.4f}")
    pred = gnn_tg if n_tasks > 1 else gnn_tg.view(-1,1)
    torch.save({"pred": pred}, Path(a.out))
    log(f"wrote {pred.shape[0]} {a.split} preds -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
