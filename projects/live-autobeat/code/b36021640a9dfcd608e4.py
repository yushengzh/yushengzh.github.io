
#!/usr/bin/env python3
"""GCN encoder + cosine decoder for ogbl-citation2 link prediction."""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 64))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 1))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 1))
ENC_LR = float(os.environ.get("SOLVER_ENC_LR", 0.01))
DEC_LR = float(os.environ.get("SOLVER_DEC_LR", 0.01))
NSUB = int(os.environ.get("SOLVER_NSUB", 4_000_000))
CHUNK = int(os.environ.get("SOLVER_CHUNK", 3_000_000))
SEED = int(os.environ.get("SOLVER_SEED", 0))

def sym_norm(row, col, n, dev):
    deg = torch.zeros(n, dtype=torch.float32, device=dev).index_add_(
        0, row, torch.ones(row.numel(), dtype=torch.float32, device=dev))
    dinv = deg.clamp(min=1).pow(-0.5)
    return dinv[row] * dinv[col]

def mean_agg(h, row, col, w):
    out = torch.zeros_like(h)
    for i in range(0, row.numel(), CHUNK):
        r = row[i:i+CHUNK]
        out.index_add_(0, r, h[col[i:i+CHUNK]] * w[i:i+CHUNK].unsqueeze(-1))
    return out

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        self.lin = nn.ModuleList([nn.Linear(in_dim, hidden)] + [nn.Linear(hidden, hidden) for _ in range(layers-1)])
    def forward(self, x, row, col, w, drop=0.0):
        h = x
        for L in self.lin:
            h = L(h)
            if drop:
                h = F.dropout(h, drop, training=self.training)
            h = mean_agg(h, row, col, w)
            h = F.relu(h)
        return h

def dot_cos(z, edges, chunk=3_000_000):
    out = []
    for i in range(0, edges.size(1), chunk):
        e = edges[:, i:i+chunk]
        out.append((z[e[0]] * z[e[1]]).sum(-1))
    return torch.cat(out)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 4)
    dev = 'cpu'
    d = torch.load(Path(a.data_dir)/"dataset.pt", map_location='cpu')
    x = d["x"].float().to(dev)
    n = int(d["num_nodes"])
    ei = d["edge_index"].long().to(dev)
    row, col = ei[0], ei[1]
    w = sym_norm(row, col, n, dev)
    tr_pos, tr_neg = d["train_edges"].to(dev), d["train_neg"].to(dev)

    enc = GCN(x.size(1), HIDDEN, LAYERS).to(dev)
    log_temp = torch.zeros((), requires_grad=True, device=dev)
    opt = torch.optim.Adam(list(enc.parameters())+[log_temp], lr=ENC_LR)

    t_start = time.time()
    for ep in range(EPOCHS):
        enc.train()
        t_ep = time.time()
        with torch.no_grad():
            npair = min(NSUB, tr_pos.size(1))
            perm = torch.randperm(tr_pos.size(1))[:npair]
            p = tr_pos[:, perm]; q = tr_neg[:, perm]
        z = F.normalize(enc(x, row, col, w), dim=-1)
        temp = torch.exp(log_temp.clamp(max=6.0))
        sp = (z[p[0]]*z[p[1]]).sum(-1) * temp
        sq = (z[q[0]]*z[q[1]]).sum(-1) * temp
        logits = torch.cat([sp, sq])
        target = torch.cat([torch.ones(npair, device=dev), torch.zeros(npair, device=dev)])
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"ep {ep+1}/{EPOCHS} loss={loss.item():.4f} t={time.time()-t_ep:.1f}s cum={time.time()-t_start:.1f}s", flush=True)

    enc.eval()
    with torch.no_grad():
        z = F.normalize(enc(x, row, col, w), dim=-1)
        if a.split == "val":
            pos = d["val_edges"].to(dev); neg = d["val_neg"].to(dev)
            if neg.dim() == 3:
                cand = torch.cat([pos.t().unsqueeze(1), neg], dim=1).reshape(-1,2).t()
            else:
                cand = torch.cat([pos, neg], dim=1)
            cand = cand.long()
        else:
            cand = torch.load(Path(a.data_dir)/"test_inputs.pt")["candidates"].long().to(dev)
        scores = dot_cos(z, cand)
    torch.save(dict(scores=scores.cpu().float()), Path(a.out))
    print(f"wrote {scores.numel()} {a.split} scores -> {a.out} total={time.time()-t_start:.1f}s", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
