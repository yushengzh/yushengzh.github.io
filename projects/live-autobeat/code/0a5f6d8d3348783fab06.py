#!/usr/bin/env python3
import argparse, os, time, copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = int(os.environ.get('SOLVER_HIDDEN', 256))
LAYERS = int(os.environ.get('SOLVER_LAYERS', 3))
EPOCHS = int(os.environ.get('SOLVER_EPOCHS', 600))
LR = float(os.environ.get('SOLVER_LR', 1e-3))
WD = float(os.environ.get('SOLVER_WD', 5e-4))
DROPOUT = float(os.environ.get('SOLVER_DROPOUT', 0.5))
SEED = int(os.environ.get('SOLVER_SEED', 0))
PATIENCE = int(os.environ.get('SOLVER_PATIENCE', 80))
DEVICE = os.environ.get('SOLVER_DEVICE', 'cuda')
CHUNK = int(os.environ.get('SOLVER_CHUNK', 131072))

def build_csr(edge_index, n):
    ei = edge_index.long()
    row, col = ei[0].contiguous(), ei[1].contiguous()
    ar = torch.arange(n)
    row = torch.cat([row, ar]).contiguous()
    col = torch.cat([col, ar]).contiguous()
    m = row.numel()
    order = torch.argsort(row, stable=True)
    r = row[order]
    c = col[order]
    r0 = torch.cat([torch.zeros(1, dtype=torch.int64),
                    torch.searchsorted(r, torch.arange(1, n + 1, dtype=torch.int64)).to(torch.int64)])
    cptr = r0
    counts = r0[1:] - r0[:-1]
    deg = torch.zeros(n, dtype=torch.float64)
    deg.index_add_(0, r, torch.ones(m, dtype=torch.float64))
    dinv = deg.pow(-0.5).to(torch.float32)
    w = dinv[r.to(torch.int64)].to(torch.float32).contiguous()
    return {
        'cptr': cptr,
        'out_idx': c.to(torch.int64).contiguous(),
        'w': w,
        'dinv': dinv.contiguous(),
        'counts': counts.to(torch.int64).contiguous(),
    }

def _agg_fwd(h, cptr, oi, w, dinv, chunk, n):
    dev = h.device
    xj = h * dinv[None, :]
    out = torch.zeros(n, h.shape[1], device=dev, dtype=h.dtype)
    for i in range(0, n, chunk):
        rs = i + torch.arange(chunk, device=dev, dtype=torch.int64)
        rs = rs[rs < n]
        woi = w[cptr[rs]]
        out.index_add_(0, rs, xj[rs, :] * woi[:, None])
    out *= dinv[None, :]
    return out

class SparseAgg(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, cptr, oi, w, dinv, chunk, n):
        with torch.no_grad():
            out = _agg_fwd(h, cptr, oi, w, dinv, chunk, n)
        ctx.save_for_backward(cptr, oi, w, dinv)
        ctx.chunk = chunk
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, gout):
        cptr, oi, w, dinv = ctx.saved_tensors
        with torch.no_grad():
            n = ctx.n
            chunk = ctx.chunk
            dev = gout.device
            xg = gout * dinv[None, :]
            dh = torch.zeros(n, gout.shape[1], device=dev, dtype=gout.dtype)
            for i in range(0, n, chunk):
                rs = i + torch.arange(chunk, device=dev, dtype=torch.int64)
                rs = rs[rs < n]
                woi = w[cptr[rs]]
                dh.index_add_(0, rs, xg[rs, :] * woi[:, None])
            dh *= dinv[None, :]
        return dh, None, None, None, None, None, None

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers, dropout, csr, device):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.lin = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) for i in range(layers)])
        self.cptr = csr['cptr'].to(device)
        self.out_idx = csr['out_idx'].to(device)
        self.w = csr['w'].to(device)
        self.dinv = csr['dinv'].to(device)
        self.counts = csr['counts'].to(device)
        self.n = int(csr['counts'].numel())
        self.layers = layers
        self.dropout = dropout
        self.chunk = CHUNK

    def forward(self, x, training=False):
        for i, L in enumerate(self.lin):
            h = L(x)
            x = SparseAgg.apply(h, self.cptr, self.out_idx, self.w, self.dinv, self.chunk, self.n)
            if i != self.layers - 1:
                x = F.relu(x)
                if training:
                    x = F.dropout(x, p=self.dropout, training=True)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    a = ap.parse_args()
    torch.manual_seed(SEED)
    t0 = time.time()
    d = torch.load(Path(a.data_dir) / 'dataset.pt', map_location='cpu')
    xs = d['x'].float().to(DEVICE)
    ei = d['edge_index'].long()
    y = d['y'].long().to(DEVICE)
    tr = d['train_idx'].long().to(DEVICE)
    va = d['val_idx'].long().to(DEVICE)
    te = d['test_idx'].long().to(DEVICE)
    n = xs.size(0)
    num_classes = int(d['num_classes'])
    target = tr if a.split == 'val' else te
    csr = build_csr(ei, n)
    del ei
    model = GCN(xs.size(1), HIDDEN, num_classes, LAYERS, DROPOUT, csr, DEVICE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ytr = y[tr]
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(xs, training=True)
        loss = F.cross_entropy(out[tr], ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vo = model(xs, training=False)
            vacc = (vo[va].argmax(1) == y[va]).float().mean().item()
            del vo
        if ep % 10 == 0 or ep == 1:
            print(f'ep {ep:3d} loss={loss.item():.4f} val={vacc:.4f} t={time.time()-t0:.0f}s', flush=True)
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if bad >= PATIENCE:
            print(f'early stop @ ep {ep} best val {best_acc:.4f}', flush=True)
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        out = model(xs, training=False)
        pred = out[target].argmax(1).cpu()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'pred': pred}, Path(a.out))
    print(f'[done] {a.split} preds={pred.numel()} best_val={best_acc:.4f} total={time.time()-t0:.0f}s dev={DEVICE} -> {a.out}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
