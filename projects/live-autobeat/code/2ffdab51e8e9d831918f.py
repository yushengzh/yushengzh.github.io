#!/usr/bin/env python3
import argparse, time, sys, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class MPBlock(nn.Module):
    def __init__(self, hidden, e_feat):
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Linear(2*hidden, hidden)
        self.act = nn.ReLU()
        self.use_e = e_feat > 0
        if self.use_e:
            self.edge_mlp = nn.Sequential(nn.Linear(e_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, h, ei, ea):
        row, col = ei[0], ei[1]
        m = h[col]
        if self.use_e and ea is not None:
            m = m + self.edge_mlp(ea)
        m = self.nn(m)
        agg = torch.zeros_like(h).index_add_(0, row, m)
        deg = torch.zeros(h.size(0), 1, device=h.device).index_add_(0, row, torch.ones(row.size(0),1,device=h.device))
        mean_agg = agg / deg.clamp(min=1.0)
        h = self.update(torch.cat([h, mean_agg], dim=1))
        return self.act(h)

class GNN(nn.Module):
    def __init__(self, feat_ranges, e_feat, hidden, out_dim, layers=3):
        super().__init__()
        self.nfeat = len(feat_ranges)
        self.embeds = nn.ModuleList([nn.Embedding(int(r+1), hidden) for r in feat_ranges])
        self.blocks = nn.ModuleList([MPBlock(hidden, e_feat) for _ in range(layers)])
        self.head = nn.Linear(3*hidden, out_dim)

    def embed(self, x):
        parts = [self.embeds[i](x[:, i].long().clamp(min=0, max=self.embeds[i].num_embeddings-1)) for i in range(self.nfeat)]
        return torch.stack(parts, dim=0).mean(0)

    def forward(self, x, ei, ea, node_g, n_graphs):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h, ei, ea)
        N = h.size(1)
        s = torch.zeros(n_graphs, N, device=h.device).index_add_(0, node_g, h)
        mx = torch.full((n_graphs, N), -1e9, device=h.device)
        mx.scatter_reduce_(0, node_g.unsqueeze(1).expand(-1, N), h, reduce='amax', include_self=False)
        cnt = torch.bincount(node_g, minlength=n_graphs).clamp(min=1).unsqueeze(-1).float()
        pooled = torch.cat([s/cnt, mx, s], dim=1)
        return self.head(pooled)

def masked_bce(logits, y):
    m = ~torch.isnan(y)
    lg, yy = logits[m], y[m].float()
    if lg.numel() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(lg, yy)

def get_batch(d, y, s0, s1, DEV):
    nptr = d['node_ptr']; eptr = d['edge_ptr']
    ng = s1 - s0
    n0 = int(nptr[s0]); n1 = int(nptr[s1])
    e0 = int(eptr[s0]); e1 = int(eptr[s1])
    x = d['x'][n0:n1].to(DEV)
    ei = (d['edge_index'][:, e0:e1].long() - n0).to(DEV)
    e_feat = d['edge_attr'].shape[1] if d['edge_attr'] is not None else 0
    ea = d['edge_attr'][e0:e1].to(DEV) if e_feat else None
    sizes = (nptr[s0+1:s1+1]-nptr[s0:s1]).long().to(DEV)
    node_g = torch.repeat_interleave(torch.arange(ng, dtype=torch.long, device=DEV), sizes)
    yy = y[s0:s1].to(DEV)
    return x, ei, ea, node_g, ng, yy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test', choices=['val','test'])
    a = ap.parse_args()
    t0 = time.time()
    d = torch.load(Path(a.data_dir)/'dataset.pt', map_location='cpu')
    s = d['split_sizes']; tr, va, te = s['train'], s['val'], s['test']
    y = d['y'].float()
    out_dim = int(y.size(1))
    feat_ranges = [int(d['x'][:, i].max()) for i in range(d['x'].shape[1])]
    e_feat = int(d['edge_attr'].shape[1]) if d['edge_attr'] is not None else 0
    print('loaded %.1fs feat_ranges=%s e_feat=%d' % (time.time()-t0, feat_ranges, e_feat), flush=True)
    hidden = 128
    model = GNN(feat_ranges, e_feat, hidden, out_dim, layers=3).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print('params', npar, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = 15; BS = 4096
    total_steps = EPOCHS * math.ceil(tr / BS)
    for ep in range(EPOCHS):
        model.train(); t1 = time.time(); total = 0.0; cnt = 0
        for s0 in range(0, tr, BS):
            s1 = min(tr, s0+BS)
            x, ei, ea, node_g, ng, yy = get_batch(d, y, s0, s1, DEV)
            opt.zero_grad(set_to_none=True)
            loss = masked_bce(model(x, ei, ea, node_g, ng), yy)
            loss.backward(); opt.step()
            total += float(loss)*ng; cnt += ng
        print('epoch %d/%d train_loss=%.4f t=%.1fs' % (ep+1, EPOCHS, total/max(1,cnt), time.time()-t1), flush=True)
    start = tr+va if a.split == 'test' else tr
    end = tr+va+te if a.split == 'test' else tr+va
    model.eval(); outs = []; B = 8192
    with torch.no_grad():
        for i in range(0, end-start, B):
            s0 = start+i; s1 = min(end, s0+B)
            x, ei, ea, node_g, ng, yy = get_batch(d, y, s0, s1, DEV)
            outs.append(torch.sigmoid(model(x, ei, ea, node_g, ng)).cpu())
    pred = torch.cat(outs, 0).float()
    torch.save({'pred': pred}, Path(a.out))
    print('wrote', tuple(pred.shape), '->', a.out, 'total=%.1fs' % (time.time()-t0), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
