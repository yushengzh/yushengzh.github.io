#!/usr/bin/env python3
import argparse, time, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class MPBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.update = nn.Linear(3*hidden, hidden)
        self.act = nn.ReLU()
    def forward(self, h, ei, evec):
        row, col = ei[0], ei[1]
        m = h[col]
        if evec is not None:
            m = m + evec
        m = self.nn(m)
        N = h.size(0)
        agg_sum = torch.zeros_like(h).index_add_(0, row, m)
        deg = torch.zeros(N, 1, device=h.device).index_add_(0, row, torch.ones(row.size(0),1,device=h.device))
        mean_agg = agg_sum / deg.clamp(min=1.0)
        h = self.update(torch.cat([h, mean_agg, agg_sum], dim=1))
        return self.act(h)

class GNN(nn.Module):
    def __init__(self, nhots, e_sizes, hidden, out_dim, layers=5):
        super().__init__()
        self.node_emb = nn.Linear(nhots, hidden)
        self.edge_embeds = nn.ModuleList([nn.Embedding(r, hidden) for r in e_sizes])
        self.blocks = nn.ModuleList([MPBlock(hidden) for _ in range(layers)])
        self.head = nn.Linear(2*hidden, out_dim)
    def node_vec(self, x):
        return torch.relu(self.node_emb(x))
    def edge_vec(self, ea):
        if ea is None:
            return None
        h = torch.zeros(ea.size(0), self.edge_embeds[0].weight.shape[1], device=ea.device)
        for i, emb in enumerate(self.edge_embeds):
            h = h + emb(ea[:, i].long().clamp(min=0, max=emb.num_embeddings-1))
        return h / len(self.edge_embeds)
    def forward(self, x, ei, ea, node_g, n_graphs):
        h = self.node_vec(x)
        evec = self.edge_vec(ea)
        for b in self.blocks:
            h = b(h, ei, evec)
        s = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, node_g, h)
        cnt = torch.bincount(node_g, minlength=n_graphs).clamp(min=1).unsqueeze(-1).float()
        pooled = torch.cat([s/cnt, s], dim=1)
        return self.head(pooled)

def get_feat_sizes(d):
    es = [3, 4, 2, 3]
    ea = d['edge_attr']
    for i in range(ea.shape[1]):
        if i < len(es):
            es[i] = max(es[i], int(ea[:,i].max().item()) + 1)
    es = es[:d['edge_attr'].shape[1]]
    nhots = int(d['x'].max().item()) + 1
    return nhots, es

def make_onehot(d, s0, s1, DEV):
    nptr = d['node_ptr']
    n0 = int(nptr[s0]); n1 = int(nptr[s1])
    X = d['x'][n0:n1].long().to(DEV)
    nh = d['onehot_dim']
    oh = torch.zeros(X.size(0), nh, device=DEV)
    oh.scatter_(1, X, 1.0)
    return oh

def fast_ap(y_pred, y_true):
    yv = y_true.cpu().numpy().astype('float64')
    pv = y_pred.cpu().numpy().astype('float64')
    from sklearn.metrics import average_precision_score
    ap_list = []
    for i in range(yv.shape[1]):
        labeled = ~np.isnan(yv[:, i])
        if labeled.sum() > 1 and (yv[labeled, i] == 1).sum() > 0 and (yv[labeled, i] == 0).sum() > 0:
            ap_list.append(average_precision_score(yv[labeled, i], pv[labeled, i]))
    return float(np.mean(ap_list)) if ap_list else 0.0

def get_batch(d, y, s0, s1, DEV):
    nptr = d['node_ptr']; eptr = d['edge_ptr']
    ng = s1 - s0
    n0 = int(nptr[s0]); n1 = int(nptr[s1])
    e0 = int(eptr[s0]); e1 = int(eptr[s1])
    x = make_onehot(d, s0, s1, DEV)
    ei = (d['edge_index'][:, e0:e1].long() - n0).to(DEV)
    ea = d['edge_attr'][e0:e1].to(DEV)
    sizes = (nptr[s0+1:s1+1]-nptr[s0:s1]).long().to(DEV)
    node_g = torch.repeat_interleave(torch.arange(ng, dtype=torch.long, device=DEV), sizes)
    yy = y[s0:s1].to(DEV)
    return x, ei, ea, node_g, ng, yy

def masked_bce(logits, y, task_w=None):
    m = ~torch.isnan(y)
    lg, yy = logits[m], y[m].float()
    if lg.numel() == 0:
        return logits.sum() * 0.0
    if task_w is not None:
        tw = task_w.expand(y.size(0), -1)[m].float()
        loss = F.binary_cross_entropy_with_logits(lg, yy, reduction='none') * tw
        return loss.mean()
    return F.binary_cross_entropy_with_logits(lg, yy)

def predict(model, d, y, s0, s1, bs, DEV, shift):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, s1-s0, bs):
            a0 = s0+i; a1 = min(s1, a0+bs)
            x, ei, ea, node_g, ng, yy = get_batch(d, y, a0, a1, DEV)
            lg = model(x, ei, ea, node_g, ng)
            if shift is not None:
                lg = lg + shift[None, :]
            outs.append(torch.sigmoid(lg).cpu())
    return torch.cat(outs, 0).float()

def train_one(d, y, tr, T, nhots, es, a, seed, task_w, prior_shift, EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GNN(nhots, es, a.hidden, T, a.layers).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    def lr_lambda(epoch): return 0.5*(1+math.cos(math.pi*epoch/EPOCHS))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = -1.0; best_model = None; best_val = -1.0
    for ep in range(EPOCHS):
        model.train(); t1 = time.time(); total = 0.0; cnt = 0
        for s0 in range(0, tr, a.bs):
            s1 = min(tr, s0+a.bs)
            x, ei, ea, node_g, ng, yy = get_batch(d, y, s0, s1, DEV)
            lg = model(x, ei, ea, node_g, ng)
            opt.zero_grad(set_to_none=True)
            loss = masked_bce(lg, yy, task_w.to(DEV))
            loss.backward(); opt.step()
            total += float(loss.detach())*ng; cnt += ng
        sched.step()
        et = time.time()-t1
        line = '  ep %d/%d loss=%.4f lr=%.5f t=%.1fs' % (ep+1, EPOCHS, total/max(1,cnt), opt.param_groups[0]['lr'], et)
        if (ep+1) % a.eval_every == 0:
            vpred = predict(model, d, y, tr, tr+a.va, a.bs, DEV, prior_shift)
            vAP = fast_ap(vpred, y[tr:tr+a.va])
            line += ' VAL=%.4f' % vAP
            if vAP > best:
                best = vAP; best_val = vAP
                best_model = {k: v.clone() for k, v in model.state_dict().items()}
            line += ' best=%.4f' % best
        print(line, flush=True)
    if best_model is not None:
        model.load_state_dict(best_model)
    model.eval()
    return model, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test', choices=['val','test'])
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--carve', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nseeds', type=int, default=3)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--shift', type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    d = torch.load(Path(a.data_dir)/'dataset.pt', map_location='cpu')
    y = d['y']
    tr = d['split_sizes']['train']; va = d['split_sizes']['val']; te = d['split_sizes']['test']
    a.va = va
    T = y.shape[1]
    d['onehot_dim'] = int(d['x'].max().item()) + 1

    yt = y[:tr]; valid = ~torch.isnan(yt); pos = (yt > 0.5)
    n_valid = valid.sum(0).float().clamp(min=1.0); n_pos = pos.sum(0).float()
    task_w = (n_valid / (n_pos + 1.0)).clamp(min=0.1, max=10.0)
    task_w = task_w / task_w.mean()
    prior = (n_pos / (n_valid + 1.0)).clamp(min=1e-4, max=1-1e-4)
    prior_shift = (a.shift * torch.log(prior / (1 - prior))).to(DEV)

    nhots, es = get_feat_sizes(d)
    nparams = sum(p.numel() for p in GNN(nhots, es, a.hidden, T, a.layers).parameters())
    print('params', nparams, 'nhots', nhots, 'hidden', a.hidden, 'layers', a.layers, 'bs', a.bs, 'nseeds', a.nseeds, 'shift', a.shift, 'DEV', DEV, flush=True)

    models = []
    for k in range(a.nseeds):
        seed = a.seed + 1000*k
        m, bv = train_one(d, y, tr, T, nhots, es, a, seed, task_w, prior_shift, a.epochs)
        print('seed %d best_val=%.4f' % (seed, bv), flush=True)
        models.append(m)

    if a.split == 'test':
        s0, s1 = tr+va, tr+va+te
    else:
        s0, s1 = tr, tr+va
    preds = []
    for m in models:
        preds.append(predict(m, d, y, s0, s1, a.bs, DEV, prior_shift))
    pred = torch.stack(preds, 0).mean(0)
    torch.save({'pred': pred}, Path(a.out))
    print('ens nseeds=%d wrote' % a.nseeds, tuple(pred.shape), '->', a.out, 'total=%.1fs' % (time.time()-t0), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
