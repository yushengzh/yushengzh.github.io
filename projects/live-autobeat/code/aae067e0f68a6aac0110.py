#!/usr/bin/env python3
import argparse, os, time, copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

def _env(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    return type(default)(raw)

HIDDEN = _env('SOLVER_HIDDEN', 512)
L1     = _env('SOLVER_L1', 1024)
L2     = _env('SOLVER_L2', 1024)
EPOCHS = _env('SOLVER_EPOCHS', 80)
LR     = _env('SOLVER_LR', 3e-3)
WD     = _env('SOLVER_WD', 1e-4)
DROPOUT = _env('SOLVER_DROPOUT', 0.3)
SEED   = _env('SOLVER_SEED', 0)
PATIENCE = _env('SOLVER_PATIENCE', 15)
BATCH  = _env('SOLVER_BATCH', 65536)
DEVICE = _env('SOLVER_DEVICE', 'cuda')

class MLP(nn.Module):
    def __init__(self, in_f, h1, h2, C, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h2, C),
        )
    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    a = ap.parse_args()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    t0 = time.time()
    d = torch.load(Path(a.data_dir) / 'dataset.pt', map_location='cpu')
    C = int(d['num_classes']); n = int(d['x'].shape[0])
    y = d['y'].long()
    target = d['val_idx'] if a.split == 'val' else d['test_idx']
    x = d['x'].float()
    if a.split == 'val':
        mask = (y >= 0) & (y < C)
        train_idx = torch.nonzero(mask, as_tuple=True)[0]
    else:
        train_idx = d['train_idx']
    va = d['val_idx']
    xtr = x[train_idx].to(DEVICE); ytr = y[train_idx].to(DEVICE)
    xva = x[va].to(DEVICE); yva = y[va].to(DEVICE)
    model = MLP(x.size(1), L1, L2, C, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=xtr.device)
        tot_loss = 0.0; nb=0
        for s in range(0, len(xtr), BATCH):
            b = perm[s:s+BATCH]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xtr[b]), ytr[b])
            loss.backward(); opt.step(); tot_loss += loss.item(); nb+=1
        sched.step()
        model.eval()
        with torch.no_grad():
            vacc = float((model(xva).argmax(1) == yva).float().mean().item())
        if ep % 5 == 0 or ep == 1:
            mem = torch.cuda.max_memory_allocated(DEVICE)/1e9 if DEVICE.startswith('cuda') else 0.0
            print(f'ep {ep:3d} loss={tot_loss/max(1,nb):.4f} val={vacc:.4f} t={time.time()-t0:.0f}s mem={mem:.1f}GB', flush=True)
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if bad >= PATIENCE:
            print(f'early stop @ ep {ep} best {best_acc:.4f}', flush=True); break
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    xtg = x[target].to(DEVICE)
    with torch.no_grad():
        pred = model(xtg).argmax(1).cpu()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pred": pred}, Path(a.out))
    print(f'[done] mlp preds={pred.shape} best_val={best_acc:.4f} total={time.time()-t0:.0f}s -> {a.out}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
