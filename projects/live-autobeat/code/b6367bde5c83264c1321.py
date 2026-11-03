
"""Toxicity prediction on ogbg-moltox21.

Command-line contract:
    python solve.py --data-dir <DATA> --out <OUT.pt> [--split {val,test}]
Output: dict with key 'pred' -> [M, T] float tensor (one score per task per graph).
Metric: mean per-task ROC-AUC on the scored split (higher is better).

Architecture: GATv2 conv layers with per-categorical node + edge embeddings.
Node input = sum over the 9 OGB atom one-hot fields of learned embeddings.
Edges: 3 bond categorical fields embedded, summed to a compact edge vector that
enters both the GATv2 attention score and the message.
"""
import argparse
import os
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tunables (overridable via env vars)
# ---------------------------------------------------------------------------
SEEDS    = [int(s) for s in os.environ.get("SOLVER_SEEDS", "0,1,2").split(",")]
EPOCHS   = int(os.environ.get("SOLVER_EPOCHS", "60"))
HIDDEN   = int(os.environ.get("SOLVER_HIDDEN", "128"))
LAYERS   = int(os.environ.get("SOLVER_LAYERS", "3"))
BATCH    = int(os.environ.get("SOLVER_BATCH", "128"))
LR       = float(os.environ.get("SOLVER_LR", "3e-4"))
WD       = float(os.environ.get("SOLVER_WD", "5e-4"))
DROPOUT  = float(os.environ.get("SOLVER_DROPOUT", "0.1"))
PATIENCE = int(os.environ.get("SOLVER_PATIENCE", "20"))
NUM_HEAD = int(os.environ.get("SOLVER_HEADS", "8"))
EDGE_DIM = int(os.environ.get("SOLVER_EDGE_DIM", "16"))


def pick_device():
    if torch.cuda.is_available():
        torch.cuda.init()
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Graph helpers (flat, per-graph offsets)
# ---------------------------------------------------------------------------
def get_cardinals(d):
    node_card = [int(d["x"][:, i].max().item()) + 1 for i in range(d["x"].shape[1])]
    edge_card = [int(d["edge_attr"][:, i].max().item()) + 1 for i in range(d["edge_attr"].shape[1])]
    return node_card, edge_card


def collate_batch(d, indices):
    """Collate graph indices into flat tensors with global (offset) node indices.
    Returns x [N,9], edge_index [2,E] (global), edge_attr [E,3], batch [N]."""
    xs, eis, eas, batches = [], [], [], []
    offset = 0
    for bi, gi in enumerate(indices):
        n0, n1 = int(d["node_ptr"][gi]), int(d["node_ptr"][gi + 1])
        e0, e1 = int(d["edge_ptr"][gi]), int(d["edge_ptr"][gi + 1])
        x = d["x"][n0:n1].long()
        ei = d["edge_index"][:, e0:e1].long() - n0          # global -> batch-local
        ea = d["edge_attr"][e0:e1].long()
        n = x.size(0)
        xs.append(x)
        eis.append(ei)
        eas.append(ea)
        batches.append(torch.full((n,), bi, dtype=torch.long))
        offset += n
    if xs:
        x = torch.cat(xs, 0)
        ea = torch.cat(eas, 0) if eas else torch.zeros(0, d["edge_attr"].shape[1], dtype=torch.long)
        if len(eis) > 1:
            ei = torch.cat(eis, 1)
        elif len(eis) == 1 and eis[0].numel() > 0:
            ei = eis[0]
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        b = torch.cat(batches)
    else:
        x = torch.zeros(0, d["x"].shape[1], dtype=torch.long)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ea = torch.zeros(0, d["edge_attr"].shape[1], dtype=torch.long)
        b = torch.zeros(0, dtype=torch.long)
    return x, ei, ea, b


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GATv2Conv(nn.Module):
    """GATv2 layer: attention in a linear space over src/dst/edge features.
    score_ij = LeakyReLU(a_src h_i + a_dst h_j + a_edge e_ij)
    h'_i = sum_j softmax(score_ij) * (W h_j + W_e e_ij)   (+ residual, LayerNorm)"""
    def __init__(self, hidden, heads, dropout, edge_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = max(hidden // heads, 1)
        self.W = nn.Linear(hidden, self.head_dim * self.heads, bias=False)
        self.We = nn.Linear(edge_dim, self.head_dim, bias=False)
        self.a_src = nn.Linear(self.head_dim, 1, bias=False)
        self.a_dst = nn.Linear(self.head_dim, 1, bias=False)
        self.a_edge = nn.Linear(edge_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(self.head_dim * heads, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, e):
        N = x.size(0)
        E = edge_index.size(1)
        h = self.W(x)                                  # [N, Hd]
        h = h.view(N, self.heads, self.head_dim)
        if E == 0:
            out = self.out_proj(h.view(N, -1))
            return x + self.dropout(self.norm(out))
        src, dst = edge_index[0], edge_index[1]
        hs = h[src]                                    # [E, heads, Hd]
        hd = h[dst]                                    # [E, heads, Hd]
        # edge feature per head
        ee = self.We(e)                                # [E, Hd]
        ee = ee.unsqueeze(1).expand(-1, self.heads, -1) # [E, heads, Hd]
        a = (self.a_src(self.leaky(hs)) +
             self.a_dst(self.leaky(hd)) +
             self.a_edge(self.leaky(ee)))              # [E, heads, 1]
        a = a.squeeze(-1)                              # [E, heads]
        a = a - a.max(dim=0, keepdim=True).values       # numerical safety
        a = torch.softmax(a, dim=0)                    # per-head, over incoming edges
        a = self.dropout(a)                            # [E, heads]
        msg = hs * a.unsqueeze(-1) + ee * a.unsqueeze(-1)
        agg = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        agg.index_add_(0, dst, msg)
        agg = agg.view(N, -1)
        return x + self.dropout(self.norm(self.out_proj(agg)))


class GNN(nn.Module):
    def __init__(self, node_card, edge_card, out_dim, hidden=128, layers=3, heads=8, dropout=0.1, edge_dim=16):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(c, hidden) for c in node_card])
        self.edge_emb = nn.ModuleList([nn.Embedding(c, edge_dim) for c in edge_card])
        self.convs = nn.ModuleList([GATv2Conv(hidden, heads, dropout, edge_dim) for _ in range(layers)])
        self.readout_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        # node input: sum over categorical fields
        h = self.embed[0](x[:, 0])
        for i in range(1, x.size(1)):
            h = h + self.embed[i](x[:, i])
        # edge features: sum over categorical fields -> [E, edge_dim]
        ee = self.edge_emb[0](edge_attr[:, 0])
        for i in range(1, edge_attr.size(1)):
            ee = ee + self.edge_emb[i](edge_attr[:, i])
        if edge_attr.size(0) == 0:
            ee = ee[:0]
        for conv in self.convs:
            h = conv(h, edge_index, ee)
        # graph-level readout: sum + mean + max, then combine
        n = batch.max().item() + 1 if batch.numel() > 0 else 0
        H = h.size(1)
        s = torch.zeros(n, H, device=h.device)
        cnt = torch.zeros(n, device=h.device)
        mx = torch.full((n, H), float('-inf'), device=h.device)
        if batch.numel() > 0:
            s.index_add_(0, batch, h)
            cnt = torch.bincount(batch, minlength=n).float()
            mx.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce='amax')
        cntc = cnt.clamp(min=1).unsqueeze(-1)
        m = s / cntc
        mx = mx.masked_fill(mx.isinf(), 0.0)
        pooled = s + m + mx                            # [n, H]
        return self.readout_head(pooled)


# ---------------------------------------------------------------------------
# Eval / AUC
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score


def eval_auc(model, d, indices, node_card, edge_card, device, return_preds=False):
    model.eval()
    preds = []
    bs = 256
    with torch.no_grad():
        for s in range(0, len(indices), bs):
            idx = indices[s:s + bs]
            x, ei, ea, b = collate_batch(d, idx)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); b = b.to(device)
            logits = model(x, ei, ea, b)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            preds.append(torch.sigmoid(logits).cpu())
    preds = torch.cat(preds, 0)
    if not return_preds:
        y = d["y"][indices].numpy()
        aucs = []
        for t in range(y.shape[1]):
            mask = ~np.isnan(y[:, t])
            if mask.sum() < 2 or mask.sum() == len(indices):
                continue
            try:
                aucs.append(roc_auc_score(y[mask, t], preds[mask, t].numpy()))
            except Exception:
                pass
        return float(np.mean(aucs)) if aucs else 0.5
    return preds


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------
def train_one(d, node_card, edge_card, seed, device, train_indices, val_indices):
    set_seed(seed)
    n_tasks = d["y"].shape[1]
    model = GNN(node_card, edge_card, n_tasks, HIDDEN, LAYERS, NUM_HEAD, DROPOUT, EDGE_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    params={n_params/1e6:.2f}M hidden={HIDDEN} layers={LAYERS} heads={NUM_HEAD}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # per-task positive weight from train positive rates
    ytr = d["y"][:len(train_indices)].float()
    pos_rate = []
    for t in range(n_tasks):
        mask = ~torch.isnan(ytr[:, t])
        r = ytr[:, t][mask].mean().item() if mask.any() else 0.5
        pos_rate.append(r)
    pos_rate = torch.tensor(pos_rate)
    w = torch.sqrt(pos_rate / (1 - pos_rate).clamp(min=1e-4)) * 2.0
    w = w.clamp(max=5.0).to(device)
    print(f"    task pos_rate={[round(r,3) for r in pos_rate.tolist()]}", flush=True)

    warmup = 5
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_auc = 0.0
    best_state = None
    bad = 0
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_indices))
        total_loss = 0.0; n_b = 0
        for s in range(0, len(perm), BATCH):
            sel = perm[s:s + BATCH]
            gi = [int(train_indices[i]) for i in sel]
            x, ei, ea, b = collate_batch(d, gi)
            yb = d["y"][gi].float().to(device)
            x = x.to(device); ei = ei.to(device); ea = ea.to(device); b = b.to(device)
            opt.zero_grad()
            logits = model(x, ei, ea, b)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            mask = ~torch.isnan(yb)
            if mask.any():
                yclip = yb.masked_fill(~mask, 0.0)
                bce = F.binary_cross_entropy_with_logits(logits, yclip, reduction='none')
                bce = bce * w.unsqueeze(0)
                loss = (bce * mask.float()).sum() / mask.float().sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += loss.item(); n_b += 1
        sched.step()
        val_auc = eval_auc(model, d, val_indices, node_card, edge_card, device)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1}/{EPOCHS} loss={total_loss/max(n_b,1):.4f} val_auc={val_auc:.4f} best={best_auc:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"    final best val_auc={best_auc:.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    t0 = time.time()
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    sizes = d["split_sizes"]
    node_card, edge_card = get_cardinals(d)
    print(f"  node cardinals: {node_card}\n  edge cardinals: {edge_card}", flush=True)
    device = pick_device()
    print(f"  device: {device}", flush=True)
    train_indices = list(range(sizes["train"]))
    val_indices = list(range(sizes["train"], sizes["train"] + sizes["val"]))
    target = (val_indices if a.split == "val"
              else list(range(sizes["train"] + sizes["val"], d["num_graphs"])))
    all_preds = []
    for seed in SEEDS:
        print(f"\nTraining seed {seed}...", flush=True)
        ts = time.time()
        model = train_one(d, node_card, edge_card, seed, device, train_indices, val_indices)
        print(f"  seed {seed} done in {time.time()-ts:.1f}s", flush=True)
        pred = eval_auc(model, d, target, node_card, edge_card, device, return_preds=True)
        all_preds.append(pred)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    avg = torch.stack(all_preds).mean(dim=0)
    avg = torch.nan_to_num(avg, nan=0.5, posinf=1.0, neginf=0.0)
    torch.save({"pred": avg}, Path(a.out))
    print(f"\nWrote {avg.shape} predictions to {a.out}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
