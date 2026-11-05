import argparse, math, os, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 128
LAYERS = 5
HEAD_HIDDEN = 64
BATCH = 500
EPOCHS = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42

ATOM_VOCAB = [120, 4, 7, 12, 8, 5, 6, 2, 2]
EDGE_VOCAB = [4, 3, 2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


def build_tensors(x_np, ei_np, ea_np, node_ptr, edge_ptr, g0, g1):
    """Build per-split tensors with local node ids and precomputed offsets for fast collate."""
    n0 = int(node_ptr[g0]); n1 = int(node_ptr[g1])
    e0 = int(edge_ptr[g0]); e1 = int(edge_ptr[g1])
    B = g1 - g0
    n_per = (node_ptr[g0+1:g1+1] - node_ptr[g0:g1]).astype(np.int64)
    e_per = (edge_ptr[g0+1:g1+1] - edge_ptr[g0:g1]).astype(np.int64)
    node_batch = np.repeat(np.arange(B, dtype=np.int64), n_per)
    edge_batch = np.repeat(np.arange(B, dtype=np.int64), e_per)
    x = x_np[n0:n1].astype(np.float32)
    ea = ea_np[e0:e1].astype(np.float32)
    ei = ei_np[:, e0:e1].copy().astype(np.int64)
    # Localize: subtract global node_ptr for each graph
    starts = node_ptr[g0:g1].astype(np.int64)
    # Localize node indices in edges: subtract the global start of each graph
    e_graph = np.repeat(np.arange(B, dtype=np.int64), e_per)
    e_base = starts[e_graph]
    ei[0] = ei[0] - e_base
    ei[1] = ei[1] - e_base
    # Precompute offsets for fast collate
    node_off = np.zeros(B + 1, dtype=np.int64)
    node_off[1:] = np.cumsum(n_per)
    edge_off = np.zeros(B + 1, dtype=np.int64)
    edge_off[1:] = np.cumsum(e_per)
    return {
        "x": torch.from_numpy(x),
        "ei": torch.from_numpy(ei).long(),
        "ea": torch.from_numpy(ea),
        "node_batch": torch.from_numpy(node_batch).long(),
        "edge_batch": torch.from_numpy(edge_batch).long(),
        "node_off": node_off,
        "edge_off": edge_off,
    }


def collate_fast(T, idx, device):
    """Fast collate using precomputed per-graph offsets. idx: numpy array of local graph ids."""
    x, ei, ea = T["x"], T["ei"], T["ea"]
    noff, eoff = T["node_off"], T["edge_off"]
    n_parts = []
    ei_parts = []
    ea_parts = []
    batch_parts = []
    edge_batch_parts = []
    for i, g in enumerate(idx):
        gs, ge = int(noff[g]), int(noff[g+1])
        if ge > gs:
            n_parts.append(x[gs:ge])
            batch_parts.append(torch.full((ge - gs,), i, dtype=torch.long))
        es, ee = int(eoff[g]), int(eoff[g+1])
        if ee > es:
            ei_parts.append(ei[:, es:ee])
            ea_parts.append(ea[es:ee])
            edge_batch_parts.append(torch.full((ee - es,), i, dtype=torch.long))
    if n_parts:
        x_out = torch.cat(n_parts, 0)
        batch = torch.cat(batch_parts, 0)
    else:
        x_out = torch.zeros(0); batch = torch.zeros(0, dtype=torch.long)
    if ei_parts:
        ei_out = torch.cat(ei_parts, 1)
        ea_out = torch.cat(ea_parts, 0)
        edge_batch = torch.cat(edge_batch_parts, 0)
    else:
        ei_out = torch.zeros(2, 0, dtype=torch.long)
        ea_out = torch.zeros(0, T["ea"].shape[1])
        edge_batch = torch.zeros(0, dtype=torch.long)
    if device != "cpu":
        x_out = x_out.to(device); ei_out = ei_out.to(device)
        ea_out = ea_out.to(device); batch = batch.to(device); edge_batch = edge_batch.to(device)
    return x_out, ei_out, ea_out, batch


class GINConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_bond = nn.Embedding(EDGE_VOCAB[0], hidden)
        self.edge_stereo = nn.Embedding(EDGE_VOCAB[1], hidden)
        self.edge_ring = nn.Embedding(EDGE_VOCAB[2], hidden)

    def forward(self, x, edge_index, edge_attr):
        b = edge_index[0]; c = edge_index[1]
        ea = edge_attr.long()
        eh = self.edge_bond(ea[:, 0]) + self.edge_stereo(ea[:, 1]) + self.edge_ring(ea[:, 2])
        msg = x[c] + eh
        agg = torch.zeros_like(x)
        agg.index_add_(0, b, msg)
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, b, torch.ones(b.size(0), device=x.device))
        agg = agg / deg.clamp(min=1.0).unsqueeze(1)
        return self.mlp((1.0 + self.eps) * x + agg)


class GIN(nn.Module):
    def __init__(self, hidden, layers, out_dim, head_hidden):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(v, hidden) for v in ATOM_VOCAB])
        self.input_proj = nn.Linear(hidden * len(ATOM_VOCAB), hidden)
        self.convs = nn.ModuleList([GINConv(hidden) for _ in range(layers)])
        self.pool_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, out_dim))

    def forward(self, x, edge_index, batch, edge_attr):
        h = torch.cat([e(x[:, i].long()) for i, e in enumerate(self.emb)], dim=1)
        h = self.input_proj(h)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        out = torch.zeros(batch.max().item() + 1, h.size(1), device=h.device)
        out.index_add_(0, batch, h)
        out = self.pool_norm(out)
        return self.head(out)


def predict_batched(model, T, device, bs=500):
    model.eval()
    B = T["node_batch"].max().item() + 1
    all_preds = []
    with torch.no_grad():
        for i in range(0, B, bs):
            idx = np.arange(i, min(i + bs, B), dtype=np.int64)
            x, ei, ea, batch = collate_fast(T, idx, device)
            logits = model(x, ei, batch, ea)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0)


def val_ap(model, Tva, yva_np, device):
    vp = predict_batched(model, Tva, device, 500).numpy()
    from sklearn.metrics import average_precision_score
    yv = yva_np
    valid = ~np.isnan(yv); aps = []
    for t in range(yv.shape[1]):
        m = valid[:, t]
        if m.sum() < 2: continue
        yt = yv[m, t].astype(int)
        if len(set(yt.tolist())) < 2: continue
        aps.append(average_precision_score(yt, vp[m, t]))
    return float(np.mean(aps)) if aps else 0.0


def main():
    args = parse_args()
    t0 = time.time()
    data = torch.load(Path(args.data_dir) / "dataset.pt", map_location="cpu")
    x_all = data["x"].numpy()
    ei_all = data["edge_index"].numpy()
    ea_all = data["edge_attr"].numpy()
    node_ptr = data["node_ptr"].numpy()
    edge_ptr = data["edge_ptr"].numpy()
    y_all = data["y"].numpy()
    split_sizes = data["split_sizes"]
    tr = int(split_sizes["train"]); va = int(split_sizes["val"]); te = int(split_sizes["test"])
    out_dim = y_all.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"G={tr+va+te} tr={tr} va={va} te={te} T={out_dim} N={x_all.shape[0]} E={ei_all.shape[1]}", flush=True)
    print(f"device={device}", flush=True)

    print("Building train tensors...", flush=True)
    Ttr = build_tensors(x_all, ei_all, ea_all, node_ptr, edge_ptr, 0, tr)
    print("Building val tensors...", flush=True)
    Tva = build_tensors(x_all, ei_all, ea_all, node_ptr, edge_ptr, tr, tr + va)
    if args.split == "test":
        print("Building test tensors...", flush=True)
        Tte = build_tensors(x_all, ei_all, ea_all, node_ptr, edge_ptr, tr + va, tr + va + te)

    ytr = torch.from_numpy(y_all[:tr]).float().to(device)
    yva = torch.from_numpy(y_all[tr:tr + va]).float().to(device)
    yva_np = y_all[tr:tr + va].copy()

    mask = ~torch.isnan(ytr)
    pos = (ytr > 0.5) & mask
    n_pos = pos.float().sum(0)
    n_neg = mask.float().sum(0) - n_pos
    pos_weight = (n_neg / n_pos.clamp(min=1)).sqrt().clamp(max=10.0).to(device)
    print(f"pos_weight sqrt min/med/max: {pos_weight.min().item():.2f}/{pos_weight.median().item():.2f}/{pos_weight.max().item():.2f}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = GIN(HIDDEN, LAYERS, out_dim, HEAD_HIDDEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(tr // BATCH, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)

    best_val_ap = -1.0; patience = 0; best_state = None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(tr, generator=torch.Generator().manual_seed(SEED + epoch))
        ep_loss = 0.0; ep_n = 0; n_batches = 0
        for i in range(0, tr, BATCH):
            bidx = perm[i:i + BATCH]
            idx = bidx.cpu().numpy()
            x, ei, ea, batch = collate_fast(Ttr, idx, device)
            yb = ytr[bidx].clone().nan_to_num(0.0)
            mbl = ~torch.isnan(ytr[bidx])
            logits = model(x, ei, batch, ea)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            mb = mbl.to(logits.dtype)
            loss = (loss * mb).sum() / mb.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            if n_batches == 0:
                print(f"GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
            ep_loss += loss.item() * mb.sum().item(); ep_n += mb.sum().item()
            n_batches += 1
        avg_loss = ep_loss / max(ep_n, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} t={time.time()-t0:.0f}s batches={n_batches}", flush=True)
        if epoch > 1 and (epoch + 1) % 2 == 0:
            vap = val_ap(model, Tva, yva_np, device)
            print(f"  val AP={vap:.4f}", flush=True)
            if vap > best_val_ap + 1e-4:
                best_val_ap = vap; patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 10 and epoch > 3:
                    print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if args.split == "val":
        target = predict_batched(model, Tva, device, 500).cpu().numpy()
    else:
        target = predict_batched(model, Tte, device, 500).cpu().numpy()
    torch.save({"pred": torch.from_numpy(target)}, Path(args.out))
    print(f"Wrote {len(target)} preds -> {args.out} (t={time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
