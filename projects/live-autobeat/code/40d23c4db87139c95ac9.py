import argparse, math, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from sklearn.metrics import roc_auc_score

HIDDEN = int(os.environ.get("SOLVER_HIDDEN", 512))
LAYERS = int(os.environ.get("SOLVER_LAYERS", 6))
EPOCHS = int(os.environ.get("SOLVER_EPOCHS", 120))
LR     = float(os.environ.get("SOLVER_LR", 1e-3))
BATCH  = int(os.environ.get("SOLVER_BATCH", 2048))
WD     = float(os.environ.get("SOLVER_WD", 1e-4))
SEED   = int(os.environ.get("SOLVER_SEED", 42))
PW_CAP = float(os.environ.get("SOLVER_PW_CAP", 20.0))
ENSEMBLE_SEEDS = [int(s) for s in os.environ.get("SOLVER_ENSEMBLE_SEEDS", "42,0,7").split(",")]


class GINNet(nn.Module):
    def __init__(self, emb_sizes, edge_sizes, hidden, n_layers, n_tasks):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, hidden) for s in emb_sizes])
        self.edge_embs = nn.ModuleList([nn.Embedding(s, hidden) for s in edge_sizes])
        convs, bns = [], []
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            convs.append(GINConv(mlp, train_eps=True, eps=0.0))
            bns.append(nn.BatchNorm1d(hidden))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, n_tasks))

    def forward(self, x, edge_index, edge_attr, batch_vec, n_graphs):
        N = x.size(0)
        h = sum(e(x[:, i].long()) for i, e in enumerate(self.embs))
        E = edge_index.size(1)
        if E > 0 and edge_attr is not None:
            eh = sum(e(edge_attr[:, i].long()) for i, e in enumerate(self.edge_embs))
            src = edge_index[0]
            msg = torch.zeros(N, h.size(1), device=h.device)
            msg.index_add_(0, src, eh)
            h = h + msg
        for conv, bn in zip(self.convs, self.bns):
            h = bn(conv(h, edge_index))
            h = F.relu(h)
        out = torch.zeros(n_graphs, h.size(1), device=h.device)
        out.index_add_(0, batch_vec, h)
        cnt = torch.bincount(batch_vec, minlength=n_graphs).float().unsqueeze(1)
        mean = out / cnt.clamp(min=1.0)
        mx = torch.full((n_graphs, h.size(1)), float("-inf"), device=h.device)
        mx.scatter_reduce_(0, batch_vec.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=False)
        g = torch.cat([mean, mx], dim=1)
        return self.head(g)


def prepare(d, device):
    x = d["x"].float().to(device)
    ei = d["edge_index"].long().to(device)
    ea = d["edge_attr"].float().to(device)
    np_ = d["node_ptr"].long().to(device)
    ep_ = d["edge_ptr"].long().to(device)
    y = d["y"].float().to(device)
    return x, ei, ea, np_, ep_, y


def batch_slice(x, ei, ea, np_, ep_, gidx, device, n_graphs):
    node_start = np_[gidx]
    node_end   = np_[gidx + 1]
    node_cnt   = node_end - node_start
    edge_start = ep_[gidx]
    edge_end   = ep_[gidx + 1]
    edge_cnt   = edge_end - edge_start
    n_nodes = node_cnt.sum().item() if node_cnt.numel() else 0
    n_edges = edge_cnt.sum().item() if edge_cnt.numel() else 0
    if n_nodes > 0:
        cum = torch.cumsum(node_cnt, 0)
        base = node_start.repeat_interleave(node_cnt)
        within = torch.arange(n_nodes, device=device) - (cum - node_cnt).repeat_interleave(node_cnt)
        node_sel = base + within
    else:
        node_sel = torch.empty(0, device=device, dtype=torch.long)
    xb = x[node_sel] if n_nodes > 0 else x.new_zeros((0, x.size(1)))
    if n_edges > 0:
        cum_e = torch.cumsum(edge_cnt, 0)
        base_e = edge_start.repeat_interleave(edge_cnt)
        within_e = torch.arange(n_edges, device=device) - (cum_e - edge_cnt).repeat_interleave(edge_cnt)
        edge_sel = base_e + within_e
        eab = ea[edge_sel]
        eib = ei[:, edge_sel]
        local_map = torch.zeros(np_[gidx.max() + 1].item() + 1, dtype=torch.long, device=device)
        local_map[node_sel] = torch.arange(n_nodes, device=device, dtype=torch.long)
        eib = local_map[eib]
    else:
        eab = ea.new_zeros((0, ea.size(1)))
        eib = ei.new_zeros((2, 0))
    if n_nodes > 0:
        bv = torch.repeat_interleave(torch.arange(n_graphs, device=device, dtype=torch.long), node_cnt)
    else:
        bv = torch.empty(0, device=device, dtype=torch.long)
    return xb, eib, eab, bv


def predict(model, x, ei, ea, np_, ep_, indices, device, bs=512):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            ch = indices[i:i + bs]
            xb, eib, eab, bv = batch_slice(x, ei, ea, np_, ep_, ch, device, len(ch))
            preds.append(torch.sigmoid(model(xb, eib, eab, bv, len(ch))).cpu())
    return torch.cat(preds, dim=0)


def compute_pos_weight(ytr, n_tasks, cap):
    pw = []
    for t in range(n_tasks):
        col = ytr[:, t]; valid = col[~torch.isnan(col)]
        if len(valid) == 0:
            pw.append(1.0); continue
        p = valid.mean().item()
        if p <= 0:
            pw.append(1.0); continue
        pw.append(min((1 - p) / p, cap))
    return torch.tensor(pw, dtype=torch.float32)


def weighted_bce(logits, tgt, pos_weight):
    valid = ~torch.isnan(tgt)
    tgt2 = torch.nan_to_num(tgt, nan=0.0)
    w = valid.float() * pos_weight.unsqueeze(0)
    loss = F.binary_cross_entropy_with_logits(logits, tgt2, weight=w, reduction="sum")
    return loss / valid.float().sum().clamp(min=1.0), valid


def train_one(seed, x, ei, ea, np_, ep_, y, emb_sizes, edge_sizes, n_tasks, tr, va, val_idx, pos_weight, device, verbose=True):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = GINNet(emb_sizes, edge_sizes, HIDDEN, LAYERS, n_tasks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_batches = max(1, math.ceil(tr / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * n_batches)
    best_auc, best_state = -1.0, None
    y_train, y_val = y[:tr], y[tr:tr + va]
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); perm = torch.randperm(tr, device=device).cpu().numpy()
        nb_tot = n_batches; sizes = [tr // nb_tot + (1 if i < tr % nb_tot else 0) for i in range(nb_tot)]
        cur = 0; epoch_loss, nb = 0.0, 0
        for s in sizes:
            gidx = torch.tensor(perm[cur:cur + s], device=device, dtype=torch.long); cur += s
            xb, eib, eab, bv = batch_slice(x, ei, ea, np_, ep_, gidx, device, len(gidx))
            logits = model(xb, eib, eab, bv, len(gidx))
            tgt = y_train[gidx]
            loss, _ = weighted_bce(logits, tgt, pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            epoch_loss += loss.item(); nb += 1
        with torch.no_grad():
            model.eval(); vp = predict(model, x, ei, ea, np_, ep_, val_idx, device, 512)
            vm = ~torch.isnan(y_val.cpu()); yval_c = y_val.cpu(); aucs = []
            for t in range(n_tasks):
                m = vm[:, t]
                if m.sum() >= 2 and yval_c[m, t].unique().numel() >= 2:
                    aucs.append(roc_auc_score(yval_c[m, t].numpy(), vp[m, t].numpy()))
            val_auc = float(sum(aucs) / len(aucs)) if aucs else 0.0
        if val_auc > best_auc:
            best_auc = val_auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  ep {epoch + 1}/{EPOCHS} loss={epoch_loss / nb:.4f} val_auc={val_auc:.4f} best={best_auc:.4f} t={time.time() - t0:.0f}s", flush=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if verbose:
        print(f"  Best val AUC (seed {seed}): {best_auc:.4f}", flush=True)
    return model, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    a = ap.parse_args(); t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(Path(a.data_dir) / "dataset.pt", map_location="cpu")
    x, ei, ea, np_, ep_, y = prepare(d, device)
    n_tasks = y.shape[1]
    tr, va, te = d["split_sizes"]["train"], d["split_sizes"]["val"], d["split_sizes"]["test"]
    emb_sizes = [int(x[:, i].max().item()) + 1 for i in range(x.shape[1])]
    edge_sizes = [int(ea[:, i].max().item()) + 1 for i in range(ea.shape[1])]
    train_idx = torch.arange(0, tr)
    val_idx = torch.arange(tr, tr + va)
    test_idx = torch.arange(tr + va, tr + va + te)
    pos_weight = compute_pos_weight(y[:tr], n_tasks, PW_CAP).to(device)

    seeds = ENSEMBLE_SEEDS if len(ENSEMBLE_SEEDS) > 1 else [SEED]
    print(f"Ensemble with {len(seeds)} seeds: {seeds}", flush=True)

    all_preds = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Model {i+1}/{len(seeds)} (seed {seed}) ===", flush=True)
        model, best_auc = train_one(seed, x, ei, ea, np_, ep_, y, emb_sizes, edge_sizes, n_tasks, tr, va, val_idx, pos_weight, device, verbose=True)
        target = val_idx if a.split == "val" else test_idx
        pred = predict(model, x, ei, ea, np_, ep_, target, device, 512)
        all_preds.append(pred)
        del model
        torch.cuda.empty_cache()

    final_pred = torch.stack(all_preds, dim=0).mean(dim=0)
    torch.save({"pred": final_pred}, Path(a.out))
    print(f"\nWrote {len(final_pred)} {a.split} preds (ensemble of {len(seeds)} models) -> {a.out}  total={time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
