import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 128
LAYERS = 5
EPOCHS = 200
LR = 0.003
WD = 1e-4
PATIENCE = 25
N_SEEDS = 5
SEEDS = [42, 123, 7, 1337, 2024]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GINLayer(nn.Module):
    def __init__(self, dim, eps=1.0):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.eps = eps

    def forward(self, h, edge_attr, src, dst):
        msg = h[src] + edge_attr
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        h_new = (1.0 + self.eps) * h + agg
        h_new = self.update(h_new)
        h_new = self.norm(h + h_new)
        return h_new


class GINModel(nn.Module):
    def __init__(self, atom_dims, bond_dims, hidden=128, n_layers=5):
        super().__init__()
        self.hidden = hidden
        self.atom_embeds = nn.ModuleList()
        for d in atom_dims:
            if d > 1:
                self.atom_embeds.append(nn.Embedding(d, hidden))
            else:
                self.atom_embeds.append(None)
        self.bond_embeds = nn.ModuleList()
        for d in bond_dims:
            if d > 1:
                self.bond_embeds.append(nn.Embedding(d, hidden))
            else:
                self.bond_embeds.append(None)
        self.convs = nn.ModuleList([GINLayer(hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def embed_atoms(self, x):
        embs = []
        for c, emb in enumerate(self.atom_embeds):
            if emb is not None:
                embs.append(emb(x[:, c]))
        if not embs:
            return torch.zeros(x.size(0), self.hidden, device=x.device)
        return sum(embs)

    def embed_bonds(self, edge_attr):
        embs = []
        for c, emb in enumerate(self.bond_embeds):
            if emb is not None:
                embs.append(emb(edge_attr[:, c]))
        if not embs:
            return torch.zeros(edge_attr.size(0), self.hidden, device=edge_attr.device)
        return sum(embs)

    def forward(self, x, edge_attr, edge_index, batch, n_graphs):
        h = self.embed_atoms(x)
        bond_h = self.embed_bonds(edge_attr)
        src = edge_index[1]
        dst = edge_index[0]
        for conv in self.convs:
            h = conv(h, bond_h, src, dst)
        pooled_mean = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_mean / counts
        pooled_max = torch.full((n_graphs, h.size(1)), -1e9, device=h.device)
        pooled_max.scatter_reduce_(0, batch.unsqueeze(-1).expand(-1, h.size(1)), h, reduce="amax", include_self=False)
        pooled_sum = torch.zeros(n_graphs, h.size(1), device=h.device).index_add_(0, batch, h)
        pooled = torch.cat([pooled_mean, pooled_max, pooled_sum], dim=-1)
        return self.head(pooled)


def make_batch(d, indices):
    atom_parts = []
    edge_parts = []
    bond_parts = []
    batch_parts = []
    for gi in range(len(indices)):
        orig_i = indices[gi]
        n0 = int(d["node_ptr"][orig_i])
        n1 = int(d["node_ptr"][orig_i + 1])
        e0 = int(d["edge_ptr"][orig_i])
        e1 = int(d["edge_ptr"][orig_i + 1])
        n_nodes = n1 - n0
        n_edges = e1 - e0
        if n_nodes > 0:
            atom_parts.append(d["x"][n0:n1].long())
            batch_parts.append(torch.full((n_nodes,), gi, dtype=torch.long))
        if n_edges > 0:
            ei = d["edge_index"][:, e0:e1].long()
            edge_parts.append(ei)
            bond_parts.append(d["edge_attr"][e0:e1].long())
    if atom_parts:
        x = torch.cat(atom_parts, dim=0)
        batch = torch.cat(batch_parts, dim=0)
    else:
        x = torch.zeros(0, 9, dtype=torch.long)
        batch = torch.zeros(0, dtype=torch.long)
    if edge_parts:
        offsets = []
        cur = 0
        for gi in range(len(indices)):
            orig_i = indices[gi]
            n0 = int(d["node_ptr"][orig_i])
            n1 = int(d["node_ptr"][orig_i + 1])
            offsets.append((cur, n0))
            cur += (n1 - n0)
        all_ei = []
        for gi in range(len(indices)):
            orig_i = indices[gi]
            e0 = int(d["edge_ptr"][orig_i])
            e1 = int(d["edge_ptr"][orig_i + 1])
            n_edges = e1 - e0
            if n_edges > 0:
                ei = d["edge_index"][:, e0:e1].long()
                local_start, global_start = offsets[gi]
                ei = ei - global_start + local_start
                all_ei.append(ei)
        if all_ei:
            edge_index = torch.cat(all_ei, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    if bond_parts:
        edge_attr = torch.cat(bond_parts, dim=0)
    else:
        edge_attr = torch.zeros(0, 3, dtype=torch.long)
    return x, edge_attr, edge_index, batch


def train_one(seed, atom_dims, bond_dims, train_x, train_ea, train_ei, train_batch,
              val_x, val_ea, val_ei, val_batch, y_train, y_val, n_train, n_val):
    """Train one GIN model with the given seed; return best-state model (eval mode)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GINModel(atom_dims, bond_dims, HIDDEN, LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [seed {seed}] n_params={n_params}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_rmse = float("inf")
    best_model_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x, train_ea, train_ei, train_batch, n_train).squeeze(-1)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x, val_ea, val_ei, val_batch, n_val).squeeze(-1)
            val_rmse = float(torch.sqrt(F.mse_loss(val_pred, y_val.to(DEVICE))))

        if (epoch + 1) % 50 == 0 or epoch < 3:
            print(f"  [seed {seed}] epoch {epoch+1}/{EPOCHS} loss={float(loss):.4f} val_rmse={val_rmse:.4f} t={time.time()-t0:.1f}s", flush=True)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [seed {seed}] early stop at epoch {epoch+1}, best val_rmse={best_val_rmse:.4f}", flush=True)
                break

    model.load_state_dict(best_model_state)
    model.eval()
    print(f"  [seed {seed}] done, best val_rmse={best_val_rmse:.4f} after {epoch+1} epochs, t={time.time()-t0:.1f}s", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    d = torch.load(Path(args.data_dir) / "dataset.pt", map_location="cpu")
    n_train = d["split_sizes"]["train"]
    n_val = d["split_sizes"]["val"]
    n_test = d["split_sizes"]["test"]
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_train + n_val + n_test))

    atom_dims = []
    for c in range(d["x"].size(1)):
        mx = int(d["x"][:, c].max().item()) + 1
        atom_dims.append(mx)
    bond_dims = []
    for c in range(d["edge_attr"].size(1)):
        mx = int(d["edge_attr"][:, c].max().item()) + 1
        bond_dims.append(mx)
    print(f"Atom dims: {atom_dims}", flush=True)
    print(f"Bond dims: {bond_dims}", flush=True)
    print(f"Splits: train={n_train} val={n_val} test={n_test}", flush=True)

    train_x, train_ea, train_ei, train_batch = make_batch(d, train_idx)
    train_x, train_ea, train_ei, train_batch = (t.to(DEVICE) for t in (train_x, train_ea, train_ei, train_batch))
    val_x, val_ea, val_ei, val_batch = make_batch(d, val_idx)
    val_x, val_ea, val_ei, val_batch = (t.to(DEVICE) for t in (val_x, val_ea, val_ei, val_batch))
    test_x, test_ea, test_ei, test_batch = make_batch(d, test_idx)
    test_x, test_ea, test_ei, test_batch = (t.to(DEVICE) for t in (test_x, test_ea, test_ei, test_batch))

    y_train = d["y"][:n_train].squeeze(-1).to(DEVICE)
    y_val = d["y"][n_train:n_train + n_val].squeeze(-1)

    t_all = time.time()
    preds = []
    for seed in SEEDS:
        model = train_one(seed, atom_dims, bond_dims,
                          train_x, train_ea, train_ei, train_batch,
                          val_x, val_ea, val_ei, val_batch,
                          y_train, y_val, n_train, n_val)
        with torch.no_grad():
            if args.split == "val":
                p = model(val_x, val_ea, val_ei, val_batch, n_val)
            else:
                p = model(test_x, test_ea, test_ei, test_batch, n_test)
        preds.append(p.cpu())
        del model
        torch.cuda.empty_cache()

    mean_pred = torch.stack(preds, dim=0).mean(dim=0).contiguous()
    torch.save({"pred": mean_pred}, Path(args.out))
    print(f"Ensemble of {len(preds)} seeds. Wrote predictions for {args.split} ({mean_pred.shape}) -> {args.out}", flush=True)
    print(f"Total time: {time.time()-t_all:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
