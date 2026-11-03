from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    t0 = time.time()
    d = torch.load(data_dir / "dataset.pt")
    x = d["x"].numpy().astype(np.float64)
    if a.split == "val":
        VP = np.concatenate([d["val_edges"].numpy(), d["val_neg"].numpy()], axis=1)
    else:
        VP = torch.load(data_dir / "test_inputs.pt")["candidates"].long().numpy()
    a_ = x[VP[0]]; b_ = x[VP[1]]
    diff = a_ - b_
    dist = np.sqrt((diff * diff).sum(-1))
    s = -np.log1p(dist)
    torch.save({"scores": torch.from_numpy(s.astype(np.float32))}, Path(a.out))
    print(f"v6 t={time.time()-t0:.0f}s n={len(s)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
