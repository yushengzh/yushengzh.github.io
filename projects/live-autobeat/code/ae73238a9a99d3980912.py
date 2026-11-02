#!/usr/bin/env python3
"""BM25 retrieval over title and text (two-field)."""
from __future__ import annotations
import argparse, json, re, math
from pathlib import Path
import numpy as np

TOP_K = 100
TOKEN = re.compile(r"[a-z0-9]+")

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def toks(s):
    return TOKEN.findall((s or "").lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.7)
    ap.add_argument("--tw", type=float, default=1.0)
    ap.add_argument("--min-df", type=int, default=1)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    n = len(corpus)

    # Build per-field index: word -> (doc_indices, tf), plus idf; per-doc length.
    fields = {}
    for fld in ("t", "x"):
        tfm = {}
        dl = np.zeros(n, dtype=np.float32)
        for di, d in enumerate(corpus):
            src = (d.get("title") or "") if fld == "t" else (d.get("text") or "")
            t = toks(src)
            dl[di] = len(t)
            cnt = {}
            for w in t:
                cnt[w] = cnt.get(w, 0) + 1
            for w, tf in cnt.items():
                tfm.setdefault(w, []).append((di, tf))
        rows = {}
        idf = {}
        for w, lst in tfm.items():
            df = len(lst)
            if df < a.min_df:
                continue
            idf[w] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            di = np.array([p[0] for p in lst], dtype=np.int64)
            tf = np.array([p[1] for p in lst], dtype=np.float32)
            rows[w] = (di, tf)
        avgdl = float(dl.mean()) if n else 1.0
        fields[fld] = (rows, idf, dl, avgdl)

    k1, b, tw = a.k1, a.b, a.tw
    rankings = {}
    for q in queries:
        qid = q["_id"]
        s = np.zeros(n, dtype=np.float32)
        for w in set(toks(q.get("text") or "")):
            for fld, wgt in (("x", 1.0), ("t", tw)):
                rows, idf, dl, avgdl = fields[fld]
                r = rows.get(w)
                if r is None:
                    continue
                di, tf = r
                norm = k1 * (1.0 - b + b * dl[di] / avgdl)
                tfw = np.log1p(tf)
                s[di] += wgt * idf[w] * tfw / norm
        idx = np.argpartition(-s, a.top_k)[:a.top_k]
        idx = idx[np.argsort(-s[idx])]
        rankings[qid] = [doc_ids[j] for j in idx]
        del s

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print("wrote", len(rankings), "->", a.out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
