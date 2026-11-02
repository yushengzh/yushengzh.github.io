#!/usr/bin/env python3
"""Two-field BM25 (title + text) for document retrieval. nDCG@10.

Standard BM25 (Lucene) over two fields with per-field length normalization
and title weight `tw`. Params tuned on the visible dev split (TUNE subset),
selected to generalize to the sealed hold-out.

CLI: python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>
Output: torch file with {"rankings": {query_id: [doc_id, ...]}} best-first.
"""
from __future__ import annotations
import argparse, json, re, math
from pathlib import Path
import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")

def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def toks(s):
    return TOKEN.findall((s or "").lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.8)
    ap.add_argument("--tw", type=float, default=1.0)
    a = ap.parse_args()
    data = Path(a.data_dir)
    corpus = read_jsonl(data / "corpus.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    n = len(corpus)

    def build(fld):
        tfm = {}
        dl = np.zeros(n, np.float32)
        for di, d in enumerate(corpus):
            src = (d.get("title") or "") if fld == "t" else (d.get("text") or "")
            t = toks(src); dl[di] = len(t)
            c = {}
            for w in t:
                c[w] = c.get(w, 0) + 1
            for w, tf in c.items():
                tfm.setdefault(w, []).append((di, tf))
        rows = {}; wmap = {}
        for w, lst in tfm.items():
            df = len(lst)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            wmap[w] = idf
            rows[w] = (np.array([p[0] for p in lst], np.int64),
                       np.array([p[1] for p in lst], np.float32))
        avg = float(dl.mean()) if n else 1.0
        return rows, wmap, dl, avg

    F = {f: build(f) for f in ("x", "t")}
    rx, wrx, dlx, avgx = F["x"]
    rt, wrt, dlv, avgt = F["t"]
    k1, b, tw = a.k1, a.b, a.tw

    rankings = {}
    for q in queries:
        s = np.zeros(n, np.float32)
        qtf = {}
        for w in toks(q.get("text") or ""):
            qtf[w] = qtf.get(w, 0) + 1
        for w, qtfv in qtf.items():
            r = rx.get(w)
            if r is not None:
                di, tf = r
                idf = wrx[w]
                norm = k1 * (1.0 - b + b * dlx[di] / avgx)
                s[di] += idf * qtfv * (tf * (k1 + 1)) / (tf + norm)
            r = rt.get(w)
            if r is not None:
                di, tf = r
                idf = wrt[w]
                norm = k1 * (1.0 - b + b * dlv[di] / avgt)
                s[di] += tw * idf * qtfv * (tf * (k1 + 1)) / (tf + norm)
        idx = np.argpartition(-s, min(a.top_k, n))[:a.top_k]
        idx = idx[np.argsort(-s[idx])]
        rankings[q["_id"]] = [doc_ids[j] for j in idx]
        del s

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print("wrote", len(rankings))

if __name__ == "__main__":
    raise SystemExit(main())
