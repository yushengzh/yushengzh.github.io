#!/usr/bin/env python3
"""BM25 retrieval for FiQA (BEIR). Correct per-term saturation scoring."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np
from scipy import sparse
import torch

TOKEN = re.compile(r"[a-z0-9%$.]+")
STOP = set(
    "a an the and or of to in on for with from by at is are was were be been being "
    "it its this that these those as but not no your you my our their they he she i "
    "we can could will would should may might do does did has have had what when where "
    "which who how why if then than so just more most other into out up down over under "
    "again very too also about after before between through during above below"
    .split())

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def tokenize(s):
    return TOKEN.findall((s or "").lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--k1", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--stop", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    n = len(corpus)
    k1, b, stop = a.k1, a.b, a.stop

    rows, cols, vals = [], [], []
    doclens = np.zeros(n, dtype=np.int32)
    for i, d in enumerate(corpus):
        toks = tokenize(d.get("text"))
        if stop:
            f = [t for t in toks if t not in STOP]
            toks = f if f else toks
        doclens[i] = len(toks)
        c = {}
        for t in toks:
            c[t] = c.get(t, 0) + 1
        for t, tf in c.items():
            rows.append(t); cols.append(i); vals.append(tf)
    doclens = np.asarray(doclens)
    avgdl = float(doclens.mean()) or 1.0
    vocab = {}
    for t in rows:
        if t not in vocab:
            vocab[t] = len(vocab)
    rows = [vocab[t] for t in rows]
    df = np.bincount(rows, minlength=len(vocab))
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(vocab), n))
    W.sum_duplicates()
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
    denom = (1.0 - b) + b * doclens / avgdl
    print("indexed %d docs vocab=%d in %.1fs" % (n, len(vocab), time.time() - t0), flush=True)

    doc_ids = [d["_id"] for d in corpus]
    k = min(a.top_k, n)
    rankings = {}
    t1 = time.time()
    for q in queries:
        toks = tokenize(q.get("text"))
        if stop:
            f = [t for t in toks if t not in STOP]
            toks = f if f else toks
        qi = [vocab[t] for t in set(toks) if t in vocab]
        if not qi:
            rankings[q["_id"]] = doc_ids[:k]
            continue
        s = np.zeros(n, dtype=np.float64)
        for t in qi:
            vec = W[t, :]
            tf = np.asarray(vec.data, dtype=np.float64)
            cols = vec.indices
            s[cols] += idf[t] * (k1 + 1.0) * tf / (tf + k1 * denom[cols])
        idx = np.argpartition(-s, k)[:k]
        idx = idx[np.argsort(-s[idx])]
        rankings[q["_id"]] = [doc_ids[j] for j in idx]
    print("ranked %d queries in %.1fs" % (len(queries), time.time() - t1), flush=True)
    torch.save({"rankings": rankings}, a.out)
    print("wrote %d rankings -> %s" % (len(rankings), a.out), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
