#!/usr/bin/env python3
"""BM25 lexical retrieval + query expansion from query metadata.

Self-contained: builds an inverted BM25 index from corpus.jsonl at runtime.
Query representation = query text + metadata.query + metadata.narrative (all
lowercased, '-'->space, whitespace-split). Full-argsort BM25, top-100 per query.

CLI and output contract preserved:
    python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>
Writes {"rankings": {query_id: [doc_id, ...]}} covering every query.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

K1 = 1.2
B = 0.6
MIN_DF = 1
TOP_K = 100

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def norm(s):
    return (s or "").lower().replace("-", " ").split()

def build_index(docs):
    N = len(docs)
    post_idx = defaultdict(list); post_tf = defaultdict(list)
    dl = np.zeros(N, dtype=np.float32)
    for did, d in enumerate(docs):
        toks = norm(d.get("title")) * 2 + norm(d.get("text"))
        dl[did] = len(toks)
        c = Counter(toks)
        for t, f in c.items():
            post_idx[t].append(did); post_tf[t].append(f)
    avgdl = float(dl.mean()) if N else 1.0
    data = {}
    for t, idxs_list in post_idx.items():
        df = len(idxs_list)
        if df < MIN_DF: continue
        idxs = np.array(idxs_list, dtype=np.int64)
        tf = np.array(post_tf[t], dtype=np.float32)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        data[t] = (idxs, tf, idf)
    return {"data": data, "N": N, "dl": dl, "avgdl": avgdl}

def bm25_scores(idx, toks):
    data = idx["data"]; N = idx["N"]; dl = idx["dl"]; avgdl = idx["avgdl"]
    scores = np.zeros(N, dtype=np.float32)
    q = Counter(toks)
    for term, qf in q.items():
        v = data.get(term)
        if v is None: continue
        idxs, tf, idf = v
        tfw = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[idxs] / avgdl))
        np.add.at(scores, idxs, idf * qf * tfw)
    return scores

def query_toks(q):
    md = q.get("metadata", {}) or {}
    return (norm(q.get("text"))
            + norm(md.get("query"))
            + norm(md.get("narrative")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    ddir = Path(a.data_dir)
    docs = read_jsonl(ddir / "corpus.jsonl")
    queries = read_jsonl(ddir / "queries.jsonl")
    doc_ids = [d["_id"] for d in docs]
    idx = build_index(docs)
    rankings = {}
    for q in queries:
        scores = bm25_scores(idx, query_toks(q))
        k = min(TOP_K, idx["N"])
        topk = np.argpartition(-scores, k)[:k] if k < idx["N"] else np.arange(idx["N"])
        topk = topk[np.argsort(-scores[topk])]
        rankings[q["_id"]] = [doc_ids[i] for i in topk]
    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
