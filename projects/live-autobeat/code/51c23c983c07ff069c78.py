#!/usr/bin/env python3
"""True BM25F lexical retrieval + query expansion."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

K1 = 2.0
B = 0.6
TITLE_W = 1.0
TEXT_W = 1.0
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

def build_field(docs, field):
    N = len(docs)
    post_idx = defaultdict(list)
    post_tf = defaultdict(list)
    flen = np.zeros(N, dtype=np.float32)
    for did, d in enumerate(docs):
        toks = norm(d.get(field))
        flen[did] = len(toks)
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, f in counts.items():
            post_idx[t].append(did)
            post_tf[t].append(f)
    avg = float(flen.mean()) if N else 1.0
    data = {}
    for t, idxs_list in post_idx.items():
        df = len(idxs_list)
        if df < MIN_DF:
            continue
        idxs = np.array(idxs_list, dtype=np.int64)
        tf = np.array(post_tf[t], dtype=np.float32)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        data[t] = (idxs, tf, idf)
    return {"data": data, "flen": flen, "avg": avg}

def bm25f_scores(idx, qtoks):
    N = idx["N"]
    scores = np.zeros(N, dtype=np.float32)
    tdata = idx["title"]["data"]; tf_ = idx["title"]["flen"]; tavg = idx["title"]["avg"]
    xdata = idx["text"]["data"];  xf_ = idx["text"]["flen"];  xavg = idx["text"]["avg"]
    for term, cq in qtoks.items():
        v = tdata.get(term)
        if v is not None:
            idxs, tf, idf = v
            tfw = tf * (K1 + 1) / (tf + K1 * (1 - B + B * tf_[idxs] / tavg))
            np.add.at(scores, idxs, TITLE_W * idf * cq * tfw)
        v = xdata.get(term)
        if v is not None:
            idxs, tf, idf = v
            tfw = tf * (K1 + 1) / (tf + K1 * (1 - B + B * xf_[idxs] / xavg))
            np.add.at(scores, idxs, TEXT_W * idf * cq * tfw)
    return scores

def query_toks(q):
    md = q.get("metadata", {}) or {}
    toks = (norm(q.get("text"))
            + norm(md.get("query"))
            + norm(md.get("narrative")))
    counts = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return counts

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
    idx = {"title": build_field(docs, "title"),
           "text": build_field(docs, "text"),
           "N": len(docs)}
    rankings = {}
    for q in queries:
        scores = bm25f_scores(idx, query_toks(q))
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
