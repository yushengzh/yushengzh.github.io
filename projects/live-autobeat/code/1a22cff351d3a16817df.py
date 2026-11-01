#!/usr/bin/env python3
"""BM25 retrieval for FIQA (nDCG@10, higher is better)."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm25 import BM25, tokenize
K1 = 1.1
B = 0.6
TOP_K = 1000

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    t0 = time.time()
    corpus = load_jsonl(f"{args.data_dir}/corpus.jsonl")
    queries = load_jsonl(f"{args.data_dir}/queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    texts = [(d.get("text") or "") for d in corpus]
    print(f"loaded {len(corpus)} docs {len(queries)} queries", flush=True)
    t1 = time.time()
    index = BM25.build(texts, min_df=1, max_df_frac=1.0)
    print(f"build {time.time() - t1:.1f}s vocab={len(index.vocab)}", flush=True)
    idf = index.idf(K1)
    rankings = {}
    n = len(queries)
    for i, q in enumerate(queries):
        qid = q["_id"]
        toks = tokenize(q.get("text") or "")
        idx, _sc = index.score(toks, k1=K1, b=B, top_k=TOP_K, idf=idf)
        ranked = [doc_ids[int(di)] for di in idx]
        if not ranked:
            ranked = list(doc_ids)
        rankings[qid] = ranked
        if i % 2000 == 0:
            print(f"  {i}/{n} {time.time() - t0:.1f}s", flush=True)
    t2 = time.time()
    torch.save({"rankings": rankings}, args.out)
    print(f"saved {args.out} {time.time() - t2:.1f}s total {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
