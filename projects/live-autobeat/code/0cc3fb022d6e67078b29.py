#!/usr/bin/env python3
"""Retrieval: pure BM25 (Lucene variant), k1=3.0, b=0.5, top-100.

Self-contained, CPU-only. Replaces the weak BM25+TF-IDF hybrid: on the 15
visible dev queries the hybrid (0.5/0.5, k1=1.2, b=0.75) scored 0.2110 while
this pure-BM25 configuration scores ~0.44, improving or tying every query and
turning the 4 zero-nDCG queries nonzero. Simpler (no TF-IDF arm) and stronger.

CLI contract (unchanged):
    python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>
Writes {"rankings": {query_id: [doc_id, ...]}} (best first, top-K).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

TOP_K = 100
TOKEN = re.compile(r"[a-z0-9]+")
K1 = 3.0
B = 0.5


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d) -> str:
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()


def tokenize(text: str) -> list:
    return TOKEN.findall(text.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    t0 = time.time()
    data_dir = Path(a.data_dir)

    print("Loading corpus...", flush=True)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    n_docs = len(corpus)
    print(f"  {n_docs} docs, {len(queries)} queries", flush=True)

    texts = [doc_text(d) for d in corpus]
    tokenized_docs = [tokenize(t) for t in texts]
    print(f"  tokenized docs in {time.time()-t0:.1f}s", flush=True)

    # BM25 inverted index: term -> list of (doc_idx, tf); df per term; doc lens.
    N = n_docs
    doc_lens = np.zeros(N, dtype=np.float64)
    inverted = {}
    df = Counter()
    for i, doc in enumerate(tokenized_docs):
        doc_lens[i] = len(doc)
        tf = Counter(doc)
        for term, freq in tf.items():
            lst = inverted.get(term)
            if lst is None:
                inverted[term] = [(i, freq)]
                df[term] = 1
            else:
                lst.append((i, freq))
                df[term] += 1
    avgdl = float(doc_lens.mean()) if N > 0 else 1.0
    print(f"  BM25 index built: {len(inverted)} terms, avgdl={avgdl:.1f}, {time.time()-t0:.1f}s", flush=True)

    k1 = K1
    b = B
    k1_plus = k1 + 1.0

    rankings = {}
    topk = min(a.top_k, n_docs)
    for qi, q in enumerate(queries):
        qtext = q.get("text") or ""
        scores = np.zeros(N, dtype=np.float64)
        for term in tokenize(qtext):
            post = inverted.get(term)
            if post is None:
                continue
            d = df[term]
            idf = max(0.0, math.log((N - d + 0.5) / (d + 0.5)) + 1.0)
            if idf <= 0.0:
                continue
            for doc_idx, tf in post:
                dl = doc_lens[doc_idx]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_idx] += idf * tf * k1_plus / denom
        top_idx = np.argpartition(scores, -topk)[-topk:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        if (qi + 1) % 10 == 0:
            print(f"  ranked {qi+1}/{len(queries)} ({time.time()-t0:.1f}s)", flush=True)

    print(f"  all queries ranked in {time.time()-t0:.1f}s", flush=True)
    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
