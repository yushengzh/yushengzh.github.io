#!/usr/bin/env python3
"""BM25 (Okapi) unigram retrieval for FIQA.  (v2)

Plain BM25 over document body text, unigrams only. Drops the bigram and
title fields (titles are absent in this corpus and bigrams consistently
hurt on the visible split). Tuned on the visible qrels.

Output: {"rankings": {query_id: [doc_id, ...]}}  best first
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# BM25 parameters
K1 = 0.9
B = 0.7
TOP_K = 100


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN.findall(text.lower())


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    data_dir = a.data_dir

    print("Loading corpus...", flush=True)
    corpus = read_jsonl(f"{data_dir}/corpus.jsonl")
    queries = read_jsonl(f"{data_dir}/queries.jsonl")
    n_docs = len(corpus)
    print(f"Loaded {n_docs} docs, {len(queries)} queries in {time.time()-t0:.1f}s", flush=True)

    print("Building index...", flush=True)
    t1 = time.time()

    doc_ids = [d["_id"] for d in corpus]

    term_freqs = {}   # term -> {doc_idx: tf}
    df = Counter()
    doc_lens = np.zeros(n_docs, dtype=np.float32)

    for i, d in enumerate(corpus):
        toks = tokenize(d.get("text") or "")
        doc_lens[i] = len(toks)
        if toks:
            counter = Counter(toks)
            for term, tf in counter.items():
                if term not in term_freqs:
                    term_freqs[term] = {}
                term_freqs[term][i] = tf
                df[term] += 1

    avg_len = doc_lens.mean() if doc_lens.mean() > 0 else 1.0
    print(f"  terms: {len(term_freqs)}, avg_len={avg_len:.1f}", flush=True)
    print(f"Index building: {time.time()-t1:.1f}s", flush=True)

    # Precompute idf per term
    idf = {}
    for term, dft in df.items():
        idf[term] = math.log((n_docs - dft + 0.5) / (dft + 0.5) + 1.0)

    # norm factor per doc (b term)
    norm = 1.0 - B + B * doc_lens / avg_len

    print("Scoring queries...", flush=True)
    t2 = time.time()

    rankings = {}
    for qi, q in enumerate(queries):
        qtoks = tokenize(q.get("text") or "")
        qset = set(qtoks)

        scores = np.zeros(n_docs, dtype=np.float32)
        for term in qset:
            idf_t = idf.get(term)
            if idf_t is None or idf_t == 0.0:
                continue
            tf_map = term_freqs.get(term)
            if not tf_map:
                continue
            tf_arr = np.zeros(n_docs, dtype=np.float32)
            for doc_idx, tf in tf_map.items():
                tf_arr[doc_idx] = tf
            denom = tf_arr + K1 * norm
            scores += (idf_t * tf_arr * (K1 + 1.0) / denom).astype(np.float32)

        if n_docs <= TOP_K:
            top_idx = np.argsort(scores)[::-1]
        else:
            top_k_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
            top_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]

        if (qi + 1) % 500 == 0:
            elapsed = time.time() - t2
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / rate if rate > 0 else 0
            print(f"  {qi+1}/{len(queries)} ({rate:.1f} q/s, ETA {eta:.0f}s)", flush=True)

    print(f"Scoring: {time.time()-t2:.1f}s", flush=True)

    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"Wrote rankings for {len(rankings)} queries -> {a.out}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
