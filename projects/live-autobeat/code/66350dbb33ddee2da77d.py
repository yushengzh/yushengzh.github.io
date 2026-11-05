#!/usr/bin/env python3
"""Sparse BM25 retrieval with title upweighting.

k1=1.5, b=0.75, title tokens duplicated TW times.
Scoring: per query, sum over query terms of idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)).
Uses scipy sparse CSR for the doc term-frequency matrix.
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
from scipy import sparse


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def toks(s: str) -> list:
    return re.findall(r"(?u)\b\w+\b", (s or "").lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=100)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    N = len(corpus)

    # Build token lists with title upweighting
    TITLE_WEIGHT = 3
    doc_tok_lists = []
    for d in corpus:
        t = toks(d.get("title") or "")
        body = toks(d.get("text") or "")
        doc_tok_lists.append(t * TITLE_WEIGHT + body)

    doc_lens = np.array([len(x) for x in doc_tok_lists], dtype=np.float64)
    avgdl = doc_lens.mean()

    # Build term DF and TF matrix
    term_df = Counter()
    tf_list = []
    for tl in doc_tok_lists:
        c = Counter(tl)
        tf_list.append(c)
        for t in c:
            term_df[t] += 1

    vocab = sorted(term_df)
    term_idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    rows = []
    cols = []
    data = []
    for di, c in enumerate(tf_list):
        for t, tf in c.items():
            rows.append(di)
            cols.append(term_idx[t])
            data.append(tf)
    D = sparse.csr_matrix((data, (rows, cols)), shape=(N, V))

    # Precompute idf for each term (BM25+ style, non-negative)
    idf_arr = np.zeros(V, dtype=np.float64)
    for i, t in enumerate(vocab):
        df = term_df[t]
        idf_arr[i] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    k1 = 1.5
    b = 0.75

    rankings = {}
    t0 = time.time()
    for qi, q in enumerate(queries):
        qtoks = toks(q.get("text") or "")
        qcounter = Counter(qtoks)
        idxs = [term_idx[t] for t in qcounter if t in term_idx]
        scores = np.zeros(N, dtype=np.float64)
        for i in idxs:
            ttf = np.asarray(D[:, i].todense()).ravel()
            idf = idf_arr[i]
            num = ttf * (k1 + 1)
            den = ttf + k1 * (1 - b + b * doc_lens / avgdl)
            scores += idf * num / den
        topk = min(a.top_k, N)
        top_idx = scores.argsort()[::-1][:topk]
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        if (qi + 1) % 200 == 0:
            print(f"ranked {qi+1}/{len(queries)} elapsed={time.time()-t0:.1f}s", flush=True)

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote {len(rankings)} queries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
