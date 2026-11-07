#!/usr/bin/env python3
"""Okapi BM25 with title repeated twice (title_tw=2).

BM25 (Okapi) with title repeated 2x in the indexed text to increase title
weighting. This was validated via /tmp/harness.py as a broad improvement
(+0.017 over v1, gains across most of 15 visible queries).

    score(D,Q) = sum_q IDF(q) * f(q,D)*(k1+1) / (f(q,D) + k1*(1-b+b*|D|/avgdl))
    IDF(q)     = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )

Indexed text per doc: "title title text"  (title appears 2x -> effective 2x weight)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

K1 = 5.5
B = 0.3
TOP_K = 100


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d) -> str:
    """Return indexed text with title repeated twice (title_tw=2)."""
    title = (d.get('title') or '').strip()
    text = (d.get('text') or '').strip()
    return ' '.join([title, title, text]).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract compatibility; retrieval ranks all queries")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    t0 = time.time()
    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    print(f"loaded {len(corpus)} docs, {len(queries)} queries "
          f"({time.time() - t0:.1f}s)", flush=True)

    doc_ids = [d["_id"] for d in corpus]
    vectorizer = CountVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
    )
    X = vectorizer.fit_transform(doc_text(d) for d in corpus)
    print(f"vocab {X.shape[1]}, nnz {X.nnz} ({time.time() - t0:.1f}s)", flush=True)

    n_docs = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) or 1.0

    Xc = X.tocsc()
    Q = vectorizer.transform(q.get("text") or "" for q in queries)

    rankings = {}
    k = min(a.top_k, n_docs)
    for qi in range(Q.shape[0]):
        row = Q.getrow(qi)
        scores = np.zeros(n_docs, dtype=np.float32)
        for term_idx, q_tf in zip(row.indices, row.data):
            col = Xc[:, term_idx]
            if col.nnz == 0:
                continue
            docs = col.indices
            tf = col.data
            denom = tf + K1 * (1.0 - B + B * doc_len[docs] / avgdl)
            scores[docs] += idf[term_idx] * (tf * (K1 + 1.0)) / denom
        if k < n_docs:
            top = np.argpartition(scores, -k)[-k:]
        else:
            top = np.arange(n_docs)
        top = top[np.argsort(scores[top])[::-1]]
        rankings[queries[qi]["_id"]] = [doc_ids[j] for j in top]
        if (qi + 1) % 200 == 0:
            print(f"  ranked {qi + 1}/{Q.shape[0]} ({time.time() - t0:.1f}s)", flush=True)

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} "
          f"({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
