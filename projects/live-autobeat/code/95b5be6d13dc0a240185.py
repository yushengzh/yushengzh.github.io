#!/usr/bin/env python3
"""Okapi BM25 for document retrieval."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d) -> str:
    return (d.get("title") or "") + " " + (d.get("text") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=100)
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
    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) or 1.0

    idf = np.log(n_docs / (df + 0.0)).astype(np.float32)  # raw idf

    Xc = X.tocsc()
    Q = vectorizer.transform(q.get("text") or "" for q in queries)

    K1 = 2.0
    B = 0.9

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
            scores[docs] += idf[term_idx] * (tf * (K1 + 1.0)) / denom * q_tf
        if k < n_docs:
            top = np.argpartition(scores, -k)[-k:]
        else:
            top = np.arange(n_docs)
        top = top[np.argsort(scores[top])[::-1]]
        rankings[queries[qi]["_id"]] = [doc_ids[j] for j in top]
        if (qi + 1) % 500 == 0:
            print(f"  ranked {qi + 1}/{Q.shape[0]} ({time.time() - t0:.1f}s)", flush=True)

    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} "
          f"({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
