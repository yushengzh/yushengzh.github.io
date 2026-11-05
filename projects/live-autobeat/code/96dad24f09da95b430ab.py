#!/usr/bin/env python3
"""Retrieval: BM25-style + TF hybrid using sklearn TfidfVectorizer.

sublinear_tf=True gives 1+log(tf), approximating BM25 with k1=0.
We blend with raw TF for a signal complement.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

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
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    doc_texts = [doc_text(d) for d in corpus]

    # BM25-like: sublinear_tf, idf=True
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        min_df=1,
    )
    doc_mat = vectorizer.fit_transform(doc_texts)  # sparse CSR

    # Raw TF: no idf, no sublinear
    tf_vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=False,
        use_idf=False,
        min_df=1,
    )
    tf_mat = tf_vectorizer.fit_transform(doc_texts)

    # Precompute row norms for cosine normalization
    doc_norms = np.sqrt((doc_mat.toarray() ** 2).sum(axis=1))
    tf_norms = np.sqrt((tf_mat.toarray() ** 2).sum(axis=1))

    rankings = {}
    batch = 256
    for i in range(0, len(queries), batch):
        chunk = queries[i:i + batch]
        q_texts = [q.get("text") or "" for q in chunk]

        q_bm25 = vectorizer.transform(q_texts).toarray()  # dense [B, V]
        q_tf = tf_vectorizer.transform(q_texts).toarray()  # dense [B, V]

        # Cosine similarity = (q . d) / (|q| * |d|)
        q_bm25_norms = np.sqrt((q_bm25 ** 2).sum(axis=1, keepdims=True))
        q_tf_norms = np.sqrt((q_tf ** 2).sum(axis=1, keepdims=True))

        # scores = (q/d_norms) @ d_norms  ... use matrix mult
        bm25_scores = (q_bm25 / q_bm25_norms) @ doc_mat.T.toarray()
        tf_scores = (q_tf / q_tf_norms) @ tf_mat.T.toarray()

        combined = 0.7 * bm25_scores + 0.3 * tf_scores

        k = min(a.top_k, combined.shape[1])
        for row, q in zip(combined, chunk):
            idx = row.argsort()[::-1][:k]
            rankings[q["_id"]] = [doc_ids[j] for j in idx]

        print(f"ranked {min(i + batch, len(queries))}/{len(queries)} queries", flush=True)

    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


def torch_free_save(path: Path, obj) -> None:
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
