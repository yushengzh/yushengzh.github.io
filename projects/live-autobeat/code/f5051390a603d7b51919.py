#!/usr/bin/env python3
"""BM25 retrieval with tuned parameters for nfcorpus biomedical dataset."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

TOP_K = 100
K1 = 2.5
B = 0.7
TOKEN = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_bm25(corpus, k1=K1, b=B):
    """Build BM25 index over title+text combined."""
    doc_tokens = []
    for d in corpus:
        title = (d.get("title") or "").lower()
        text = (d.get("text") or "").lower()
        all_tokens = TOKEN.findall(title + " " + text)
        doc_tokens.append(all_tokens)

    n_docs = len(doc_tokens)
    doc_lengths = [len(t) for t in doc_tokens]
    avgdl = sum(doc_lengths) / n_docs if n_docs > 0 else 1.0

    df = defaultdict(int)
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] += 1

    inv_idx = defaultdict(list)
    for i, tokens in enumerate(doc_tokens):
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for t, freq in tf.items():
            inv_idx[t].append((i, freq))

    idf = {}
    for t, d in df.items():
        idf[t] = math.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)

    return doc_ids_from(corpus), doc_lengths, avgdl, inv_idx, idf, n_docs


def doc_ids_from(corpus):
    return [d["_id"] for d in corpus]


def bm25_rank(query_text, doc_lengths, avgdl, inv_idx, idf, n_docs, k1=K1, b=B, top_k=TOP_K):
    """Score all docs for a query and return top-k doc indices."""
    q_tokens = TOKEN.findall(query_text.lower())
    scores = np.zeros(n_docs)

    for qt in q_tokens:
        if qt not in inv_idx:
            continue
        idf_val = idf[qt]
        if idf_val <= 0:
            continue
        for doc_i, tf in inv_idx[qt]:
            dl = doc_lengths[doc_i]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            scores[doc_i] += idf_val * numerator / denominator

    top_idx = np.argpartition(scores, -top_k)[-top_k:][::-1]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract compatibility; retrieval ranks all queries")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids, doc_lengths, avgdl, inv_idx, idf, n_docs = build_bm25(corpus)

    rankings = {}
    for i, q in enumerate(queries):
        q_text = q.get("text") or ""
        top_idx = bm25_rank(q_text, doc_lengths, avgdl, inv_idx, idf, n_docs,
                            k1=K1, b=B, top_k=min(a.top_k, n_docs))
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        if (i + 1) % 500 == 0:
            print(f"ranked {i+1}/{len(queries)} queries", flush=True)

    print(f"ranked {len(queries)}/{len(queries)} queries", flush=True)

    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
