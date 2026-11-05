#!/usr/bin/env python3
"""BM25 retrieval (Lucene-style) for arguana.

CLI contract: python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>
Output: torch.save({"rankings": {query_id: [doc_id, ...]}})  best-first.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import torch

TOP_K = 100
TOK = re.compile(r"(?u)\b\w+\b")


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d) -> str:
    t = (d.get("title") or "").strip()
    x = (d.get("text") or "").strip()
    return (t + " " + x).strip() if t else x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--bm25-k1", type=float, default=2.0)
    ap.add_argument("--bm25-b", type=float, default=0.8)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    doc_texts = [doc_text(d) for d in corpus]
    query_texts = [q.get("text") or "" for q in queries]

    n_docs = len(doc_texts)
    n_queries = len(queries)
    N = n_docs

    # ---- tokenization + document frequency ----
    doc_terms = [TOK.findall(t.lower()) for t in doc_texts]
    q_terms = [TOK.findall(t.lower()) for t in query_texts]

    df = {}
    for terms in doc_terms:
        for term in set(terms):
            df[term] = df.get(term, 0) + 1
    vocab = sorted(df.keys())
    term_index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    print(f"corpus: {n_docs} docs, {V} terms", flush=True)

    # ---- document lengths ----
    doc_len = np.array([len(terms) for terms in doc_terms], dtype=np.float64)
    avgdl = max(doc_len.mean(), 1e-9)
    k1, b = a.bm25_k1, a.bm25_b

    # ---- Lucene-style BM25 idf ----
    idf = np.empty(V, dtype=np.float64)
    for t, i in term_index.items():
        dft = df[t]
        idf[i] = np.log(1.0 + (N - dft + 0.5) / (dft + 0.5))

    # ---- build weighted term x doc matrix (V, n_docs) ----
    cols, rows, vals = [], [], []
    for di, terms in enumerate(doc_terms):
        if not terms:
            continue
        counts = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        dlf = 1.0 - b + b * doc_len[di] / avgdl
        for term, c in counts.items():
            ti = term_index[term]
            denom = c + k1 * dlf
            vals.append((c * (k1 + 1.0)) / denom)
            rows.append(ti)
            cols.append(di)
    A = csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(V, n_docs),
    )
    A = A.multiply(idf[:, None])

    # ---- score queries ----
    batch = 256
    scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    for i in range(0, n_queries, batch):
        chunk = q_terms[i:i + batch]
        qrows, qcols, qvals = [], [], []
        for qi, terms in enumerate(chunk):
            if not terms:
                continue
            counts = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            for term, c in counts.items():
                if term in term_index:
                    qrows.append(qi)
                    qcols.append(term_index[term])
                    qvals.append(float(c))
        Q = csr_matrix(
            (np.array(qvals, dtype=np.float64),
             (np.array(qrows, dtype=np.int64), np.array(qcols, dtype=np.int64))),
            shape=(len(chunk), V),
        )
        scores[i:i + len(chunk)] = (Q @ A).toarray()

    # ---- rankings ----
    rankings = {}
    k = min(a.top_k, n_docs)
    for i, q in enumerate(queries):
        order = scores[i].argsort()[::-1][:k]
        rankings[q["_id"]] = [doc_ids[j] for j in order]

    print(f"ranked {len(rankings)} queries", flush=True)
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
