#!/usr/bin/env python3
"""Improved retrieval: BM25 + TF-IDF hybrid."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
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


def build_bm25_index(docs_text):
    """Build inverted index for BM25."""
    n_docs = len(docs_text)
    doc_token_freqs = []
    doc_lens = []
    df = {}
    for dt in docs_text:
        tokens = dt.lower().split()
        freqs = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        doc_token_freqs.append(freqs)
        doc_lens.append(len(tokens))
        for t in freqs:
            df[t] = df.get(t, 0) + 1
    
    avg_dl = sum(doc_lens) / n_docs if n_docs > 0 else 1
    idf = {}
    for t, d in df.items():
        idf[t] = (d, math.log((n_docs - d + 0.5) / (d + 0.5) + 1))
    
    return doc_token_freqs, doc_lens, avg_dl, idf


def bm25_query_score(query_text, doc_token_freqs, doc_lens, avg_dl, idf, k1=1.5, b=0.75):
    """Score one query against all docs using BM25. Vectorized with numpy."""
    n_docs = len(doc_token_freqs)
    scores = np.zeros(n_docs)
    q_tokens = query_text.lower().split()
    q_tf = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
    
    for t, qt in q_tf.items():
        if t not in idf:
            continue
        d, idf_val = idf[t]
        # Vectorize: get tf for all docs
        tf_arr = np.array([doc_token_freqs[j].get(t, 0) for j in range(n_docs)])
        mask = tf_arr > 0
        if mask.sum() == 0:
            continue
        dl_arr = np.array(doc_lens, dtype=np.float64)
        bm25_tf = tf_arr[mask] * (k1 + 1) / (tf_arr[mask] + k1 * (1 - b + b * dl_arr[mask] / avg_dl))
        scores[mask] += idf_val * bm25_tf
    return scores


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
    docs_text = [doc_text(d) for d in corpus]
    titles = [d.get('title') or '' for d in corpus]
    texts = [d.get('text') or '' for d in corpus]
    n_docs = len(corpus)
    n_queries = len(queries)

    print(f"Loading done: {n_docs} docs, {n_queries} queries", flush=True)

    # Build BM25 index
    print("Building BM25 index...", flush=True)
    doc_token_freqs, doc_lens, avg_dl, idf = build_bm25_index(docs_text)
    
    # Build TF-IDF variants
    print("Building TF-IDF variants...", flush=True)
    tfidf_full = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1, norm='l2'
    )
    tfidf_full_mat = tfidf_full.fit_transform(docs_text)
    
    tfidf_title = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1, norm='l2'
    )
    tfidf_title_mat = tfidf_title.fit_transform(titles)
    
    tfidf_text = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1, norm='l2'
    )
    tfidf_text_mat = tfidf_text.fit_transform(texts)

    # Score queries
    print("Scoring queries...", flush=True)
    rankings = {}
    batch_size = 50
    
    for i in range(0, n_queries, batch_size):
        chunk = queries[i:i + batch_size]
        q_texts = [q.get('text') or '' for q in chunk]
        
        # BM25 scores (one at a time for memory)
        bm25_scores_arr = np.zeros((len(chunk), n_docs))
        for qi, qt in enumerate(q_texts):
            bm25_scores_arr[qi] = bm25_query_score(
                qt, doc_token_freqs, doc_lens, avg_dl, idf
            )
        
        # TF-IDF cosine scores - use .dot() for sparse-sparse
        q_full_mat = tfidf_full.transform(q_texts)
        tfidf_full_scores = (q_full_mat @ tfidf_full_mat.T).toarray()
        
        q_title_mat = tfidf_title.transform(q_texts)
        tfidf_title_scores = (q_title_mat @ tfidf_title_mat.T).toarray()
        
        q_text_mat = tfidf_text.transform(q_texts)
        tfidf_text_scores = (q_text_mat @ tfidf_text_mat.T).toarray()
        
        # Normalize each to [0,1]
        def normalize(s):
            mn, mx = s.min(), s.max()
            if mx > mn:
                return (s - mn) / (mx - mn)
            return s
        
        bm25_norm = normalize(bm25_scores_arr)
        tfidf_full_norm = normalize(tfidf_full_scores)
        tfidf_title_norm = normalize(tfidf_title_scores)
        tfidf_text_norm = normalize(tfidf_text_scores)
        
        # Ensemble
        combined = (0.35 * bm25_norm + 0.30 * tfidf_full_norm +
                    0.20 * tfidf_title_norm + 0.15 * tfidf_text_norm)
        
        k = min(a.top_k, n_docs)
        for qi, q in enumerate(chunk):
            idx = combined[qi].argsort()[::-1][:k]
            rankings[q["_id"]] = [doc_ids[j] for j in idx]
        
        print(f"ranked {min(i + batch_size, n_queries)}/{n_queries} queries", flush=True)

    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


def torch_free_save(path: Path, obj) -> None:
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
