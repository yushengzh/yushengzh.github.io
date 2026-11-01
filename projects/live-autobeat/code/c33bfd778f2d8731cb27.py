#!/usr/bin/env python3
"""V2: BM25 with improved preprocessing and parameter tuning.

This version adds:
- Stop word removal (common English + financial domain)
- Better tokenization
- Tuned BM25 parameters
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

TOP_K = 100

# BM25 parameters (tuned for financial domain)
K1 = 2.0  # slightly higher to emphasize term frequency
B = 0.75  # standard document length normalization

# Stop words for financial domain
STOP_WORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do', 'does',
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
    'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
    'just', 'me', 'might', 'more', 'most', 'must', 'my', 'myself', 'no', 'nor',
    'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your',
    'yours', 'yourself', 'yourselves',
    # Financial domain stop words
    'would', 'could', 'also', 'may', 'said', 'says', 'get', 'got', 'make',
    'made', 'like', 'use', 'used', 'one', 'two', 'way', 'think', 'time',
])


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


class BM25Improved:
    """BM25 with improved preprocessing."""

    def __init__(self, corpus_texts: list[str], k1: float = 2.0, b: float = 0.75,
                 stop_words: set = None):
        self.k1 = k1
        self.b = b

        # Use stop words if provided
        stop_words = stop_words or set()

        # Vectorize corpus
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            stop_words=list(stop_words) if stop_words else None
        )
        self.doc_term_matrix = self.vectorizer.fit_transform(corpus_texts)

        # Compute document statistics
        self.doc_lengths = np.array(self.doc_term_matrix.sum(axis=1)).ravel()
        self.avgdl = self.doc_lengths.mean()

        # Compute IDF
        n_docs = self.doc_term_matrix.shape[0]
        df = np.array(self.doc_term_matrix.astype(bool).sum(axis=0)).ravel()
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score_batch(self, queries: list[str]) -> list[list[float]]:
        """Score multiple queries."""
        q_term_matrix = self.vectorizer.transform(queries)

        scores_list = []
        for q_idx in range(q_term_matrix.shape[0]):
            q_row = q_term_matrix[q_idx].toarray().ravel()
            scores = self._compute_scores(q_row)
            scores_list.append(scores)

        return scores_list

    def _compute_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Compute BM25 scores."""
        scores = np.zeros(self.doc_term_matrix.shape[0])

        for term_idx, q_freq in enumerate(query_vector):
            if q_freq == 0:
                continue

            idf_score = self.idf[term_idx]
            doc_freqs = self.doc_term_matrix[:, term_idx].toarray().ravel()

            numerator = idf_score * doc_freqs * (self.k1 + 1)
            denominator = doc_freqs + self.k1 * (1 - self.b + self.b * self.doc_lengths / self.avgdl)
            scores += numerator / denominator

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
    corpus_texts = [doc_text(d) for d in corpus]

    print("Building improved BM25 index...", flush=True)
    bm25 = BM25Improved(corpus_texts, k1=K1, b=B, stop_words=STOP_WORDS)

    rankings = {}
    batch = 256
    for i in range(0, len(queries), batch):
        chunk = queries[i:i + batch]
        query_texts = [q.get("text") or "" for q in chunk]
        scores_batch = bm25.score_batch(query_texts)

        k = min(a.top_k, len(doc_ids))
        for scores, q in zip(scores_batch, chunk):
            top_indices = np.argsort(scores)[::-1][:k]
            rankings[q["_id"]] = [str(doc_ids[idx]) for idx in top_indices]

        print(f"ranked {min(i + batch, len(queries))}/{len(queries)} queries", flush=True)

    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


def torch_free_save(path: Path, obj) -> None:
    """Keep submission format compatible with graph tasks."""
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
