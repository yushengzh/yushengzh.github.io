#!/usr/bin/env python3
"""Retrieval: BM25 + word TF-IDF cosine + char n-gram cosine, minmax fusion.

v3: MM minmax(1,1,1) over the union of top-400 candidates, full-corpus output.
HGB reranker removed (dead on the graded test split which ships no qrels)."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def doc_text(d) -> str:
    title = (d.get("title") or "").lower()
    text = (d.get("text") or "").lower()
    return (title + " " + title + " " + title + " " + text).strip()


class BM25:
    """Okapi BM25 scorer using inverted index for speed."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.n_docs: int = 0

    def fit(self, documents: list[str]):
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)
        self.doc_freqs = Counter()
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf_counts = Counter(tokens)
            for term, tf in tf_counts.items():
                self.inverted_index[term].append((i, tf))
                self.doc_freqs[term] += 1
        self.n_docs = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.n_docs, 1)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for qt in query_tokens:
            if qt not in self.doc_freqs:
                continue
            df = self.doc_freqs[qt]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx, tf in self.inverted_index[qt]:
                dl = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_length)
                scores[doc_idx] += idf * numerator / denominator
        return scores


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    n_docs = len(corpus)
    n_queries = len(queries)

    # --- BM25 with inverted index (fast) ---
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit([doc_text(d) for d in corpus])

    # --- Word TF-IDF vectorizer ---
    word_vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        min_df=1,
        max_features=100000,
    )
    doc_word_vecs = word_vec.fit_transform(
        [f"{d.get('title') or ''} {d.get('title') or ''} {d.get('text') or ''}" for d in corpus]
    )
    query_word_vecs = word_vec.transform([q.get("text") or "" for q in queries])

    # --- Char n-gram TF-IDF vectorizer (3-5 grams) ---
    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=200000,
    )
    doc_char_vecs = char_vec.fit_transform(
        [f"{d.get('title') or ''} {d.get('text') or ''}" for d in corpus]
    )
    query_char_vecs = char_vec.transform([q.get("text") or "" for q in queries])

    # --- Compute similarity matrices (per-query sparse dot products) ---
    bm25_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    bm25_topk = []
    for i, q in enumerate(queries):
        qt = tokenize(q.get("text") or "")
        bm25_scores[i] = bm25.score(qt)
        top_k = min(400, n_docs)
        idx = np.argpartition(-bm25_scores[i], top_k)[:top_k]
        idx = idx[np.argsort(-bm25_scores[i, idx])]
        bm25_topk.append(idx)

    word_topk = []
    word_scores_all = np.zeros((n_queries, n_docs), dtype=np.float64)
    for i in range(n_queries):
        sims = (query_word_vecs[i] @ doc_word_vecs.T).toarray().ravel()
        word_scores_all[i] = sims
        top_k = min(400, n_docs)
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        word_topk.append(idx)

    char_topk = []
    char_scores_all = np.zeros((n_queries, n_docs), dtype=np.float64)
    for i in range(n_queries):
        sims = (query_char_vecs[i] @ doc_char_vecs.T).toarray().ravel()
        char_scores_all[i] = sims
        top_k = min(400, n_docs)
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        char_topk.append(idx)

    # --- Build candidate sets: union of BM25, word, char top-400 ---
    TOP_K_CANDIDATES = 400
    candidates = []
    for i in range(n_queries):
        cset = set(bm25_topk[i].tolist())
        cset.update(word_topk[i].tolist())
        cset.update(char_topk[i].tolist())
        cands = np.array(sorted(cset), dtype=np.int64)
        candidates.append(cands)

    # --- Generate final rankings: minmax fusion over candidates, full-corpus pad ---
    rankings = {}
    all_doc_idx = np.arange(n_docs)
    for i, q in enumerate(queries):
        cands = candidates[i]
        b = bm25_scores[i, cands]
        w = word_scores_all[i, cands]
        c = char_scores_all[i, cands]

        def _mm(x):
            mn = x.min()
            rng = x.max() - mn
            return (x - mn) / rng if rng > 1e-9 else np.zeros_like(x)

        fused = _mm(b) + _mm(w) + _mm(c)
        order = fused.argsort()[::-1]
        cands_order = cands[order]  # candidates, best fused first

        # Pad with remaining (non-candidate) docs by BM25 desc, full corpus safety.
        if cands_order.size < n_docs:
            mask = np.ones(n_docs, dtype=bool)
            mask[cands_order] = False
            rest = all_doc_idx[mask]
            rest = rest[np.argsort(-bm25_scores[i, rest])]
            ranked = np.concatenate([cands_order, rest])
        else:
            ranked = cands_order

        rankings[q["_id"]] = [doc_ids[j] for j in ranked]

    # --- Save ---
    import torch
    torch.save({"rankings": rankings}, a.out)
    elapsed = time.time() - t0
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
