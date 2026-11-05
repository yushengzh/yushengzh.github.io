#!/usr/bin/env python3
"""Improved retrieval: BM25 + TF-IDF cosine + char n-gram cosine + HGB reranker.

v2: Broader candidate union, graded target, char n-grams, rank-lookup dicts, HGB.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics.pairwise import linear_kernel

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
    # BM25 scores for all queries
    bm25_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    bm25_topk = []
    for i, q in enumerate(queries):
        qt = tokenize(q.get("text") or "")
        bm25_scores[i] = bm25.score(qt)
        top_k = min(400, n_docs)
        idx = np.argpartition(-bm25_scores[i], top_k)[:top_k]
        idx = idx[np.argsort(-bm25_scores[i, idx])]
        bm25_topk.append(idx)

    # Word cosine similarities (sparse matmul, stored per-query for top candidates only)
    # We compute full sparse products but only store top-k per query
    word_topk = []
    word_scores_all = np.zeros((n_queries, n_docs), dtype=np.float64)
    for i in range(n_queries):
        sims = (query_word_vecs[i] @ doc_word_vecs.T).toarray().ravel()
        word_scores_all[i] = sims
        top_k = min(400, n_docs)
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        word_topk.append(idx)

    # Char n-gram cosine similarities
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

    # --- Build rank lookup dicts for O(1) rank access ---
    bm25_rank = []
    word_rank = []
    char_rank = []
    for i in range(n_queries):
        r = np.full(n_docs, TOP_K_CANDIDATES, dtype=np.float64)
        for rank, d in enumerate(bm25_topk[i]):
            r[d] = float(rank)
        bm25_rank.append(r)
        r = np.full(n_docs, TOP_K_CANDIDATES, dtype=np.float64)
        for rank, d in enumerate(word_topk[i]):
            r[d] = float(rank)
        word_rank.append(r)
        r = np.full(n_docs, TOP_K_CANDIDATES, dtype=np.float64)
        for rank, d in enumerate(char_topk[i]):
            r[d] = float(rank)
        char_rank.append(r)

    # --- Load visible qrels for training ---
    qrels_path = data_dir / "qrels_dev.tsv"
    qrels = {}
    if qrels_path.exists():
        with open(qrels_path, encoding="utf-8") as fh:
            fh.readline()  # skip header
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    qid, cid, score = parts[0], parts[1], int(parts[2])
                    qrels.setdefault(qid, {})[cid] = score

    qid_to_idx = {q["_id"]: i for i, q in enumerate(queries)}

    # --- Build training data from visible qrels ---
    X_train_list = []
    y_train_list = []
    for qid, rels in qrels.items():
        q_idx = qid_to_idx.get(qid)
        if q_idx is None:
            continue
        cands = candidates[q_idx]
        max_rel = max(rels.values()) if rels else 0
        for d in cands:
            doc_id = doc_ids[d]
            bm25_s = bm25_scores[q_idx, d]
            word_s = word_scores_all[q_idx, d]
            char_s = char_scores_all[q_idx, d]
            b_rank = bm25_rank[q_idx][d]
            w_rank = word_rank[q_idx][d]
            c_rank = char_rank[q_idx][d]
            rel = rels.get(doc_id, 0)
            # Graded target: use raw relevance score (0, 1, or 2)
            X_train_list.append([bm25_s, word_s, char_s, b_rank, w_rank, c_rank])
            y_train_list.append(float(rel))

    # --- Train reranker: HistGradientBoosting on graded target ---
    reranker = None
    use_reranker = False
    if len(X_train_list) > 100 and len(set(y_train_list)) > 1:
        X_train = np.array(X_train_list, dtype=np.float64)
        y_train = np.array(y_train_list, dtype=np.float64)
        try:
            reranker = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            ).fit(X_train, y_train)
            use_reranker = True
        except Exception as e:
            print(f"HGB failed: {e}, falling back to BM25 order")

    # --- Generate final rankings ---
    rankings = {}
    top_k_output = 100

    for i, q in enumerate(queries):
        cands = candidates[i]

        if use_reranker and reranker is not None:
            # Build feature matrix for candidates (O(1) rank lookup)
            X_cands = np.array([
                [bm25_scores[i, d], word_scores_all[i, d], char_scores_all[i, d],
                 bm25_rank[i][d], word_rank[i][d], char_rank[i][d]]
                for d in cands
            ], dtype=np.float64)
            pred_scores = reranker.predict(X_cands)
            order = pred_scores.argsort()[::-1]
            ranked_doc_indices = cands[order]
        else:
            # Fallback: score = bm25 + word + char (simple fusion)
            fused = bm25_scores[i, cands] + 0.5 * word_scores_all[i, cands] + 0.3 * char_scores_all[i, cands]
            order = fused.argsort()[::-1]
            ranked_doc_indices = cands[order]

        top = ranked_doc_indices[:top_k_output]
        rankings[q["_id"]] = [doc_ids[j] for j in top]

    # --- Save ---
    import torch
    torch.save({"rankings": rankings}, a.out)
    elapsed = time.time() - t0
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
