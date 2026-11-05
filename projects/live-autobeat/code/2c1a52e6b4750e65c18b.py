#!/usr/bin/env python3
"""Improved retrieval: BM25 + TF-IDF cosine + learned reranker.

v1: BM25 with title boosting, plus logistic regression reranker trained on visible qrels.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    # Repeat title 3x for boosting
    return (title + " " + title + " " + title + " " + text).strip()


class BM25:
    """Okapi BM25 scorer using inverted index for speed."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
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

    # --- TF-IDF vectorizer for cosine similarity ---
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        min_df=1,
        max_features=50000,
    )
    doc_vecs = vectorizer.fit_transform(
        [f"{d.get('title') or ''} {d.get('title') or ''} {d.get('text') or ''}" for d in corpus]
    )
    query_vecs = vectorizer.transform([q.get("text") or "" for q in queries])

    # --- Cosine similarities: [n_queries, n_docs] ---
    cos_sims = linear_kernel(query_vecs, doc_vecs)

    # --- BM25 scores for all queries ---
    bm25_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    for i, q in enumerate(queries):
        q_tokens = tokenize(q.get("text") or "")
        bm25_scores[i] = bm25.score(q_tokens)

    # --- Candidate sets ---
    top_k_candidates = 200
    bm25_topk = np.argsort(bm25_scores, axis=1)[:, ::-1][:, :top_k_candidates]
    cos_topk = np.argsort(cos_sims, axis=1)[:, ::-1][:, :top_k_candidates]

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

    # Build query index map
    qid_to_idx = {q["_id"]: i for i, q in enumerate(queries)}

    # --- Build training data from visible qrels ---
    X_train_list = []
    y_train_list = []
    for qid, rels in qrels.items():
        q_idx = qid_to_idx.get(qid)
        if q_idx is None:
            continue
        cand = bm25_topk[q_idx]
        for doc_idx in cand:
            doc_id = doc_ids[doc_idx]
            bm25_s = bm25_scores[q_idx, doc_idx]
            cos_s = cos_sims[q_idx, doc_idx]
            bm25_rank = float(np.where(cand == doc_idx)[0][0])
            cos_rank = float(np.where(cos_topk[q_idx] == doc_idx)[0][0]) if doc_idx in cos_topk[q_idx] else float(top_k_candidates)
            rel = 1 if rels.get(doc_id, 0) > 0 else 0
            X_train_list.append([bm25_s, cos_s, bm25_rank, cos_rank])
            y_train_list.append(rel)

    # --- Train reranker ---
    reranker = None
    if X_train_list and len(set(y_train_list)) > 1:
        X_train = np.array(X_train_list, dtype=np.float64)
        y_train = np.array(y_train_list, dtype=np.int32)
        reranker = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(X_train, y_train)

    # --- Generate final rankings ---
    rankings = {}
    top_k_output = 100

    for i, q in enumerate(queries):
        cands = bm25_topk[i]

        if reranker is not None:
            # Build feature matrix for candidates
            X_cands = np.array([
                [bm25_scores[i, d], cos_sims[i, d],
                 float(np.where(cands == d)[0][0]),
                 float(np.where(cos_topk[i] == d)[0][0]) if d in cos_topk[i] else float(top_k_candidates)]
                for d in cands
            ], dtype=np.float64)
            pred_scores = reranker.predict_proba(X_cands)[:, 1]
            order = pred_scores.argsort()[::-1]
            ranked_doc_indices = cands[order]
        else:
            ranked_doc_indices = cands

        top = ranked_doc_indices[:top_k_output]
        rankings[q["_id"]] = [doc_ids[j] for j in top]

    # --- Save ---
    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


def torch_free_save(path: Path, obj) -> None:
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
