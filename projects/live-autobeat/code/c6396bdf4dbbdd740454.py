#!/usr/bin/env python3
"""Improved retrieval: BM25 + TF-IDF hybrid.

Combines BM25 (Lucene variant) with TF-IDF cosine using normalized score fusion.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TOP_K = 100
TOKEN = re.compile(r"[a-z0-9]+")


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


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """BM25 (Lucene variant) with k1=1.2, b=0.75."""
    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.doc_lens = np.zeros(self.N)
        self.df = Counter()  # document frequency for each term
        # Inverted index: term -> list of (doc_id, term_freq)
        self.inverted = {}
        
        for i, doc in enumerate(documents):
            self.doc_lens[i] = len(doc)
            tf = Counter(doc)
            for term, freq in tf.items():
                if term not in self.inverted:
                    self.inverted[term] = []
                self.inverted[term].append((i, freq))
                self.df[term] += 1
        
        self.avgdl = self.doc_lens.mean() if self.N > 0 else 1.0
    
    def score(self, query_tokens: list[str]) -> np.ndarray:
        """Score all documents for a query."""
        scores = np.zeros(self.N)
        for term in query_tokens:
            if term not in self.inverted:
                continue
            # IDF (Lucene variant: max(0, log((N - df + 0.5)/(df + 0.5)) + 1))
            df = self.df[term]
            idf = max(0.0, math.log((self.N - df + 0.5) / (df + 0.5)) + 1.0)
            for doc_id, tf in self.inverted[term]:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        return scores


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--bm25-weight", type=float, default=0.5)
    ap.add_argument("--tfidf-weight", type=float, default=0.5)
    a = ap.parse_args()
    
    t0 = time.time()
    data_dir = Path(a.data_dir)
    
    print("Loading corpus...", flush=True)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    n_docs = len(corpus)
    print(f"  {n_docs} docs, {len(queries)} queries", flush=True)
    
    # Prepare text
    texts = [doc_text(d) for d in corpus]
    
    # BM25
    print("Building BM25 index...", flush=True)
    tokenized_docs = [tokenize(t) for t in texts]
    bm25 = BM25(tokenized_docs, k1=1.2, b=0.75)
    print(f"  BM25 done in {time.time()-t0:.1f}s", flush=True)
    
    # TF-IDF
    print("Building TF-IDF...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1, ngram_range=(1, 2),
    )
    doc_mat = vectorizer.fit_transform(texts)
    print(f"  TF-IDF done in {time.time()-t0:.1f}s (vocab={vectorizer.vocabulary_.__len__()})", flush=True)
    
    # Score all queries
    rankings = {}
    bm25_w = a.bm25_weight
    tfidf_w = a.tfidf_weight
    
    for qi, q in enumerate(queries):
        qtext = q.get("text") or ""
        qtokens = tokenize(qtext)
        
        # BM25 score
        bm25_scores = bm25.score(qtokens)
        
        # TF-IDF score
        q_mat = vectorizer.transform([qtext])
        tfidf_scores = linear_kernel(q_mat, doc_mat)[0]  # [n_docs]
        
        # Normalize and fuse
        bm25_norm = normalize_scores(bm25_scores)
        tfidf_norm = normalize_scores(tfidf_scores)
        
        fused = bm25_w * bm25_norm + tfidf_w * tfidf_norm
        
        # Top-K
        k = min(a.top_k, n_docs)
        top_idx = np.argpartition(fused, -k)[-k:]
        top_idx = top_idx[np.argsort(fused[top_idx])[::-1]]
        
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        
        if (qi + 1) % 10 == 0:
            print(f"  ranked {qi+1}/{len(queries)} queries ({time.time()-t0:.1f}s)", flush=True)
    
    print(f"  All queries ranked in {time.time()-t0:.1f}s", flush=True)
    
    # Save
    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
