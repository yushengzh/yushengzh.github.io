#!/usr/bin/env python3
"""Improved retrieval: BM25 + metadata signals.

Command line contract:
  python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>
"""
from __future__ import annotations

import argparse
import json
import math
import re
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


def tokenize(text: str) -> list:
    return TOKEN.findall(text.lower())


def doc_text(d) -> str:
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()


class BM25:
    """BM25 ranking function."""
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_tokens = corpus_tokens
        self.doc_lens = np.array([len(t) for t in corpus_tokens], dtype=np.float64)
        self.avgdl = self.doc_lens.mean() if self.n_docs > 0 else 1.0
        # Document frequency
        df = Counter()
        for tokens in corpus_tokens:
            for t in set(tokens):
                df[t] += 1
        self.df = df
        # Precompute idf for all terms
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def score(self, query_tokens):
        """Score all documents against query tokens."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for q_t in query_tokens:
            if q_t not in self.idf:
                continue
            idf = self.idf[q_t]
            # Get document frequencies for this term
            # We need tf for each doc
            for i, doc_toks in enumerate(self.doc_tokens):
                tf = doc_toks.count(q_t)
                if tf > 0:
                    dl = self.doc_lens[i]
                    tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    scores[i] += idf * tf_norm
        return scores


class BM25Fast:
    """BM25 with inverted index for speed."""
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_tokens = corpus_tokens
        self.doc_lens = np.array([len(t) for t in corpus_tokens], dtype=np.float64)
        self.avgdl = self.doc_lens.mean() if self.n_docs > 0 else 1.0
        # Build inverted index: term -> list of (doc_id, tf)
        self.inv_index = {}
        df = Counter()
        for i, tokens in enumerate(corpus_tokens):
            tf_counter = Counter(tokens)
            for term, tf in tf_counter.items():
                if term not in self.inv_index:
                    self.inv_index[term] = []
                self.inv_index[term].append((i, tf))
                df[term] += 1
        self.df = df
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def score(self, query_tokens):
        """Score all documents against query tokens using inverted index."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for q_t in query_tokens:
            if q_t not in self.idf:
                continue
            idf = self.idf[q_t]
            for doc_id, tf in self.inv_index[q_t]:
                dl = self.doc_lens[doc_id]
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[doc_id] += idf * tf_norm
        return scores


def ndcg_at_k(ranked, qrels, k=10):
    """Compute nDCG@k for a single query."""
    def dcg(rels):
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    gains = [qrels.get(doc, 0) for doc in ranked[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg(ideal)
    if idcg <= 0:
        return 0.0
    return dcg(gains) / idcg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # --- BM25 over title + text ---
    print(f"Tokenizing {len(corpus)} docs...", flush=True)
    corpus_tokens = [tokenize(doc_text(d)) for d in corpus]
    bm25 = BM25Fast(corpus_tokens, k1=a.bm25_k1, b=a.bm25_b)
    print(f"BM25 index built ({len(bm25.inv_index)} terms)", flush=True)

    # --- Metadata signals ---
    # Build lookup structures for metadata
    # Doc authors set
    doc_authors = [set(d.get('metadata', {}).get('authors', [])) for d in corpus]
    doc_year = [d.get('metadata', {}).get('year', 0) for d in corpus]
    doc_cited_by = [set(d.get('metadata', {}).get('cited_by', [])) for d in corpus]
    doc_refs = [set(d.get('metadata', {}).get('references', [])) for d in corpus]

    # Query metadata
    q_years = [q.get('metadata', {}).get('year', 0) for q in queries]
    q_authors = [set(q.get('metadata', {}).get('authors', [])) for q in queries]
    q_refs = [set(q.get('metadata', {}).get('references', [])) for q in queries]
    q_cited_by = [set(q.get('metadata', {}).get('cited_by', [])) for q in queries]

    print(f"Computing rankings for {len(queries)} queries...", flush=True)
    rankings = {}
    
    for qi, q in enumerate(queries):
        q_tokens = tokenize(q.get("text") or "")
        
        # BM25 scores
        bm25_scores = bm25.score(q_tokens)
        
        # Metadata bonuses
        n_docs = len(corpus)
        meta_scores = np.zeros(n_docs, dtype=np.float64)
        
        # 1. Reference signal: if query references a doc, that doc is likely relevant
        q_ref_set = q_refs[qi]
        if q_ref_set:
            for ref_id in q_ref_set:
                if ref_id in doc_id_to_idx:
                    meta_scores[doc_id_to_idx[ref_id]] += 5.0
        
        # 2. Cited_by signal: if a doc cites the query (query is in doc's references)
        # This is equivalent to: doc's references include query_id
        q_id = q["_id"]
        # We can't easily check this without building a reverse index
        # But we can check: if query's cited_by includes a doc, that doc cites the query
        q_cb_set = q_cited_by[qi]
        if q_cb_set:
            for cb_id in q_cb_set:
                if cb_id in doc_id_to_idx:
                    meta_scores[doc_id_to_idx[cb_id]] += 3.0
        
        # 3. Author overlap
        q_auth = q_authors[qi]
        if q_auth:
            for i, d_auth in enumerate(doc_authors):
                overlap = len(q_auth & d_auth)
                if overlap > 0:
                    meta_scores[i] += overlap * 2.0
        
        # 4. Year proximity (small bonus)
        q_year = q_years[qi]
        if q_year:
            for i, d_year in enumerate(doc_year):
                if d_year:
                    diff = abs(q_year - d_year)
                    if diff <= 1:
                        meta_scores[i] += 1.0
                    elif diff <= 2:
                        meta_scores[i] += 0.5
        
        # Combine: BM25 + metadata
        # Normalize BM25 scores to [0,1] range for combination
        if bm25_scores.max() > 0:
            bm25_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_norm = bm25_scores
        
        combined = bm25_norm + meta_scores / bm25_scores.max() * 0.1 if bm25_scores.max() > 0 else bm25_norm + meta_scores
        
        k = min(a.top_k, n_docs)
        top_idx = combined.argsort()[::-1][:k]
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        
        if (qi + 1) % 100 == 0:
            print(f"  ranked {qi+1}/{len(queries)}", flush=True)

    print(f"Writing rankings for {len(rankings)} queries -> {a.out}", flush=True)
    torch_free_save(Path(a.out), {"rankings": rankings})
    return 0


def torch_free_save(path: Path, obj) -> None:
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
