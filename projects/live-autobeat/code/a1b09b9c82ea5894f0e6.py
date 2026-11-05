#!/usr/bin/env python3
"""BM25F + bi-gram features for FIQA retrieval.

Improvements over TF-IDF cosine:
1. BM25F with field-aware term weighting (title gets higher weight)
2. Bi-gram features to capture multi-word phrases
3. Tuned parameters (k1, b, title weight)

Output: {"rankings": {query_id: [doc_id, ...]}}  best first
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# BM25F parameters - tuned on visible qrels
K1_BODY = 1.2      # saturation parameter for body text
B_BODY = 0.75      # length normalization for body
K1_TITLE = 1.5     # saturation parameter for title
B_TITLE = 0.75     # length normalization for title
TITLE_WEIGHT = 3.0 # relative importance of title field
K1_BI = 1.2        # saturation for bigrams
B_BI = 0.75        # length normalization for bigrams


def tokenize(text: str) -> list[str]:
    """Tokenize: lowercase, extract alphanumeric tokens (allow hyphens)."""
    if not text:
        return []
    return TOKEN.findall(text.lower())


def make_bigrams(tokens: list[str]) -> list[str]:
    """Create bigram features from token list."""
    if len(tokens) < 2:
        return []
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_bm25f_scores(query_tokens, query_bigrams,
                         body_term_freqs, body_doc_lens, body_avg_len,
                         title_term_freqs, title_doc_lens, title_avg_len,
                         bi_term_freqs, bi_doc_lens, bi_avg_len,
                         df_body, df_title, df_bi, n_docs):
    """Compute BM25F score for one query against all docs.
    
    BM25F: for each field f, score = sum over terms of:
        idf(t) * (tf(t,f) * (k1_f + 1)) / (tf(t,f) + k1_f * (1 - b_f + b_f * dl_f / avgdl_f))
    
    For multi-field: tf is summed across fields, but weighted.
    """
    scores = np.zeros(n_docs, dtype=np.float32)
    
    # Process unigram terms (body + title combined)
    for term in set(query_tokens):
        # Get term frequencies in body and title
        if term in body_term_freqs:
            tf_body = body_term_freqs[term]  # dict: doc_idx -> tf
        else:
            tf_body = {}
        if term in title_term_freqs:
            tf_title = title_term_freqs[term]
        else:
            tf_title = {}
        
        # IDF for combined field
        df = df_body.get(term, 0) + df_title.get(term, 0)
        if df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        # Get arrays of tf for all docs (default 0)
        body_tfs = np.zeros(n_docs, dtype=np.float32)
        for doc_idx, tf in tf_body.items():
            body_tfs[doc_idx] = tf
        title_tfs = np.zeros(n_docs, dtype=np.float32)
        for doc_idx, tf in tf_title.items():
            title_tfs[doc_idx] = tf
        
        # BM25F: effective tf = body_tf + TITLE_WEIGHT * title_tf
        # Use combined normalization
        # body norm factor
        body_norm = 1.0 - B_BODY + B_BODY * body_doc_lens / body_avg_len
        title_norm = 1.0 - B_TITLE + B_TITLE * title_doc_lens / title_avg_len
        
        # BM25 score contribution from body
        body_score = idf * body_tfs * (K1_BODY + 1) / (body_tfs + K1_BODY * body_norm)
        # BM25 score contribution from title
        title_score = idf * TITLE_WEIGHT * title_tfs * (K1_TITLE + 1) / (title_tfs + K1_TITLE * title_norm)
        
        scores += (body_score + title_score).astype(np.float32)
    
    # Process bigram terms
    for term in set(query_bigrams):
        if term in bi_term_freqs:
            tf_bi = bi_term_freqs[term]
        else:
            continue
        df = df_bi.get(term, 0)
        if df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        bi_tfs = np.zeros(n_docs, dtype=np.float32)
        for doc_idx, tf in tf_bi.items():
            bi_tfs[doc_idx] = tf
        
        bi_norm = 1.0 - B_BI + B_BI * bi_doc_lens / bi_avg_len
        bi_score = idf * bi_tfs * (K1_BI + 1) / (bi_tfs + K1_BI * bi_norm)
        scores += bi_score.astype(np.float32)
    
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    t0 = time.time()
    data_dir = Path(a.data_dir)
    
    print(f"Loading corpus...", flush=True)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    n_docs = len(corpus)
    print(f"Loaded {n_docs} docs, {len(queries)} queries in {time.time()-t0:.1f}s", flush=True)
    
    # --- Build inverted indexes ---
    print("Building indexes...", flush=True)
    t1 = time.time()
    
    doc_ids = [d["_id"] for d in corpus]
    
    body_term_freqs = {}   # term -> {doc_idx: tf}
    title_term_freqs = {}  # term -> {doc_idx: tf}
    bi_term_freqs = {}     # bigram -> {doc_idx: tf}
    df_body = Counter()
    df_title = Counter()
    df_bi = Counter()
    body_doc_lens = np.zeros(n_docs, dtype=np.float32)
    title_doc_lens = np.zeros(n_docs, dtype=np.float32)
    bi_doc_lens = np.zeros(n_docs, dtype=np.float32)
    
    for i, d in enumerate(corpus):
        text = d.get("text") or ""
        title = d.get("title") or ""
        
        # Body tokens
        body_tokens = tokenize(text)
        body_doc_lens[i] = len(body_tokens)
        if body_tokens:
            body_counter = Counter(body_tokens)
            for term, tf in body_counter.items():
                if term not in body_term_freqs:
                    body_term_freqs[term] = {}
                body_term_freqs[term][i] = tf
                df_body[term] += 1
        
        # Title tokens (boosted)
        title_tokens = tokenize(title)
        title_doc_lens[i] = len(title_tokens)
        if title_tokens:
            title_counter = Counter(title_tokens)
            for term, tf in title_counter.items():
                if term not in title_term_freqs:
                    title_term_freqs[term] = {}
                title_term_freqs[term][i] = tf
                df_title[term] += 1
        
        # Bigrams from body + title
        all_tokens = title_tokens + body_tokens  # title first for bigrams
        bigrams = make_bigrams(all_tokens)
        bi_doc_lens[i] = len(bigrams)
        if bigrams:
            bi_counter = Counter(bigrams)
            for term, tf in bi_counter.items():
                if term not in bi_term_freqs:
                    bi_term_freqs[term] = {}
                bi_term_freqs[term][i] = tf
                df_bi[term] += 1
    
    body_avg_len = body_doc_lens.mean() if body_doc_lens.mean() > 0 else 1.0
    title_avg_len = title_doc_lens.mean() if title_doc_lens.mean() > 0 else 1.0
    bi_avg_len = bi_doc_lens.mean() if bi_doc_lens.mean() > 0 else 1.0
    
    print(f"  body: {len(body_term_freqs)} terms, avg_len={body_avg_len:.1f}", flush=True)
    print(f"  title: {len(title_term_freqs)} terms, avg_len={title_avg_len:.1f}", flush=True)
    print(f"  bigrams: {len(bi_term_freqs)} terms, avg_len={bi_avg_len:.1f}", flush=True)
    print(f"Index building: {time.time()-t1:.1f}s", flush=True)
    
    # --- Score queries ---
    print("Scoring queries...", flush=True)
    t2 = time.time()
    
    rankings = {}
    TOP_K = 100
    
    for qi, q in enumerate(queries):
        qtext = q.get("text") or ""
        q_tokens = tokenize(qtext)
        q_bigrams = make_bigrams(q_tokens)
        
        scores = compute_bm25f_scores(
            q_tokens, q_bigrams,
            body_term_freqs, body_doc_lens, body_avg_len,
            title_term_freqs, title_doc_lens, title_avg_len,
            bi_term_freqs, bi_doc_lens, bi_avg_len,
            df_body, df_title, df_bi, n_docs,
        )
        
        # Top-K
        if n_docs <= TOP_K:
            top_idx = np.argsort(scores)[::-1]
        else:
            # Partial argsort for speed
            top_k_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
            top_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        
        if (qi + 1) % 500 == 0:
            elapsed = time.time() - t2
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / rate if rate > 0 else 0
            print(f"  {qi+1}/{len(queries)} queries ({rate:.1f} q/s, ETA {eta:.0f}s)", flush=True)
    
    print(f"Scoring: {time.time()-t2:.1f}s", flush=True)
    
    # Save
    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"Wrote rankings for {len(rankings)} queries -> {a.out}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
