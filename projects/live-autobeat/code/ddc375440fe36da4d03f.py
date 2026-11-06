#!/usr/bin/env python3
"""Improved retrieval: IDF-weighted BM25 + TF-IDF with bigrams.
v2: Down-weight rare terms in BM25 to fix zero-score queries where
rare query terms dominate but aren't in relevant docs' text."""
from __future__ import annotations
import argparse, json, math, re, time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TOP_K = 100
TOKEN = re.compile(r"[a-z0-9]+")

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def doc_text(d):
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()

def tokenize(text):
    return TOKEN.findall(text.lower())

class BM25:
    def __init__(self, documents, k1=1.2, b=0.75, title_boost=2.0):
        self.k1 = k1; self.b = b; self.N = len(documents)
        self.doc_lens = np.zeros(self.N)
        self.df = Counter(); self.inverted = {}
        for i, doc in enumerate(documents):
            self.doc_lens[i] = len(doc)
            tf = Counter(doc)
            for term, freq in tf.items():
                if term not in self.inverted:
                    self.inverted[term] = []
                self.inverted[term].append((i, freq))
                self.df[term] += 1
        self.avgdl = self.doc_lens.mean() if self.N > 0 else 1.0
        # Precompute IDF for each term
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5)) + 1.0

    def score(self, query_tokens):
        scores = np.zeros(self.N)
        # IDF-weighted: terms in >10% of docs are likely topic-relevant,
        # terms in <5% are likely query-specific noise
        for term in query_tokens:
            if term not in self.inverted:
                continue
            df = self.df[term]
            frac = df / self.N
            # Term importance: down-weight very rare terms, up-weight common topic terms
            if frac < 0.01:
                term_w = 0.3  # rare term - likely query-specific
            elif frac < 0.03:
                term_w = 0.6
            elif frac < 0.10:
                term_w = 1.0  # normal
            else:
                term_w = 1.5  # common topic term - up-weight
            idf = self.idf[term]
            for doc_id, tf in self.inverted[term]:
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += term_w * idf * tf * (self.k1 + 1) / denom
        return scores

def normalize_scores(scores):
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--bm25-weight", type=float, default=0.4)
    ap.add_argument("--tfidf-weight", type=float, default=0.6)
    a = ap.parse_args()
    t0 = time.time()
    data_dir = Path(a.data_dir)
    print("Loading corpus...", flush=True)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    n_docs = len(corpus)
    print(f"  {n_docs} docs, {len(queries)} queries", flush=True)
    texts = [doc_text(d) for d in corpus]
    print("Building BM25 index...", flush=True)
    tokenized_docs = [tokenize(t) for t in texts]
    bm25 = BM25(tokenized_docs, k1=1.2, b=0.75)
    print(f"  BM25 done in {time.time()-t0:.1f}s", flush=True)
    print("Building TF-IDF...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1, ngram_range=(1, 2),
    )
    doc_mat = vectorizer.fit_transform(texts)
    print(f"  TF-IDF done in {time.time()-t0:.1f}s (vocab={len(vectorizer.vocabulary_)})", flush=True)
    rankings = {}
    bm25_w = a.bm25_weight; tfidf_w = a.tfidf_weight
    for qi, q in enumerate(queries):
        qtext = q.get("text") or ""
        qtokens = tokenize(qtext)
        bm25_scores = bm25.score(qtokens)
        q_mat = vectorizer.transform([qtext])
        tfidf_scores = linear_kernel(q_mat, doc_mat)[0]
        bm25_norm = normalize_scores(bm25_scores)
        tfidf_norm = normalize_scores(tfidf_scores)
        fused = bm25_w * bm25_norm + tfidf_w * tfidf_norm
        k = min(a.top_k, n_docs)
        top_idx = np.argpartition(fused, -k)[-k:]
        top_idx = top_idx[np.argsort(fused[top_idx])[::-1]]
        rankings[q["_id"]] = [doc_ids[j] for j in top_idx]
        if (qi + 1) % 10 == 0:
            print(f"  ranked {qi+1}/{len(queries)} queries ({time.time()-t0:.1f}s)", flush=True)
    print(f"  All queries ranked in {time.time()-t0:.1f}s", flush=True)
    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} ({time.time()-t0:.1f}s total)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
