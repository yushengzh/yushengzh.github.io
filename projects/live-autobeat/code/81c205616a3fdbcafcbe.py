#!/usr/bin/env python3
"""Baseline: TF-IDF cosine ranking."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

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
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1, max_features=50000,
    )
    doc_texts = [f"{d.get('title') or ''} {d.get('text') or ''}" for d in corpus]
    doc_vecs = vectorizer.fit_transform(doc_texts)
    query_vecs = vectorizer.transform([q.get("text") or "" for q in queries])
    cos_sims = linear_kernel(query_vecs, doc_vecs)
    top_k = 100
    rankings = {}
    for i, q in enumerate(queries):
        top = np.argsort(cos_sims[i])[::-1][:top_k]
        rankings[q["_id"]] = [doc_ids[j] for j in top]
    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
