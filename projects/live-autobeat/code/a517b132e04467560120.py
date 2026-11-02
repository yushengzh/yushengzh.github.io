#!/usr/bin/env python3
"""Document retrieval: TF-IDF cosine with claim-focused doc indexing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TOP_K = 100

def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def claim(text):
    if not text:
        return ""
    if "  " in text:
        return text.split("  ", 1)[0]
    return text

def doc_text(d):
    text = (d.get("title") or "") + " " + (d.get("text") or "")
    return claim(text)

def main():
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
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=3)
    doc_mat = vectorizer.fit_transform(doc_text(d) for d in corpus)
    rankings = {}
    batch = 256
    for i in range(0, len(queries), batch):
        chunk = queries[i:i + batch]
        q_mat = vectorizer.transform([(q.get("text") or "") for q in chunk])
        sims = linear_kernel(q_mat, doc_mat)
        k = min(a.top_k, sims.shape[1])
        for row, q in zip(sims, chunk):
            idx = row.argsort()[::-1][:k]
            rankings[q["_id"]] = [doc_ids[j] for j in idx]
        print(f"ranked {min(i + batch, len(queries))}/{len(queries)} queries", flush=True)
    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0

def torch_free_save(path, obj):
    import torch
    torch.save(obj, path)

if __name__ == "__main__":
    raise SystemExit(main())
