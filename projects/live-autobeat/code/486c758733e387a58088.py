#!/usr/bin/env python3
"""Weak baseline, retrieval: TF-IDF over title+text, cosine ranking.

Deliberately lexical and untuned. BM25, dense retrieval and reranking are all
improvements the agent can make from here.

The output covers **every query in the corpus directory**; the visible
self-check scores the queries with dev qrels, the sealed grader scores the ones
with test qrels. Ranking all of them keeps the two paths identical.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract compatibility; retrieval ranks all queries")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True, min_df=1,
    )
    doc_mat = vectorizer.fit_transform(doc_text(d) for d in corpus)

    rankings = {}
    batch = 256
    for i in range(0, len(queries), batch):
        chunk = queries[i:i + batch]
        q_mat = vectorizer.transform([q.get("text") or "" for q in chunk])
        sims = linear_kernel(q_mat, doc_mat)          # [batch, n_docs]
        k = min(a.top_k, sims.shape[1])
        for row, q in zip(sims, chunk):
            idx = row.argsort()[::-1][:k]
            rankings[q["_id"]] = [doc_ids[j] for j in idx]
        print(f"ranked {min(i + batch, len(queries))}/{len(queries)} queries", flush=True)

    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


def torch_free_save(path: Path, obj) -> None:
    """Keep the submission format identical to the graph tasks (.pt via torch)."""
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
