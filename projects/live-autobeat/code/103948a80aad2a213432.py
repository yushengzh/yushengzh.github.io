#!/usr/bin/env python3
"""Retrieval baseline v1: TF-IDF (title weighted 2x) sublinear cosine, top-1000.

Command contract: python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>
Writes {"rankings": {qid: [doc_id, ...]}} (best first) covering every query.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

TOP_K = 1000
TITLE_R = 2


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d, title_r=TITLE_R) -> str:
    t = (d.get("title") or "")
    return (" ".join([t] * title_r) + " " + (d.get("text") or "")).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract; retrieval ranks all queries")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b",
                          sublinear_tf=True, min_df=1)
    doc_mat = vec.fit_transform(doc_text(d) for d in corpus)

    rankings = {}
    batch = 256
    for i in range(0, len(queries), batch):
        chunk = queries[i:i + batch]
        q_mat = vec.transform([q.get("text") or "" for q in chunk])
        sims = linear_kernel(q_mat, doc_mat)
        k = min(a.top_k, sims.shape[1])
        for row, q in zip(sims, chunk):
            idx = row.argsort()[::-1][:k]
            rankings[q["_id"]] = [doc_ids[j] for j in idx]
        print(f"ranked {min(i + batch, len(queries))}/{len(queries)}", flush=True)

    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote {len(rankings)} queries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
