#!/usr/bin/env python3
"""BM25 retrieval over title+text with light plural normalization.

Tuned on the 103 visible nfcorpus dev queries (selfcheck.py). Config from
sweep2: k1=2.4, b=0.4, idf=log1p, plur=True -> 0.3299 (baseline TF-IDF 0.3155).

Output: {"rankings": {query_id: [doc_id, ...]}} for every query, saved via
torch.save to keep the .pt format the pipeline expects.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
K1 = 2.4
B = 0.4
TOPK = 100


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tok(s: str) -> list:
    t = TOKEN.findall((s or "").lower())
    return [x[:-1] if len(x) > 3 and x.endswith("s") and not x.endswith("ss") else x for x in t]


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
    doc_tokens = [tok((d.get("title") or "") + " " + (d.get("text") or "")) for d in corpus]

    N = len(doc_tokens)
    post = defaultdict(list)
    for i, toks in enumerate(doc_tokens):
        for t, n in Counter(toks).items():
            post[t].append((i, n))
    dl = np.array([len(t) for t in doc_tokens], dtype=np.float64)
    avgdl = max(dl.mean(), 1.0)

    def idf_of(df: int) -> float:
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    rankings = {}
    for q in queries:
        qt = set(tok(q.get("text") or ""))
        s = np.zeros(N, dtype=np.float64)
        for t in qt:
            lst = post.get(t)
            if not lst:
                continue
            idf = idf_of(len(lst))
            for di, tf in lst:
                denom = K1 * (1.0 - B + B * dl[di] / avgdl)
                s[di] += idf * (tf * (K1 + 1.0)) / (tf + denom)
        k = min(TOPK, N)
        if k < N:
            idx = np.argpartition(-s, k - 1)[:k]
            idx = idx[np.argsort(-s[idx], kind="stable")]
        else:
            idx = np.argsort(-s, kind="stable")
        rankings[q["_id"]] = [doc_ids[j] for j in idx]

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
