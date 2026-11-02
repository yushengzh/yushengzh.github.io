#!/usr/bin/env python3
"""Sparse lexical retrieval: BM25 (Lucene) + title boost.
Config: k1=2.0, b=0.5, tf-weighted query terms, title+1.0 boost."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

K1 = 2.0
B = 0.5
TITLE_BOOST = 1.0  # added to tf for title terms
TOKEN = re.compile(r"[a-z0-9]+")


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenize(s):
    if not s:
        return []
    return TOKEN.findall(s.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    data_dir = Path(a.data_dir)

    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    n = len(corpus)

    postings = {}  # term -> list of (doc_idx, weight)
    doc_len = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(corpus):
        tt = tokenize(d.get("title") or "")
        xt = tokenize(d.get("text") or "")
        tf = Counter(xt)
        for t in tt:
            tf[t] += TITLE_BOOST
        doc_len[i] = len(tt) + len(xt)
        for t, c in tf.items():
            postings.setdefault(t, []).append((i, float(c)))
    avgdl = float(doc_len.mean())

    # per-term: (idf, docids, tf_weights)
    for t, lst in postings.items():
        df = len(lst)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        docids = np.fromiter((p[0] for p in lst), dtype=np.int32, count=df)
        weights = np.fromiter((p[1] for p in lst), dtype=np.float32, count=df)
        postings[t] = (idf, docids, weights)

    def bm25_rank(qtext):
        """BM25 with tf-weighted query terms. Returns all doc indices ranked."""
        qt = Counter(tokenize(qtext))
        qterms = {t: float(c) for t, c in qt.items()}  # tf weighting
        scores = {}
        for term, w in qterms.items():
            rec = postings.get(term)
            if rec is None:
                continue
            idf, docids, tf = rec
            dl = doc_len[docids]
            denom = tf + K1 * (1.0 - B + B * dl / avgdl)
            vals = (w * idf * (tf * (K1 + 1.0)) / denom)
            dlst = docids.tolist()
            vlist = vals.tolist()
            for di, v in zip(dlst, vlist):
                scores[di] = scores.get(di, 0.0) + v
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [d for d, _ in items]

    doc_ids = [d["_id"] for d in corpus]
    rankings = {}
    for q in queries:
        ranks = bm25_rank(q.get("text") or "")
        ranked_set = set(ranks)
        # Append unranked docs to cover all documents
        rankings[q["_id"]] = [doc_ids[j] for j in ranks] + [doc_ids[j] for j in range(n) if j not in ranked_set]

    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"wrote {len(rankings)} rankings -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
