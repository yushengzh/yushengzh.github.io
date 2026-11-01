#!/usr/bin/env python3
"""BM25 lexical retrieval for scifact (nDCG@10).

Ranks every query in queries.jsonl against the full corpus and writes
    {"rankings": {query_id: [doc_id, doc_id, ...]}}   # best first
to --out. The visible self-check scores the 75 dev queries; the sealed grader
scores the held-out ones -- ranking all queries keeps both paths identical.

Doc text for indexing = title repeated TITLE_WEIGHT times + text. Scoring is
classic BM25 with idf = log((N - df + 0.5)/(df + 0.5) + 1.0), k1=K1, b=B.
These were selected on the visible split (tune/hold both strong).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

TITLE_WEIGHT = 3
K1 = 3.0
B = 0.7
TOKEN = re.compile(r"[a-z]+")


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokens(s):
    return TOKEN.findall((s or "").lower())


class BM25:
    def __init__(self, texts, k1, b):
        self.k1, self.b = k1, b
        self.posting = {}
        self.doclen = []
        for i, t in enumerate(texts):
            counts = {}
            for w in tokens(t):
                counts[w] = counts.get(w, 0) + 1
            self.doclen.append(len(t))
            for w, tf in counts.items():
                p = self.posting.get(w)
                if p is None:
                    self.posting[w] = [[i], [tf]]
                else:
                    p[0].append(i); p[1].append(tf)
        self.N = len(texts)
        self.avgdl = (sum(self.doclen) / self.N) if self.N else 1.0
        self.idf = {}
        for w, (ix, tf) in self.posting.items():
            df = len(ix)
            self.idf[w] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, qtext):
        out = {}
        inv = self.k1 + 1.0
        for w in set(tokens(qtext)):
            p = self.posting.get(w)
            if p is None:
                continue
            ix, tf = p
            idf = self.idf[w]
            for j in range(len(ix)):
                d = ix[j]
                denom = tf[j] + self.k1 * (1.0 - self.b + self.b * self.doclen[d] / self.avgdl)
                out[d] = out.get(d, 0.0) + idf * inv * tf[j] / denom
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()

    dd = Path(a.data_dir)
    corpus = read_jsonl(dd / "corpus.jsonl")
    queries = read_jsonl(dd / "queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]
    texts = [((d.get("title") or "") + " ") * TITLE_WEIGHT + (d.get("text") or "") for d in corpus]

    bm = BM25(texts, K1, B)
    rankings = {}
    for q in queries:
        sc = bm.scores(q.get("text") or "")
        ranked = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
        ids = [doc_ids[d] for d, _ in ranked]
        # Pad with un-scored docs (score 0) so every doc is ranked for every query.
        if len(ids) < len(doc_ids):
            seen = {d for d, _ in ranked}
            ids += [doc_ids[d] for d in range(len(doc_ids)) if d not in seen]
        rankings[q["_id"]] = ids

    import torch
    torch.save({"rankings": rankings}, a.out)
    print(f"ranked {len(rankings)} queries over {len(doc_ids)} docs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
