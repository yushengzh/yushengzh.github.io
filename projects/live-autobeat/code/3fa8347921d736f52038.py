#!/usr/bin/env python3
"""Lexical retrieval: BM25 over title(2x)+text, finance-aware tokenizer."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path

TOP_K = 1000
K1 = 1.5
B = 0.75

TOKEN = re.compile(r"[a-z0-9%$]+")

STOP = set("""a an the and or of to in on for with from by at is are was were be been
being it its this that these those as but not no your you my our their they he she
i we can could will would should may might do does did has have had do what when
where which who how why if then than so just more most other into out up down over
under again very too also about after before between through during above below""".split())


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenize(s, keep_stop=True):
    toks = TOKEN.findall((s or "").lower())
    if not keep_stop:
        toks = [t for t in toks if t not in STOP]
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    import numpy as np
    import time
    t0 = time.time()
    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    n = len(corpus)

    docs_tokens = []
    for d in corpus:
        tt = tokenize((d.get("title") or ""), keep_stop=False)
        tx = tokenize(d.get("text") or "", keep_stop=False)
        docs_tokens.append(tt + tx + tt)

    doclens = np.fromiter((len(t) for t in docs_tokens), dtype=np.float32, count=n)
    avgdl = float(doclens.mean()) or 1.0

    doc_freq = {}
    postings = {}
    for i, toks in enumerate(docs_tokens):
        c = {}
        for t in toks:
            c[t] = c.get(t, 0) + 1
        for t in c:
            doc_freq[t] = doc_freq.get(t, 0) + 1
        for t, tf in c.items():
            postings.setdefault(t, []).append((i, tf))

    idf = {}
    for t, df in doc_freq.items():
        idf[t] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    print("indexed %d docs in %.1fs" % (n, time.time() - t0), flush=True)

    doc_ids = [d["_id"] for d in corpus]
    k = min(a.top_k, n)
    rankings = {}
    t1 = time.time()
    for q in queries:
        toks = tokenize(q.get("text") or "", keep_stop=False)
        s = np.zeros(n, dtype=np.float64)
        for t in set(toks):
            post = postings.get(t)
            if not post:
                continue
            idf_t = idf[t]
            for i, tf in post:
                denom = tf + K1 * (1.0 - B + B * doclens[i] / avgdl)
                s[i] += idf_t * (tf * (K1 + 1)) / denom
        if not any(s):
            rankings[q["_id"]] = []
            continue
        if k >= n:
            idx = np.argsort(-s)
        else:
            idx = np.argpartition(-s, k)[:k]
            idx = idx[np.argsort(-s[idx])]
        rankings[q["_id"]] = [doc_ids[j] for j in idx]
    print("ranked %d queries in %.1fs" % (len(queries), time.time() - t1), flush=True)

    import torch
    torch.save({"rankings": rankings}, a.out)
    print("wrote %d rankings -> %s" % (len(rankings), a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
