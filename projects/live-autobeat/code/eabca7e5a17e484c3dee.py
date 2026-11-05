#!/usr/bin/env python3
"""BM25+ retrieval with stop-word removal and inverted index."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

TOKEN = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    "a an and are as at be but by can do does for from has have he her him his i if in is it its "
    "me my no nor not of on or our she so that the their them then there these they this to was "
    "we were what when where which who will with you your how much why about above after again "
    "all also any because been before being below between both down during each few further "
    "had here into just more most other out over own same some such than too under until very "
    "could should would may might must shall"
)


def tokenize(text: str) -> list[str]:
    toks = TOKEN.findall(text.lower())
    return [t for t in toks if t not in STOP_WORDS]


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_inverted_index(documents: list[str]):
    n = len(documents)
    postings = defaultdict(list)
    doc_lens = np.zeros(n, dtype=np.float32)
    for i, doc in enumerate(documents):
        toks = tokenize(doc)
        doc_lens[i] = len(toks)
        tf_counts = defaultdict(int)
        for t in toks:
            tf_counts[t] += 1
        for term, tf in tf_counts.items():
            postings[term].append((i, tf))
    term_df = {term: len(lst) for term, lst in postings.items()}
    term_doc_ids = {}
    term_tfs = {}
    for term, lst in postings.items():
        term_doc_ids[term] = np.array([x[0] for x in lst], dtype=np.int32)
        term_tfs[term] = np.array([x[1] for x in lst], dtype=np.float32)
    avgdl = doc_lens.mean() if n > 0 else 1.0
    return term_doc_ids, term_tfs, term_df, doc_lens, avgdl, n


def bm25plus_query(query_tokens: list[str], term_doc_ids, term_tfs, term_df,
                   doc_lens, avgdl, k1, b, eps, n_docs) -> np.ndarray:
    scores = np.zeros(n_docs, dtype=np.float64)
    for term in set(query_tokens):
        if term not in term_df:
            continue
        df = term_df[term]
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        if idf <= 0:
            continue
        doc_ids = term_doc_ids[term]
        tfs = term_tfs[term]
        dl = doc_lens[doc_ids]
        denom = tfs + k1 * (1 - b + b * dl / avgdl) + 2 * eps * dl / avgdl
        scores[doc_ids] += idf * tfs * (k1 + 1) / denom
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--k1", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=0.50)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--top-k", type=int, default=100)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    doc_ids = [d["_id"] for d in corpus]
    doc_texts = [d.get("text") or "" for d in corpus]
    query_texts = [q.get("text") or "" for q in queries]

    print(f"Building inverted index for {len(doc_texts)} documents...", flush=True)
    term_doc_ids, term_tfs, term_df, doc_lens, avgdl, n_docs = build_inverted_index(doc_texts)
    print(f"  Vocab: {len(term_df)}, avg doc len: {avgdl:.1f}", flush=True)

    k1 = a.k1
    b = a.b
    eps = a.eps
    top_k = a.top_k

    rankings = {}
    for qi in range(len(queries)):
        q_toks = tokenize(query_texts[qi])
        scores = bm25plus_query(q_toks, term_doc_ids, term_tfs, term_df,
                               doc_lens, avgdl, k1, b, eps, n_docs)
        k = min(top_k, n_docs)
        top_indices = np.argpartition(-scores, k)[:k]
        top_indices = top_indices[scores[top_indices].argsort()[::-1]]
        rankings[queries[qi]["_id"]] = [doc_ids[j] for j in top_indices[:k]]

        if (qi + 1) % 1000 == 0:
            print(f"  {qi+1}/{len(queries)} queries done", flush=True)

    torch.save({"rankings": rankings}, Path(a.out))
    print(f"Wrote {len(rankings)} rankings -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
