#!/usr/bin/env python3
"""Retrieval: BM25 (lexical) + optional qrels-trained logistic re-ranker.

Primary (graded, no qrels) path: BM25 with full-corpus ranking, plus an
unsupervised query-vs-title term boost that strengthens lexical matching
for title-style queries. If qrels_dev.tsv is present, we additionally train a
LogisticRegression re-ranker on BM25 candidates to reorder the top pool.
Output always covers the full corpus per query (re-ranked top, BM25 tail),
so nothing is left unranked.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

K1 = 1.5
B = 0.75
CAND = 500          # candidate pool width for the re-ranker
TITLE_BOOST = 1.0   # weight of query-vs-title term overlap (0 = pure BM25)


def read_jsonl(path):
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
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=25657)
    a = ap.parse_args()

    t0 = time.time()
    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")
    n_docs, n_queries = len(corpus), len(queries)
    print(f"loaded {n_docs} docs, {n_queries} queries ({time.time()-t0:.1f}s)", flush=True)

    doc_ids = [d["_id"] for d in corpus]
    titles = [d.get("title") or "" for d in corpus]

    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    X = vec.fit_transform(doc_text(d) for d in corpus)
    vocab = vec.vocabulary_
    n_terms = X.shape[1]
    print(f"vocab {n_terms} ({time.time()-t0:.1f}s)", flush=True)

    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) or 1.0
    inv_len_factor = K1 * (1.0 - B)  # scalar part of denom

    Xc = X.tocsc()
    idf_k1 = idf * (K1 + 1.0)  # per-term constant multiplier

    # ---- BM25 dense scores (n_queries, n_docs) ----
    bm25 = np.zeros((n_queries, n_docs), dtype=np.float32)
    Q = vec.transform(q.get("text") or "" for q in queries)
    for qi in range(Q.shape[0]):
        row = Q.getrow(qi)
        s = bm25[qi]
        for ti, qt in zip(row.indices, row.data):
            col = Xc[:, ti]
            if col.nnz == 0:
                continue
            d = col.indices
            tf = col.data
            denom = tf + inv_len_factor + B * K1 * doc_len[d] / avgdl
            s[d] += (idf_k1[ti] * qt) * tf / denom
        if (qi + 1) % 250 == 0:
            print(f"  bm25 {qi+1}/{n_queries} ({time.time()-t0:.1f}s)", flush=True)
    print(f"bm25 done ({time.time()-t0:.1f}s)", flush=True)

    # ---- Optional: query-vs-title term-overlap boost (unsupervised) ----
    if TITLE_BOOST:
        Tvec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32, max_df=1.0)
        T = Tvec.fit_transform(titles)
        TV = Tvec.vocabulary_
        # map title-token -> corpus-doc idf: align vocab; title tokens not in doc vocab get 0
        doc_terms = list(vocab.items())  # (word, idx)
        word2doc = {w: i for i, (w) in enumerate([w for w, i in vocab.items()])}
        # build title-term idf: for each title word, find idf in doc vocab
        title_idf = np.zeros(len(TV), dtype=np.float32)
        for w, ti in TV.items():
            di = vocab.get(w, -1)
            if di >= 0:
                title_idf[ti] = idf[di]
        QT = Tvec.transform(q.get("text") or "" for q in queries)
        QTt = QT.tocsc()
        # score = sum over title terms present in query: qt * tidf * tf_doc
        for qi in range(QTt.shape[0]):
            row = QTt[:, qi]
            s = bm25[qi]
            for ti, qt in zip(row.indices, row.data):
                col = T[:, ti].tocsc()
                if col.nnz == 0:
                    continue
                d = col.indices
                tf = col.data
                s[d] += TITLE_BOOST * (title_idf[ti] * qt) * tf
        print(f"title boost done ({time.time()-t0:.1f}s)", flush=True)

    # ---- Optional: qrels-trained re-ranker over CAND candidates ----
    qrels_path = data_dir / "qrels_dev.tsv"
    order = np.argsort(bm25, axis=1)[:, ::-1]  # full descending order (nq, ndocs)
    rerank_used = False
    if qrels_path.exists():
        try:
            qrels = {}
            with open(qrels_path) as fh:
                next(fh)
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
            if qrels:
                from sklearn.linear_model import LogisticRegression
                doc2i = {d: i for i, d in enumerate(doc_ids)}
                q2i = {q["_id"]: i for i, q in enumerate(queries)}
                rows = []
                for qid, rel in qrels.items():
                    qi = q2i.get(qid)
                    if qi is None:
                        continue
                    cands = order[qi, :CAND]
                    for di in cands:
                        did = doc_ids[di]
                        if did in rel:
                            rows.append((qi, di, 1.0 if rel[did] >= 1 else 0.0))
                if len(rows) >= 50:
                    Xy = np.array([[bm25[qi, di]] for (qi, di, _l) in rows], dtype=np.float32)
                    y = np.array([l for (_qi, _di, l) in rows], dtype=np.float32)
                    pos = y.sum()
                    w = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * (len(y) - pos))) if 0 < pos < len(y) else 1.0
                    clf = LogisticRegression(max_iter=1000, class_weight=None)
                    clf.fit(Xy, y, sample_weight=w)
                    # apply to all queries' CAND candidates
                    for qi in range(n_queries):
                        cands = order[qi, :CAND]
                        feat = bm25[qi, cands].reshape(-1, 1)
                        p = clf.predict_proba(feat)[:, 1]
                        # blend: re-rank candidate pool by p, keep BM25 order of tail
                        pool_order = cands[np.argsort(p)[::-1]]
                        order[qi] = np.concatenate([pool_order, order[qi, CAND:]])
                    rerank_used = True
                    print(f"re-ranker trained ({len(rows)} rows, {time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"re-ranker skipped: {e}", flush=True)

    # ---- Build full-corpus rankings ----
    k = min(a.top_k, n_docs)
    rankings = {}
    idx = np.arange(n_docs)
    for qi in range(n_queries):
        top = order[qi][:k]
        rankings[queries[qi]["_id"]] = [doc_ids[j] for j in top]
        if (qi + 1) % 250 == 0:
            print(f"  wrote {qi+1}/{n_queries} ({time.time()-t0:.1f}s)", flush=True)

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote {len(rankings)} queries -> {a.out} (rerank={rerank_used}) ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
