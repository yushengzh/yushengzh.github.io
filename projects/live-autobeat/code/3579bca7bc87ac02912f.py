#!/usr/bin/env python3
"""BM25 retrieval with title-weighting and bigrams.

Config: k1=2.5, b=0.6, idf=bm25, title_weight=3.0, bigrams=True.
Tokenization: (?u)\\b\\w+\\b lowercased.
"""
import argparse, json, math, re, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.sparse as sp

K = 10
TOK_RE = re.compile(r"(?u)\b\w+\b")

def read_jsonl(p):
    out = []
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def tok(t):
    return [w.lower() for w in TOK_RE.findall(t or '')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', required=True, choices=['val','test'])
    a = ap.parse_args()

    t0 = time.time()
    dd = Path(a.data_dir)
    corpus = read_jsonl(dd / 'corpus.jsonl')
    queries = read_jsonl(dd / 'queries.jsonl')
    print(f"loaded {len(corpus)} docs, {len(queries)} queries ({time.time()-t0:.1f}s)", flush=True)

    docs = [(d.get('title') or '', d.get('text') or '') for d in corpus]
    doc_ids = [d['_id'] for d in corpus]

    # Build BM25 index
    K1, B, TW, USE_BIGRAMS = 2.5, 0.6, 3.0, True
    n = len(docs)
    vocab = {}
    rows = []; cols = []; vals = []
    doclen = []
    for ti, (title, text) in enumerate(docs):
        c = Counter()
        for w in tok(text):
            c[w] += 1.0
        if title:
            for w in tok(title):
                c[w] += TW
        if USE_BIGRAMS:
            all_toks = (tok(title) if title else []) + tok(text)
            for i in range(len(all_toks) - 1):
                c['$' + all_toks[i] + '$' + all_toks[i+1]] += 1.0
        for w, f in c.items():
            if w not in vocab:
                vocab[w] = len(vocab)
            rows.append(ti); cols.append(vocab[w]); vals.append(f)
        doclen.append(sum(c.values()))

    X = sp.csr_matrix((vals, (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
                       shape=(n, len(vocab)), dtype=np.float64)
    Xc = X.tocsc()
    doclen = np.array(doclen, dtype=np.float64)
    avgdl = float(doclen.mean()) or 1.0
    df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
    idf[df == 0] = 0.0

    # Inverse vocab
    vocab_inv = {v: k for k, v in vocab.items()}

    print(f"vocab {len(vocab)}, nnz {len(vals)}, avgdl {avgdl:.1f} ({time.time()-t0:.1f}s)", flush=True)

    rankings = {}
    for qi, q in enumerate(queries):
        qt = Counter(tok(q.get('text', '')))
        qids = [vocab[w] for w in qt if w in vocab]
        if not qids:
            rankings[q['_id']] = []
            continue
        qf = np.array([qt[vocab_inv[i]] for i in qids], dtype=np.float64)
        scores = np.zeros(n, dtype=np.float64)
        for col, qtf in zip(qids, qf):
            c = Xc[:, col]
            if c.nnz == 0:
                continue
            d = c.indices
            tf = c.data
            denom = tf + K1 * (1.0 - B + B * doclen[d] / avgdl)
            scores[d] += idf[col] * (tf * (K1 + 1.0)) / denom * qtf
        if K < n:
            top = np.argpartition(-scores, K)[:K]
        else:
            top = np.arange(n)
        top = top[np.argsort(-scores[top])]
        rankings[q['_id']] = [doc_ids[j] for j in top]
        if (qi + 1) % 200 == 0:
            print(f"  ranked {qi+1}/{len(queries)} ({time.time()-t0:.1f}s)", flush=True)

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote rankings for {len(rankings)} queries -> {a.out} ({time.time()-t0:.1f}s)")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
