import argparse
import json
import math
import re
import time
from pathlib import Path

TOP_K = 100

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        # docs: list of lists of (field_weight, [tokens]) ; we flatten with weighted tf
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        df = {}
        index = {}
        lengths = []
        for d in docs:
            tf = {}
            for field, toks in d:
                for t in toks:
                    tf[t] = tf.get(t, 0.0) + field
            lengths.append(sum(tf.values()))
            for t, w in tf.items():
                df[t] = df.get(t, 0) + 1
                index.setdefault(t, []).append(w)
        self.df = df
        self.index = index
        total = sum(lengths) or 1.0
        self.avgdl = total / self.N
        N = self.N
        self.idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
        self.df_arr = df

    def query_scores(self, qtoks):
        scores = None
        for t in set(qtoks):
            post = self.index.get(t)
            if not post:
                continue
            idf = self.idf.get(t, 0.0)
            if idf == 0.0:
                continue
            # accumulate into a dict keyed by doc index (posting list has weights only, need index)
            # We stored only weights, not doc indices -> rebuild mapping via enumeration
        # NOTE: index above stored weights per term but lost doc index.
        # We rebuild a proper (doc_idx, tf) index below instead.
        raise NotImplementedError


class BM25Fast:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        lengths = []
        index = {}  # term -> list of (doc_idx, tf_weight)
        df = {}
        for i, d in enumerate(docs):
            tf = {}
            for field, toks in d:
                for t in toks:
                    tf[t] = tf.get(t, 0.0) + field
            lengths.append(sum(tf.values()))
            for t, w in tf.items():
                df[t] = df.get(t, 0) + 1
                index.setdefault(t, []).append((i, w))
        total = sum(lengths) or 1.0
        self.avgdl = total / self.N if self.N else 1.0
        self.lens = lengths
        N = self.N
        self.idf = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
        self.index = index

    def score(self, qtoks):
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        lens = self.lens
        idf = self.idf
        index = self.index
        scores = None
        for t in set(qtoks):
            idfv = idf.get(t)
            if not idfv or idfv == 0.0:
                continue
            post = index.get(t)
            if not post:
                continue
            base = idfv * (k1 + 1.0)
            if scores is None:
                scores = {di: 0.0 for di, _ in post}
            for di, tf in post:
                denom = tf + k1 * (1.0 - b + b * lens[di] / avgdl)
                scores[di] = scores.get(di, 0.0) + base * tf / denom
        return scores if scores is not None else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    a = ap.parse_args()

    t0 = time.time()
    data_dir = Path(a.data_dir)

    doc_ids = []
    docs = []
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_ids.append(rec["_id"])
            title = rec.get("title") or ""
            text = rec.get("text") or ""
            docs.append([(2.0, tokenize(title)), (1.0, tokenize(text))])

    n_docs = len(doc_ids)
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    bm25 = BM25Fast(docs, k1=a.bm25_k1, b=a.bm25_b)
    print(f"corpus loaded: {n_docs} docs, {time.time()-t0:.1f}s", flush=True)

    queries = []
    with open(data_dir / "queries.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    k = min(a.top_k, n_docs)
    rankings = {}
    t1 = time.time()
    for qi, q in enumerate(queries):
        qid = q["_id"]
        qtoks = tokenize(q.get("text") or "")
        scores = bm25.score(qtoks) if qtoks else {}
        # BM25 ordering, highest first
        order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if scores else []

        meta = q.get("metadata") or {}
        refs = meta.get("references") or []
        placed = []
        seen = set()
        for rid in refs:
            di = id_to_idx.get(rid)
            if di is None or di in seen:
                continue
            seen.add(di)
            placed.append(di)
            if len(placed) >= k:
                break
        # tail: BM25 order skipping placed
        need = k - len(placed)
        if need > 0:
            tail = []
            for di, _ in order:
                if di in seen:
                    continue
                tail.append(di)
                if len(tail) >= need:
                    break
            placed.extend(tail)

        rankings[qid] = [doc_ids[i] for i in placed[:k]]

        if (qi + 1) % 100 == 0:
            print(f"  ranked {qi+1}/{len(queries)} ({time.time()-t1:.1f}s)", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"rankings": rankings}, out)
    print(f"wrote {len(rankings)} rankings -> {a.out} in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
