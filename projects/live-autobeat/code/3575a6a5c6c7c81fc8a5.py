import json, time, argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    data = a.data_dir
    docs = [json.loads(l) for l in open(f"{data}/corpus.jsonl")]
    queries = [json.loads(l) for l in open(f"{data}/queries.jsonl")]
    n_docs = len(docs); n_q = len(queries)
    doc_ids = [d["_id"] for d in docs]
    titles = [(d.get("title") or "").strip() for d in docs]
    texts = [(d.get("text") or "").strip() for d in docs]
    full = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    print(f"load {n_docs} docs {n_q} q {time.time()-t0:.1f}s", flush=True)

    K1, B = 1.5, 0.75
    K1t, Bt = 1.5, 0.75
    WT_TITLE = 0.5

    vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    X = vec.fit_transform(full)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = (np.log((n_docs - df + 0.5)/(df + 0.5) + 1.0)).astype(np.float32)
    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(doc_len.mean()) or 1.0
    idf_k1 = (idf * (K1 + 1.0)).astype(np.float32)
    Xc = X.tocsc()

    tvec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
    T = tvec.fit_transform(titles)
    tdf = np.asarray((T > 0).sum(axis=0)).ravel()
    tidf = (np.log((n_docs - tdf + 0.5)/(tdf + 0.5) + 1.0)).astype(np.float32)
    tdoc_len = np.asarray(T.sum(axis=1)).ravel().astype(np.float32)
    tavgdl = float(tdoc_len.mean()) or 1.0
    idf_k1t = (tidf * (K1t + 1.0)).astype(np.float32)
    Tc = T.tocsc()

    def bm25_correct(matrix_csc, dlen, avg, idf_k1, K1v, Bv, Q):
        invl = K1v * (1.0 - Bv); k1Bv = K1v * Bv
        out = np.zeros((n_q, n_docs), dtype=np.float32)
        for qi in range(n_q):
            row = Q.getrow(qi)
            s = out[qi]
            for ti, qt in zip(row.indices, row.data):
                col = matrix_csc.getcol(ti)
                if col.nnz == 0:
                    continue
                d = col.indices
                tf = col.data.astype(np.float32)
                denom = tf + invl + k1Bv * (dlen[d] / avg)
                s[d] += (idf_k1[ti] * qt) * tf / denom
        return out

    qt = [q.get("text") or "" for q in queries]
    Qf = vec.transform(qt)
    Tq = tvec.transform(qt)
    tf0 = time.time()
    BM = bm25_correct(Xc, doc_len, avgdl, idf_k1, K1, B, Qf)
    TV = bm25_correct(Tc, tdoc_len, tavgdl, idf_k1t, K1t, Bt, Tq)
    print(f"scores done {time.time()-tf0:.1f}s", flush=True)

    S = BM + WT_TITLE * TV
    order = np.argsort(-S, axis=1)

    k = a.top_k if a.top_k else n_docs
    k = min(k, n_docs)
    rankings = {}
    for qi in range(n_q):
        top = order[qi][:k]
        rankings[queries[qi]["_id"]] = [doc_ids[j] for j in top]
        if (qi + 1) % 250 == 0:
            print(f"  wrote {qi+1}/{n_q} ({time.time()-t0:.1f}s)", flush=True)

    import torch
    torch.save({"rankings": rankings}, Path(a.out))
    print(f"wrote {len(rankings)} queries -> {a.out} ({time.time()-t0:.1f}s)", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(build())
