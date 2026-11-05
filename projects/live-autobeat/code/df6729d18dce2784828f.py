#!/usr/bin/env python3
"""Lexical BM25 document retrieval. Deterministic, label-free, no title (corpus has none).
Emits top-100 per query as torch.save({"rankings": {qid: [doc_id,...]}})."""
from __future__ import annotations
import argparse, json, re, math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import torch

TOP_K = 100
TOK = re.compile(r"(?u)\b\w+\b")

def read_jsonl(path):
    out=[]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--bm25-k1", type=float, default=2.5)
    ap.add_argument("--bm25-b", type=float, default=1.35)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir/"corpus.jsonl")
    queries = read_jsonl(data_dir/"queries.jsonl")
    doc_ids=[str(d["_id"]) for d in corpus]
    texts=[(d.get("text") or "").strip() for d in corpus]
    n_docs=len(doc_ids); N=n_docs
    q_texts=[(q.get("text") or "") for q in queries]
    k1,b = a.bm25_k1, a.bm25_b

    xt=[TOK.findall(t.lower()) for t in texts]
    qt=[TOK.findall(t.lower()) for t in q_texts]

    df={}
    for terms in xt:
        for term in set(terms): df[term]=df.get(term,0)+1
    vocab=sorted(df); tidx={t:i for i,t in enumerate(vocab)}; V=len(vocab)
    idf=np.empty(V)
    for t,i in tidx.items():
        dft=df[t]; idf[i]=np.log(1.0+(N-dft+0.5)/(dft+0.5))

    # build doc-term BM25 matrix for text field
    ln=np.array([len(x) for x in xt],dtype=np.float64)
    avg=max(ln.mean(),1e-9)
    rows,cols,vals=[],[],[]
    for di,terms in enumerate(xt):
        if not terms: continue
        counts={}
        for term in terms: counts[term]=counts.get(term,0)+1
        dlf=1.0-b+b*ln[di]/avg
        for term,c in counts.items():
            ti=tidx[term]; denom=c+k1*dlf
            vals.append((c*(k1+1.0))/denom); rows.append(ti); cols.append(di)
    Ax=csr_matrix((np.array(vals),(np.array(rows,dtype=np.int64),np.array(cols,dtype=np.int64))),shape=(V,n_docs))
    Ax=Ax.multiply(idf[:,None])

    def qmat(chunk):
        qrows,qcols,qvals=[],[],[]
        for qi,terms in enumerate(chunk):
            if not terms: continue
            counts={}
            for term in terms: counts[term]=counts.get(term,0)+1
            for term,c in counts.items():
                if term in tidx:
                    qrows.append(qi); qcols.append(tidx[term]); qvals.append(1.0+math.log(c))
        return csr_matrix((np.array(qvals),(np.array(qrows),np.array(qcols))),shape=(len(chunk),V))

    batch=256
    nq=len(queries)
    scores=np.zeros((nq,n_docs))
    for i in range(0,nq,batch):
        chunk=qt[i:i+batch]; Q=qmat(chunk)
        S=(Q@Ax).toarray()
        scores[i:i+len(chunk)]=S

    k=min(a.top_k,n_docs)
    rankings={}
    for i,q in enumerate(queries):
        # stable descending sort
        order=np.argsort(-scores[i], kind="stable")[:k]
        rankings[str(q["_id"])]=[doc_ids[j] for j in order]
    torch.save({"rankings":rankings}, Path(a.out))
    print(f"pure BM25: ranked {len(rankings)} queries, {n_docs} docs")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
