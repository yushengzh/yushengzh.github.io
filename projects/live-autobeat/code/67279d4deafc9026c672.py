#!/usr/bin/env python3
"""Two-field BM25 (title + text) with title boost. v5-compatible default (title_w=1.0 two-field == merged approx)."""
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
    ap.add_argument("--bm25-b", type=float, default=1.36)
    ap.add_argument("--title-w", type=float, default=7.5)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir/"corpus.jsonl")
    queries = read_jsonl(data_dir/"queries.jsonl")
    doc_ids=[d["_id"] for d in corpus]
    titles=[(d.get("title") or "").strip() for d in corpus]
    texts=[(d.get("text") or "").strip() for d in corpus]
    n_docs=len(doc_ids); N=n_docs
    q_texts=[q.get("text") or "" for q in queries]
    k1,b = a.bm25_k1, a.bm25_b
    tw = a.title_w

    # tokenize both fields
    tt=[TOK.findall(t.lower()) for t in titles]
    xt=[TOK.findall(t.lower()) for t in texts]
    qt=[TOK.findall(t.lower()) for t in q_texts]

    df={}
    for terms in tt+xt:
        for term in set(terms): df[term]=df.get(term,0)+1
    vocab=sorted(df); tidx={t:i for i,t in enumerate(vocab)}; V=len(vocab)

    def build(field_terms):
        ln=np.array([len(x) for x in field_terms],dtype=np.float64)
        avg=max(ln.mean(),1e-9)
        idf=np.empty(V)
        for t,i in tidx.items():
            dft=df[t]; idf[i]=np.log(1.0+(N-dft+0.5)/(dft+0.5))
        rows,cols,vals=[],[],[]
        for di,terms in enumerate(field_terms):
            if not terms: continue
            counts={}
            for term in terms: counts[term]=counts.get(term,0)+1
            dlf=1.0-b+b*ln[di]/avg
            for term,c in counts.items():
                ti=tidx[term]; denom=c+k1*dlf
                vals.append((c*(k1+1.0))/denom); rows.append(ti); cols.append(di)
        M=csr_matrix((np.array(vals),(np.array(rows,dtype=np.int64),np.array(cols,dtype=np.int64))),shape=(V,n_docs))
        return M.multiply(idf[:,None])

    At=build(tt); Ax=build(xt)

    # query matrix
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
    scores=np.zeros((len(queries),n_docs))
    for i in range(0,len(queries),batch):
        chunk=qt[i:i+batch]; Q=qmat(chunk)
        S=(Q@Ax).toarray() + tw*(Q@At).toarray()  # NOTE: text weight =1, title weight =tw (adds) 
        scores[i:i+len(chunk)]=S
    rankings={}
    k=min(a.top_k,n_docs)
    for i,q in enumerate(queries):
        order=scores[i].argsort()[::-1][:k]
        rankings[q["_id"]]=[doc_ids[j] for j in order]
    torch.save({"rankings":rankings}, Path(a.out))
    print(f"ranked {len(rankings)} queries title_w={tw}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
