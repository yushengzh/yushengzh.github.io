#!/usr/bin/env python3
"""BM25+ fused score. Same CLI as methods/main/solve.py."""
from __future__ import annotations
import argparse, json, re, math
from pathlib import Path
import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
def toks(s): return TOKEN.findall((s or "").lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.7)
    ap.add_argument("--tw", type=float, default=1.0)
    ap.add_argument("--w", type=float, default=1.0)     # weight on 1/sqrt(df) term
    ap.add_argument("--p", type=float, default=0.5)     # df power in 1/df^p
    a = ap.parse_args()
    data = Path(a.data_dir)
    corpus = read_jsonl(data/"corpus.jsonl"); queries = read_jsonl(data/"queries.jsonl")
    doc_ids = [d["_id"] for d in corpus]; n = len(corpus)

    def build(fld):
        tfm = {}; dl = np.zeros(n, np.float32)
        for di, d in enumerate(corpus):
            src = (d.get("title") or "") if fld=="t" else (d.get("text") or "")
            t = toks(src); dl[di]=len(t); c={}
            for w in t: c[w]=c.get(w,0)+1
            for w,tf in c.items(): tfm.setdefault(w,[]).append((di,tf))
        rows={}; wmap={}
        for w,lst in tfm.items():
            df=len(lst)
            idf=math.log(1.0+(n-df+0.5)/(df+0.5))
            wdf=1.0/(df**a.p)
            wmap[w]=(idf, wdf)
            rows[w]=(np.array([p[0] for p in lst],np.int64), np.array([p[1] for p in lst],np.float32))
        avg=float(dl.mean()) if n else 1.0
        return rows, wmap, dl, avg
    F={f:build(f) for f in ("x","t")}
    k1,b,tw,W = a.k1,a.b,a.tw,a.w
    rankings={}
    for q in queries:
        s=np.zeros(n,np.float32)
        for w in set(toks(q.get("text") or "")):
            for fld,wt in (("x",1.0),("t",tw)):
                rows,wmap,dl,avg=F[fld]
                r=rows.get(w)
                if r is None: continue
                di,tf=r
                idf,wdf=wmap[w]
                norm=k1*(1.0-b+b*dl[di]/avg)
                score=W*wdf+(1.0)*idf*np.log1p(tf)
                s[di]+=wt*score/norm
        idx=np.argpartition(-s,a.top_k)[:a.top_k]
        idx=idx[np.argsort(-s[idx])]
        rankings[q["_id"]]=[doc_ids[j] for j in idx]
        del s
    import torch; torch.save({"rankings":rankings}, Path(a.out)); print("wrote",len(rankings))
if __name__=="__main__": raise SystemExit(main())
