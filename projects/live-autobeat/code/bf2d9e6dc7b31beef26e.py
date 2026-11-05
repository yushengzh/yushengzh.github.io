#!/usr/bin/env python3
"""Two-field BM25 retrieval (top-500 candidates) + GBDT re-ranker trained on the visible
(tune, h(qid)<7) qrels. Emits top-100 per query. Falls back to plain BM25 if qrels_dev.tsv
is absent or unusable (e.g. sealed test set with no dev split)."""
from __future__ import annotations
import argparse, json, re, math, hashlib, os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import torch

TOP_K = 100          # docs to emit per query
CAND_K = 500         # candidate window from BM25
TOK = re.compile(r"(?u)\b\w+\b")

def h(qid):
    return int(hashlib.md5(str(qid).encode()).hexdigest()[:8], 16) % 10

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
    ap.add_argument("--cand-k", type=int, default=CAND_K)
    ap.add_argument("--bm25-k1", type=float, default=2.5)
    ap.add_argument("--bm25-b", type=float, default=1.35)
    ap.add_argument("--title-w", type=float, default=7.5)
    ap.add_argument("--gbdt-est", type=int, default=120)
    ap.add_argument("--gbdt-depth", type=int, default=3)
    ap.add_argument("--gbdt-lr", type=float, default=0.1)
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
    cand_k = min(a.cand_k, n_docs)

    # tokenize
    tt=[TOK.findall(t.lower()) for t in titles]
    xt=[TOK.findall(t.lower()) for t in texts]
    qt=[TOK.findall(t.lower()) for t in q_texts]

    df={}
    for terms in tt+xt:
        for term in set(terms): df[term]=df.get(term,0)+1
    vocab=sorted(df); tidx={t:i for i,t in enumerate(vocab)}; V=len(vocab)
    idf=np.empty(V)
    for t,i in tidx.items():
        dft=df[t]; idf[i]=np.log(1.0+(N-dft+0.5)/(dft+0.5))

    def build(field_terms):
        ln=np.array([len(x) for x in field_terms],dtype=np.float64)
        avg=max(ln.mean(),1e-9)
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
        S=(Q@Ax).toarray() + tw*(Q@At).toarray()
        scores[i:i+len(chunk)]=S

    # title/text lengths + doc term sets (precompute for features)
    tlen=np.array([len(x) for x in tt],dtype=np.float64)
    xlen=np.array([len(x) for x in xt],dtype=np.float64)
    tsets=[set(x) for x in tt]
    xsets=[set(x) for x in xt]

    # doc-id -> index
    dmap={d:di for di,d in enumerate(doc_ids)}
    queries_index={q["_id"]:i for i,q in enumerate(queries)}

    # --- attempt to load qrels (visible dev) for training ---
    qrels_path=data_dir/"qrels_dev.tsv"
    train_ok=False
    if qrels_path.exists():
        try:
            import csv
            rels={}
            with open(qrels_path, encoding="utf-8") as fh:
                rd=csv.reader(fh, delimiter="\t")
                header=next(rd)
                for row in rd:
                    if len(row)<3: continue
                    qid,cid,sc=row[0].strip(),row[1].strip(),row[2].strip()
                    try: s=float(sc)
                    except: continue
                    if s<=0: continue
                    rels.setdefault(qid,[]).append(cid)
            # keep only recoverable, tune-slice (h<7) queries with exactly one rel doc in corpus
            train_qids=[]
            for qid,rl in rels.items():
                if len(rl)!=1: continue
                if qid not in queries_index: continue
                di=dmap.get(rl[0])
                if di is None: continue
                if h(qid)<7:
                    train_qids.append(qid)
            if len(train_qids)>=20:
                train_ok=True
        except Exception:
            train_ok=False

    if not train_ok:
        # fallback: plain BM25 top-k
        rankings={}
        k=min(a.top_k,n_docs)
        for i,q in enumerate(queries):
            order=scores[i].argsort()[::-1][:k]
            rankings[q["_id"]]=[doc_ids[j] for j in order]
        torch.save({"rankings":rankings}, Path(a.out))
        print(f"no-qrels fallback: ranked {len(rankings)} queries (plain BM25)")
        return 0

    # --- GBDT re-ranker ---
    def feat(qi,di):
        s=scores[qi,di]
        qs=qt[qi]
        ntov=len(set(qs) & tsets[di])
        nxov=len(set(qs) & xsets[di])
        # max idf among query terms present in doc title or text
        mi=0.0
        qset=qs
        union=tsets[di]|xsets[di]
        for term in qset:
            if term in union and term in tidx:
                v=idf[tidx[term]]
                if v>mi: mi=v
        norm=s/(scores[qi].mean()+1e-9)
        return [s, ntov, nxov, mi, xlen[di], tlen[di], norm]

    # top-cand_k indices per query
    cand_idx=np.argsort(scores,axis=1)[:,::-1][:, :cand_k]  # (nq, cand_k)

    from sklearn.ensemble import GradientBoostingClassifier
    X=[]; y=[]
    for qid in train_qids:
        qi=queries_index[qid]
        rel_di=dmap[rels[qid][0]]
        cand=cand_idx[qi]
        for di in cand:
            X.append(feat(qi,int(di)))
            y.append(1 if int(di)==rel_di else 0)
    X=np.asarray(X); y=np.asarray(y)
    clf=GradientBoostingClassifier(n_estimators=a.gbdt_est,max_depth=a.gbdt_depth,
                                    learning_rate=a.gbdt_lr,random_state=0).fit(X,y)

    # re-rank all queries
    rankings={}
    k_out=min(a.top_k,n_docs)
    for i,q in enumerate(queries):
        cand=cand_idx[i]
        fs=clf.predict_proba(np.asarray([feat(i,int(di)) for di in cand]))[:,1]
        order=cand[np.argsort(fs)[::-1]][:k_out]
        rankings[q["_id"]]=[doc_ids[j] for j in order]
    torch.save({"rankings":rankings}, Path(a.out))
    print(f"GBDT rerank: {len(train_qids)} train qids, ranked {len(rankings)} queries")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
