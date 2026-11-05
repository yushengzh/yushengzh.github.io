#!/usr/bin/env python3
"""Document retrieval: two-field BM25 with light suffix-stemming (v12)."""
import argparse, json, math, re, os, sys
from collections import Counter, defaultdict

TOK = re.compile(r"(?u)\b\w+\b")

SUFFIXES = ('ational','tional','iveness','fulness','ousness',
           'isation','ization','ically','ally','ingly','ously',
           'ation','ition','ment','ness','able','ible','ful',
           'ous','ive','ing','ed','ly','es','s')

def stem_word(w):
    if len(w) <= 3:
        return w
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            return w[:-len(suf)]
    return w

def tokenize(text):
    return [stem_word(t) for t in TOK.findall(text.lower())]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test', choices=['val','test'])
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--bm25-k1', type=float, default=2.5)
    ap.add_argument('--bm25-b', type=float, default=1.25)
    ap.add_argument('--title-w', type=float, default=7.5)
    args = ap.parse_args()

    # Load corpus
    docs = []
    with open(os.path.join(args.data_dir, 'corpus.jsonl')) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    # Load queries
    queries = []
    with open(os.path.join(args.data_dir, 'queries.jsonl')) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    N = len(docs)

    # Tokenize all docs
    title_terms = [tokenize(d.get('title', '')) for d in docs]
    text_terms = [tokenize(d.get('text', '')) for d in docs]

    # Global IDF: df over union of title+text per doc
    df = Counter()
    for i in range(N):
        union = set(title_terms[i]) | set(text_terms[i])
        for t in union:
            df[t] += 1
    idf = {t: math.log(1.0 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    # Per-field avg doc length
    At_lens = [len(t) for t in title_terms]
    Ax_lens = [len(t) for t in text_terms]
    At_avg = sum(At_lens) / N if N else 1.0
    Ax_avg = sum(Ax_lens) / N if N else 1.0

    # Precompute per-term doc scores for each field
    # For text field: term_scores_text[t] = {doc_idx: score}
    # score = idf[t] * tf * (k1+1) / (tf + k1*dlf)
    k1 = args.bm25_k1
    b = args.bm25_b
    tw = args.title_w
    topk = args.top_k

    term_scores_text = defaultdict(dict)
    for i in range(N):
        if not text_terms[i]:
            continue
        dlf = 1 - b + b * (Ax_lens[i] / Ax_avg if Ax_avg > 0 else 1.0)
        tf = Counter(text_terms[i])
        for term, cnt in tf.items():
            s = idf[term] * cnt * (k1 + 1.0) / (cnt + k1 * dlf)
            term_scores_text[term][i] = s

    term_scores_title = defaultdict(dict)
    for i in range(N):
        if not title_terms[i]:
            continue
        dlf = 1 - b + b * (At_lens[i] / At_avg if At_avg > 0 else 1.0)
        tf = Counter(title_terms[i])
        for term, cnt in tf.items():
            s = idf[term] * cnt * (k1 + 1.0) / (cnt + k1 * dlf)
            term_scores_title[term][i] = s

    # Score queries
    rankings = {}
    doc_ids = [str(d['_id']) for d in docs]

    for q in queries:
        qid = str(q['_id'])
        qterms = tokenize(q.get('text', ''))
        if not qterms:
            rankings[qid] = doc_ids[:]
            continue
        qt = Counter(qterms)
        # Query weights
        scores = defaultdict(float)
        for term, cnt in qt.items():
            qw = 1.0 + math.log(cnt)
            # Text field
            if term in term_scores_text:
                for doc_i, s in term_scores_text[term].items():
                    scores[doc_i] += qw * s
            # Title field (weighted by tw)
            if term in term_scores_title:
                for doc_i, s in term_scores_title[term].items():
                    scores[doc_i] += tw * qw * s
        # Rank
        if scores:
            ranked_indices = sorted(scores.keys(), key=lambda i: -scores[i])
            ranked = [doc_ids[i] for i in ranked_indices[:topk]]
        else:
            ranked = []
        # Fill with unranked docs
        ranked_set = set(ranked)
        if len(ranked) < N:
            for i in range(N):
                if doc_ids[i] not in ranked_set:
                    ranked.append(doc_ids[i])
        rankings[qid] = ranked

    import torch
    torch.save({'rankings': rankings}, args.out)
    print(f"Done. {len(rankings)} queries, {N} docs, top-k={topk}", file=sys.stderr)

if __name__ == '__main__':
    main()
