#!/usr/bin/env python3
"""BM25 lexical retrieval with Porter stemming and title boosting.

Improvement over the v0 TF-IDF baseline: Robertson-Sparck BM25 (tunable k1/b)
over a tokenized index, a hand-inlined Porter stemmer (no sklearn stemmer
dependency), a title boost, and a small stopword set. The index is
tokenization-only (train-independent), so it generalizes identically to the
sealed split. Ranking all queries keeps the val and test paths identical.

Output contract (unchanged): a dict {"rankings": {qid: [doc_id, ...]}} saved
as .pt via torch. Every query in queries.jsonl is covered.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

TOP_K = 100
K1 = 1.2
B = 0.75
TB = 3  # title boost
STEM = True

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOP = set("""a an and are as at be by for from has have in is it its of on or
that the to was were will with your you i we they he she this these those what
when where who which how why not no yes but so if can could do does did would
should may might must about into over under out up down off again more most
other some such only own same than then there here just also each few both very
like us am been being get got let make makes way day week month year thing
things one two three people person man woman child children use used using
water food foods""".split())


# ---------------------------------------------------------------------------
# Porter stemmer (compact stdlib-only, Martino Tartari port). Inlined so the
# submission needs no external stemmer module.
# ---------------------------------------------------------------------------

def _con(s, i):
    ch = s[i]
    if ch in "aeiou":
        return 0
    if ch == "y":
        return 1 if i == 0 else 0
    return 1


def _has_vowel(s):
    for i in range(len(s)):
        if not _con(s, i):
            return True
    return False


def _m(s):
    n = 0
    i = 0
    L = len(s)
    while True:
        while i < L and _con(s, i):
            i += 1
        if i >= L:
            break
        n += 1
        while i < L and not _con(s, i):
            i += 1
    return n


def _doublec(s):
    return len(s) >= 2 and s[-1] == s[-2] and _con(s, len(s) - 1)


def _cvc(s):
    L = len(s)
    if L < 3 or not _con(s, L - 1) or _con(s, L - 2) or not _con(s, L - 3):
        return False
    if s[L - 1] in "wxy":
        return False
    return True


def _endsv(s, suf):
    if not s.endswith(suf):
        return False
    return _con(s, len(s) - len(suf) - 1)


def _replace(s, suf, rep):
    return s[:len(s) - len(suf)] + rep


def step1a(w):
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if not w.endswith("ss") and w.endswith("s"):
        return w[:-1]
    return w


def step1b(w):
    flag = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            return w[:-1]
        return w
    if w.endswith("ed"):
        if _has_vowel(w[:-2]):
            w = w[:-2]
            flag = True
    elif w.endswith("ing"):
        if _has_vowel(w[:-3]):
            w = w[:-3]
            flag = True
    if flag:
        if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
            return w + "e"
        if _doublec(w) and not (w.endswith("l") or w.endswith("s") or w.endswith("z")):
            return w[:-1]
        if _m(w) == 1 and _cvc(w):
            return w + "e"
    return w


def step1c(w):
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


def step2(w, pairs):
    if _m(w) <= 0:
        return w
    for suf, rep in pairs:
        if _endsv(w, suf):
            return _replace(w, suf, rep)
    return w


def step3(w, pairs):
    if _m(w) <= 0:
        return w
    for suf, rep in pairs:
        if w.endswith(suf):
            return _replace(w, suf, rep)
    return w


def step4(w):
    if _m(w) <= 1:
        return w
    for suf in ("alize", "icate", "ical", "ful", "ness", "ent", "ement", "ment",
                "ion", "ity", "ize", "iti", "ous", "ive", "al"):
        if w.endswith(suf):
            if suf == "ion" and len(w) > 3 and w[-4] in "se":
                return w[:-len(suf)]
            return w[:-len(suf)]
    return w


def step5a(w):
    if w.endswith("e"):
        m = _m(w)
        if (m > 1) or (m == 1 and not _cvc(w[:-1])):
            return w[:-1]
    return w


def step5b(w):
    if _m(w) > 1 and _doublec(w) and w.endswith("l"):
        return w[:-1]
    return w


S2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
      ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
      ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
      ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
      ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")]
S3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
      ("ical", "ic"), ("ful", "")]


def stem(w):
    w = w.lower()
    if len(w) < 3:
        return w
    w = step1a(w)
    w = step1b(w)
    w = step1c(w)
    w = step2(w, S2)
    w = step3(w, S3)
    w = step4(w)
    w = step5a(w)
    w = step5b(w)
    return w


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tok(text):
    out = []
    for w in TOKEN.findall(text.lower()):
        if w in STOP:
            continue
        if STEM:
            w = stem(w)
        if w:
            out.append(w)
    return out


def build(corpus):
    N = len(corpus)
    tfs = []
    dl = []
    df = {}
    for d in corpus:
        title = d.get("title") or ""
        text = d.get("text") or ""
        tf = {}
        for w in tok(title):
            tf[w] = tf.get(w, 0) + TB
        for w in tok(text):
            tf[w] = tf.get(w, 0) + 1
        tfs.append(tf)
        dl.append(sum(tf.values()))
        for w in tf:
            df[w] = df.get(w, 0) + 1
    avgdl = sum(dl) / N if N else 0.0
    idf = {w: math.log(1.0 + (N - nf + 0.5) / (nf + 0.5)) for w, nf in df.items()}
    postings = {}
    for idx, tf in enumerate(tfs):
        for w, ftf in tf.items():
            postings.setdefault(w, []).append((idx, ftf))
    return tfs, dl, idf, postings, avgdl, N


def score_query(qtext, idx):
    tfs, dl, idf, postings, avgdl, N = idx
    qt = tok(qtext)
    scores = {}
    for w in set(qt):
        idf_w = idf.get(w)
        if idf_w is None:
            continue
        for (idx2, ftf) in postings[w]:
            dlx = dl[idx2]
            s = idf_w * (ftf * (K1 + 1.0)) / (ftf + K1 * (1.0 - B + B * dlx / avgdl))
            scores[idx2] = scores.get(idx2, 0.0) + s
    return scores


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract compatibility; retrieval ranks all queries")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "queries.jsonl")

    idx = build(corpus)
    n = len(corpus)
    topk = min(a.top_k, n)
    fallback = [d["_id"] for d in corpus[:topk]]

    rankings = {}
    for qi, q in enumerate(queries):
        sc = score_query(q.get("text") or "", idx)
        if sc:
            order = sorted(sc.items(), key=lambda x: -x[1])[:topk]
            rankings[q["_id"]] = [corpus[i]["_id"] for i, _ in order]
        else:
            rankings[q["_id"]] = list(fallback)
        if (qi + 1) % 512 == 0:
            print(f"ranked {qi + 1}/{len(queries)} queries", flush=True)

    torch_free_save(Path(a.out), {"rankings": rankings})
    print(f"wrote rankings for {len(rankings)} queries -> {a.out}")
    return 0


def torch_free_save(path: Path, obj) -> None:
    """Keep the submission format identical to the graph tasks (.pt via torch)."""
    import torch
    torch.save(obj, path)


if __name__ == "__main__":
    raise SystemExit(main())
