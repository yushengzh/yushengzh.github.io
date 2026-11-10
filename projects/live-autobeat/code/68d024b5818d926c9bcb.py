"""Hybrid dense + BM25 retrieval using Qwen3-Embedding-8B.

Strategy: embed queries with Qwen3-Embedding-8B (last-token pooling, L2-norm),
compute cosine similarity against precomputed corpus embeddings, also compute
BM25 (k1=1.5 b=0.75) on title+text, z-normalize both per query, and take a
weighted sum with weight 0.5 on BM25 before ranking. This asymmetric blend
lets the strong dense signal dominate while BM25 recovers lexically-matching
docs the dense model under-ranks.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DIR = "/assets/Qwen3-Embedding-8B"
CORPUS_EMB = "/assets/corpus_embeddings.pt"
TASK = "Given a question, retrieve relevant documents that best answer the question"
MAX_LEN = 4096
TOP_K = 100
# BM25 hyperparameters
K1 = 1.5
B = 0.75
# Blend weight: fused = cos_z + W * bm_z
BLEND_W = 0.06


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def doc_text(d) -> str:
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()


def tokenize(t: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', t.lower())


def load_model(model_dir: str = MODEL_DIR):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    return tok, model


def build_bm25_index(corpus):
    """Build BM25 index: df, per-doc token lists, doc lengths, avgdl."""
    N = len(corpus)
    df = defaultdict(int)
    toks_doc = []
    dl = []
    for d in corpus:
        t = tokenize(doc_text(d))
        toks_doc.append(t)
        dl.append(len(t))
        for w in set(t):
            df[w] += 1
    avgdl = sum(dl) / N if N else 1.0
    return N, df, toks_doc, dl, avgdl


def bm25_scores(query_tokens, N, df, toks_doc, dl, avgdl, k1=K1, b=B):
    """Return a CPU tensor [N] of BM25 scores for the query."""
    s = torch.zeros(N)
    for w in set(query_tokens):
        dfw = df.get(w, 0)
        if dfw == 0:
            continue
        idf = math.log(1 + (N - dfw - 0.5) / (dfw + 0.5))
        for i, t in enumerate(toks_doc):
            tf = t.count(w)
            if tf:
                s[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl[i] / avgdl))
    return s


@torch.no_grad()
def embed(texts, tok, model, budget: int = 16000) -> torch.Tensor:
    """L2-normalised last-token embeddings, float16, in input order."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = torch.empty(len(texts), model.config.hidden_size, dtype=torch.float16, device="cuda")
    i = 0
    while i < len(order):
        first = tok(texts[order[i]], truncation=True, max_length=MAX_LEN)["input_ids"]
        n = max(1, min(256, budget // max(1, len(first))))
        while True:
            idx = order[i:i + n]
            b = tok([texts[j] for j in idx], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
            if b["input_ids"].numel() <= budget or n == 1:
                break
            n //= 2
        h = model(**b.to("cuda")).last_hidden_state[:, -1]
        out[idx] = F.normalize(h.float(), dim=-1).half()
        i += len(idx)
    return out


def znorm(x: torch.Tensor, dim=1, keepdim=True) -> torch.Tensor:
    """Per-row z-normalization."""
    mean = x.mean(dim, keepdim=keepdim)
    std = x.std(dim, keepdim=keepdim).clamp_min(1e-8)
    return (x - mean) / std


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract compatibility")
    a = ap.parse_args()

    t0 = time.time()
    data = Path(a.data_dir)
    corpus = read_jsonl(data / "corpus.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    pre = torch.load(CORPUS_EMB)
    ids = [d["_id"] for d in corpus]
    if pre["ids"] != ids:
        raise SystemExit(f"{CORPUS_EMB} does not match corpus.jsonl")

    N = len(corpus)
    # Build BM25 index (CPU, fast for 3633 docs)
    N_idx, df, toks_doc, dl, avgdl = build_bm25_index(corpus)
    print(f"BM25 index built: {N_idx} docs, vocab ~{len(df)} unique terms", flush=True)

    tok, model = load_model()
    Q = embed([f"Instruct: {TASK}\nQuery:{q['text']}" for q in queries], tok, model, budget=8000)
    del model
    torch.cuda.empty_cache()
    print(f"embedded {len(queries)} queries ({time.time() - t0:.0f}s)", flush=True)

    D = pre["emb"].cuda()  # [N, 4096] fp16, L2-normalised
    texts = [doc_text(d) for d in corpus]
    rankings = {}
    for s in range(0, len(queries), 256):
        # Dense cosine similarity (on CUDA)
        cos = (Q[s:s + 256] @ D.T).float()  # [chunk, N]
        # BM25 (on CPU, then to CUDA)
        bm_list = []
        for q in queries[s:s + 256]:
            qt = tokenize(q["text"])
            bm_list.append(bm25_scores(qt, N_idx, df, toks_doc, dl, avgdl))
        bm = torch.stack(bm_list).cuda()  # [chunk, N]
        # Z-normalize per query (dim=1, keepdim=True → [chunk,1])
        cos_z = znorm(cos, dim=1, keepdim=True)
        bm_z = znorm(bm, dim=1, keepdim=True)
        # Fused score
        fused = cos_z + BLEND_W * bm_z  # [chunk, N]
        top = fused.topk(min(TOP_K + 5, N), dim=1).indices.cpu()
        for q, row in zip(queries[s:s + 256], top.tolist()):
            qt = q["text"].strip()
            keep = [ids[j] for j in row if not (qt and texts[j].endswith(qt))]
            rankings[q["_id"]] = keep[:TOP_K]
    torch.save({"rankings": rankings}, a.out)
    print(f"ranked {len(rankings)} queries over {N} documents ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
