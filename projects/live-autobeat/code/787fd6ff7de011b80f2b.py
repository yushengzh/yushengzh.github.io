"""Hybrid dense + BM25 retrieval using Qwen3-Embedding-8B.

Strategy: embed queries with Qwen3-Embedding-8B (last-token pooling, L2-norm),
compute cosine similarity against precomputed corpus embeddings, also compute
BM25 (k1=1.5 b=0.75) on title+text, z-normalize both per query, and take a
weighted sum (BLEND_W on BM25) before ranking.

v3: score-neutral robustness refactor vs v2 (identical scoring math):
  - fast inverted-index BM25 (precomputed per-doc denom; validated maxabsdiff 4.6e-6)
  - MAX_LEN 4096->512, embed budget 8000->4000 (queries are <=50 tokens, so safe)
  - removed dead endswith self-match filter
  - CPU fallback for model load under GPU OOM pressure
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
MAX_LEN = 512
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
    device = "cuda"
    try:
        model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    except RuntimeError:
        # GPU OOM / unavailable: fall back to CPU (slower, but keeps output valid)
        device = "cpu"
        model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    return tok, model, device


def build_bm25_index(corpus):
    """Build fast inverted-index BM25: df, term->[(doc,tf)], per-doc denom, avgdl."""
    N = len(corpus)
    df = defaultdict(int)
    inv = defaultdict(list)
    dl = []
    for i, d in enumerate(corpus):
        toks = tokenize(doc_text(d))
        dl.append(len(toks))
        c = defaultdict(int)
        for w in toks:
            c[w] += 1
        for w, tf in c.items():
            inv[w].append((i, tf))
            df[w] += 1
    avgdl = sum(dl) / N if N else 1.0
    denom = torch.tensor([K1 * (1 - B) + K1 * B * l / avgdl for l in dl], dtype=torch.float32)
    return N, df, inv, denom, avgdl


def bm25_scores(query_tokens, N, df, inv, denom, k1=K1):
    """Return a CPU float32 tensor [N] of BM25 scores for the query."""
    s = torch.zeros(N, dtype=torch.float32)
    for w in set(query_tokens):
        dfw = df.get(w, 0)
        if dfw == 0:
            continue
        idf = math.log(1 + (N - dfw - 0.5) / (dfw + 0.5))
        for i, tf in inv[w]:
            s[i] += idf * (tf * (k1 + 1)) / (tf + denom[i])
    return s


@torch.no_grad()
def embed(texts, tok, model, device: str = "cuda", budget: int = 4000) -> torch.Tensor:
    """L2-normalised last-token embeddings, float16, in input order."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = torch.empty(len(texts), model.config.hidden_size, dtype=torch.float16, device=device)
    i = 0
    n = len(texts)
    while i < n:
        n_try = min(n, budget)
        while n_try > 1:
            idx = order[i:i + n_try]
            b = tok([texts[j] for j in idx], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
            if b["input_ids"].numel() <= budget or n_try == 1:
                break
            n_try //= 2
        idx = order[i:i + n_try]
        h = model(**b.to(device)).last_hidden_state[:, -1]
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
    N_idx, df, inv, denom, avgdl = build_bm25_index(corpus)
    print(f"BM25 index built: {N_idx} docs, vocab ~{len(df)} unique terms", flush=True)

    tok, model, device = load_model()
    Q = embed([f"Instruct: {TASK}\nQuery:{q['text']}" for q in queries], tok, model, device, budget=4000)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"embedded {len(queries)} queries ({time.time() - t0:.0f}s)", flush=True)

    D = pre["emb"].to(device)  # [N, 4096] fp16, L2-normalised
    rankings = {}
    for s in range(0, len(queries), 256):
        chunk_q = queries[s:s + 256]
        cos = (Q[s:s + 256] @ D.T).float()  # [chunk, N]
        bm_list = []
        for q in chunk_q:
            qt = tokenize(q["text"])
            bm_list.append(bm25_scores(qt, N_idx, df, inv, denom))
        bm = torch.stack(bm_list).to(device)  # [chunk, N]
        cos_z = znorm(cos, dim=1, keepdim=True)
        bm_z = znorm(bm, dim=1, keepdim=True)
        fused = cos_z + BLEND_W * bm_z  # [chunk, N]
        top = fused.topk(min(TOP_K + 5, N), dim=1).indices.cpu()
        for q, row in zip(chunk_q, top.tolist()):
            rankings[q["_id"]] = [ids[j] for j in row][:TOP_K]
    torch.save({"rankings": rankings}, a.out)
    print(f"ranked {len(rankings)} queries over {N} documents ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
