#!/usr/bin/env python3
"""Reference solution, retrieval: Qwen3-Embedding-8B dense retrieval.

Qwen3-Embedding (arXiv 2506.05176, Apache-2.0) is the open-source state of the art on
these BEIR datasets. Everything it needs is prepared before the session, because the
container has no network:

    /assets/Qwen3-Embedding-8B      the model (read-only)
    /assets/corpus_embeddings.pt    every document of corpus.jsonl, embedded offline by
                                    scripts/embed_beir_corpus.py with doc_text() and
                                    embed() below: {"ids": [...], "emb": float16 [N, 4096]}

Encoding the corpus takes up to ~55 min on a whole H100 (trec-covid, 171k documents),
more than the grading limit allows; the queries take seconds. So this program embeds
only the queries and ranks the precomputed document vectors by cosine similarity.

As published: last-token pooling (left padding), L2-normalised, at most 4096 tokens;
documents are title + text without an instruction; queries are
"Instruct: {task}\\nQuery:{text}" with the task description of the MTEB evaluation.

A document whose text is the query itself is never an answer (ArguAna has one for
every query, and the model ranks it first); such documents are dropped, as BEIR does.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DIR = "/assets/Qwen3-Embedding-8B"
CORPUS_EMB = "/assets/corpus_embeddings.pt"
TASK = "Given a question, retrieve relevant documents that best answer the question"   # the task description, set per dataset by scripts/make_beir_tasks.py
MAX_LEN = 4096
TOP_K = 100


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def doc_text(d) -> str:
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()


def load_model(model_dir: str = MODEL_DIR):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    return tok, model


@torch.no_grad()
def embed(texts, tok, model, budget: int = 16000) -> torch.Tensor:
    """L2-normalised last-token embeddings, float16, in input order.

    Batches hold at most `budget` tokens (padding included), longest texts first.
    """
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = torch.empty(len(texts), model.config.hidden_size, dtype=torch.float16, device="cuda")
    i = 0
    while i < len(order):
        first = tok(texts[order[i]], truncation=True, max_length=MAX_LEN)["input_ids"]
        n = max(1, min(256, budget // max(1, len(first))))
        while True:   # character order is not token order: shrink until the batch fits
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="accepted for contract compatibility; retrieval ranks all queries")
    a = ap.parse_args()

    t0 = time.time()
    data = Path(a.data_dir)
    corpus = read_jsonl(data / "corpus.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    pre = torch.load(CORPUS_EMB)
    ids = [d["_id"] for d in corpus]
    if pre["ids"] != ids:
        raise SystemExit(f"{CORPUS_EMB} does not match corpus.jsonl")

    tok, model = load_model()
    Q = embed([f"Instruct: {TASK}\nQuery:{q['text']}" for q in queries], tok, model, budget=8000)
    del model
    torch.cuda.empty_cache()
    print(f"embedded {len(queries)} queries ({time.time() - t0:.0f}s)", flush=True)

    D = pre["emb"].cuda()
    texts = [doc_text(d) for d in corpus]
    rankings = {}
    for s in range(0, len(queries), 256):
        top = (Q[s:s + 256] @ D.T).float().topk(min(TOP_K + 5, len(ids)), dim=1).indices.cpu()
        for q, row in zip(queries[s:s + 256], top.tolist()):
            qt = q["text"].strip()
            keep = [ids[j] for j in row if not (qt and texts[j].endswith(qt))]
            rankings[q["_id"]] = keep[:TOP_K]
    torch.save({"rankings": rankings}, a.out)
    print(f"ranked {len(rankings)} queries over {len(ids)} documents ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
