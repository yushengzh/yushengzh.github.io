#!/usr/bin/env python3
"""v1: Qwen3-Embedding-8B dense retrieval, no endswith filter, TOP_K=10."""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

TASK = "Given a financial question, retrieve user replies that best answer the question"
TOP_K = 10
MAX_LEN = 4096
CORPUS_EMB = "/assets/corpus_embeddings.pt"

def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_model():
    tok = AutoTokenizer.from_pretrained("/assets/Qwen3-Embedding-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "/assets/Qwen3-Embedding-8B", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return tok, model

def embed(texts, tok, model, budget=8000):
    """Embed texts using last-token pooling. budget = rough token cap per batch."""
    # Sort by length (longest first) to minimize padding waste
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    embs = [None] * len(texts)
    i = 0
    while i < len(order):
        first = len(texts[order[i]])
        n = min(256, budget // max(1, first))
        batch_idx = order[i:i + n]
        i += n
        # Tokenize batch
        enc = tok([texts[j] for j in batch_idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        # If batch too large, shrink
        while enc["input_ids"].numel() > budget:
            n = n // 2
            if n < 1: n = 1
            batch_idx = order[i - n:i]
            enc = tok([texts[j] for j in batch_idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc.cuda())
        h = out.last_hidden_state  # [B, L, H]
        # Last-token pooling (left padding would be ideal; use actual last token)
        # For right-padded sequences, the last real token is at position (length-1)
        # We use the last non-pad token for each sequence
        attn = enc["attention_mask"].cuda()  # [B, L]
        last_idx = attn.sum(dim=1) - 1  # [B] index of last real token
        pooled = h[torch.arange(h.size(0)), last_idx]  # [B, H]
        pooled = F.normalize(pooled.float(), dim=-1).half()
        for j, idx in enumerate(batch_idx):
            embs[idx] = pooled[j]
    return torch.stack(embs, dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
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

    D = pre["emb"].cuda()  # [n_docs, 4096] float16
    rankings = {}
    for s in range(0, len(queries), 256):
        top = (Q[s:s+256] @ D.T).float().topk(min(TOP_K, len(ids)), dim=1).indices.cpu()
        for q, row in zip(queries[s:s+256], top.tolist()):
            rankings[q["_id"]] = [ids[j] for j in row]
    torch.save({"rankings": rankings}, a.out)
    print(f"ranked {len(rankings)} queries over {len(ids)} docs ({time.time() - t0:.0f}s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
