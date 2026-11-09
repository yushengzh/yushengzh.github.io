#!/usr/bin/env python3
"""v2: Qwen3-Embedding-8B dense retrieval (fixed loader: AutoModel, no device_map).
No endswith filter, TOP_K=10 (nDCG@10 only reads top-10)."""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "/assets/Qwen3-Embedding-8B"
CORPUS_EMB = "/assets/corpus_embeddings.pt"
TASK = "Given a financial question, retrieve user replies that best answer the question"
TOP_K = 10
MAX_LEN = 4096


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
    return tok, model


@torch.no_grad()
def embed(texts, tok, model, budget: int = 8000) -> torch.Tensor:
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
        input_ids = b["input_ids"].cuda()
        attention_mask = b["attention_mask"].cuda()
        h = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, H] bfloat16
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = h[torch.arange(h.size(0)), last_idx]  # [B, H]
        pooled = F.normalize(pooled.float(), dim=-1).half()
        out[idx] = pooled
        del h, input_ids, attention_mask, b
        i += len(idx)
    return out


def main() -> int:
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
        top = (Q[s:s + 256] @ D.T).float().topk(min(TOP_K, len(ids)), dim=1).indices.cpu()
        for q, row in zip(queries[s:s + 256], top.tolist()):
            rankings[q["_id"]] = [ids[j] for j in row]
    torch.save({"rankings": rankings}, a.out)
    print(f"ranked {len(rankings)} queries over {len(ids)} documents ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
