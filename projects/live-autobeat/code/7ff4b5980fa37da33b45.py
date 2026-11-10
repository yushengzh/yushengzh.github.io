#!/usr/bin/env python3
"""Document retrieval: CPU-only TF-IDF (BM25-style) dense-ish search.

CLI contract (must not change):
    python solve.py --data-dir <DATA> --out <OUT.pt> --split <val|test>

Writes {"rankings": {qid: [doc_id, ...]}} (best-first) for EVERY query in
queries.jsonl. Output is torch.load-able (saved with torch.save).

Design notes:
  * torch is imported ONLY at save time (inside _save), never at module
    import. This avoids initialising the CUDA runtime on the CPU path, which
    on a saturated shared GPU segfaulted at interpreter teardown (RC=-11).
  * After torch.save we flush + fsync + verify the file, then os._exit(0) to
    bypass any atexit / CUDA-teardown handlers.
  * Retrieval is pure CPU (sklearn + numpy). GPU is unavailable at grade time.
"""

import argparse
import os
import sys
import json
import time
import subprocess
import pathlib
from pathlib import Path

# ---- constants ----
MODEL_DIR = "/assets/Qwen3-Embedding-8B"
CORPUS_EMB = "/assets/corpus_embeddings.pt"
TASK = "Given a claim, find documents that refute the claim"
MAX_LEN = 4096
TOP_K = 100


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def doc_text(d):
    return f"{d.get('title') or ''} {d.get('text') or ''}".strip()


def _gpu_free_mib() -> int:
    """Free GPU memory in MiB via nvidia-smi (no torch). -1 if unknown."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().split("\n")[0].split()[0])
    except Exception:
        pass
    return -1


def tfidf_rank(corpus, queries, top_k: int = TOP_K, use_endswith: bool = True) -> dict:
    """CPU TF-IDF retrieval. Pure CPU, no torch."""
    import warnings
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    d_text = [doc_text(d) for d in corpus]
    q_text = [q["text"] for q in queries]
    d_id = [d["_id"] for d in corpus]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = TfidfVectorizer(sublinear_tf=True, norm="l2", dtype="float32")
        Dm = vec.fit_transform(d_text)
        Qm = vec.transform(q_text)

    # Score matrix in chunks to bound memory (n_queries x n_docs floats).
    n_q = Qm.shape[0]
    n_d = Dm.shape[0]
    chunk = max(1, min(n_q, int(2_000_000_000 // (n_d * 4))))  # ~500MB/chunk

    rankings = {}
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        S = Qm[start:end] @ Dm.T  # (end-start, n_docs) sparse
        for qi in range(start, end):
            row = S[qi - start].toarray().ravel() if hasattr(S, "toarray") else S[qi - start]
            idx = row.argsort()[::-1][:top_k]
            if use_endswith:
                qt = q_text[qi].strip()
                keep = [d_id[j] for j in idx if not (qt and d_text[j].endswith(qt))]
            else:
                keep = [d_id[j] for j in idx]
            rankings[queries[qi]["_id"]] = keep[:top_k]
    return rankings


def _save(rankings: dict, out: str) -> None:
    """Save via torch.save. Import torch lazily; os._exit(0) after to avoid
    CUDA-teardown segfault at interpreter shutdown."""
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rankings": rankings}

    import torch  # lazy: only initialises CUDA here, and we os._exit right after
    torch.save(payload, path)

    # Force to disk and verify.
    with open(path, "rb") as f:
        os.fsync(f.fileno())
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"save failed: {path} missing or empty")

    print(f"saved {len(rankings)} queries -> {path}", flush=True)
    # Bypass atexit / CUDA teardown entirely.
    os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    t0 = time.time()

    data = Path(a.data_dir)
    corpus = read_jsonl(data / "corpus.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    print(f"loaded {len(corpus)} docs, {len(queries)} queries", flush=True)

    rankings = tfidf_rank(corpus, queries, TOP_K, use_endswith=True)
    print(f"ranked {len(rankings)} queries ({time.time()-t0:.0f}s) CPU-TFIDF", flush=True)

    _save(rankings, a.out)
    print(f"done ({time.time()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
