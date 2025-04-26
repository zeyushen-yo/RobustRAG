#!/usr/bin/env python3
"""
simpleqa_rerank_bge_v2.py
Re-rank SimpleQA contexts with BGE-v2-m3 cross-encoder.
Robust to context entries that lack a 'text' field.
"""

import json, unicodedata, sys, torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# ----------------------------------------------------------------------
IN_PATH  = Path("/home/zs7353/RobustRAG/data/open_nq.json")
OUT_PATH = Path("/home/zs7353/RobustRAG/data/open_nq_sorted.json")
MODEL_ID = "/scratch/gpfs/zs7353/bge-reranker-v2-m3"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------------


def passage_text(ctx) -> str:
    """Return a string for scoring, regardless of context entry format."""
    if isinstance(ctx, dict):
        if ctx.get("text"):                      # preferred field
            return ctx["text"]
        # graceful fallback: concatenate whatever is available
        title = ctx.get("title", "")
        link  = ctx.get("link", "")
        return f"{title} {link}".strip()
    return str(ctx) if ctx is not None else ""   # raw string / other


def normalize(text: str) -> str:  # optional utility, not used here
    return unicodedata.normalize("NFKC", text).lower()


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"Input file not found: {IN_PATH}")

    data = json.loads(IN_PATH.read_text())

    reranker = CrossEncoder(MODEL_ID, device=DEVICE, max_length=512)

    for item in tqdm(data, desc="reranking"):
        q = item["question"]
        passages = [passage_text(c) for c in item["context"]]
        if len(passages) == 0:
            continue
        scores = reranker.predict(
            [(q, p) for p in passages],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # sort contexts by score ↓
        item["context"] = [
            ctx
            for _, ctx in sorted(
                zip(scores, item["context"]), key=lambda x: x[0], reverse=True
            )
        ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("Saved sorted dataset →", OUT_PATH.resolve())


if __name__ == "__main__":
    main()