#!/usr/bin/env python3
"""
simpleqa_rerank_by_answer.py
Sort contexts so that passages containing a correct answer string appear first.

No external models needed – pure string matching.
"""

import json, unicodedata, sys
from pathlib import Path
from typing import List

# ----------------------------------------------------------------------
IN_PATH  = Path("/home/zs7353/RobustRAG/data/triviaqa.json")
OUT_PATH = Path("/home/zs7353/RobustRAG/data/triviaqa_sorted_answer_based.json")
# ----------------------------------------------------------------------


def normalize(text: str) -> str:
    """Unicode-normalize and lowercase for robust substring matches."""
    return unicodedata.normalize("NFKC", text).lower()


def passage_text(ctx) -> str:
    """Return a text string from a context entry (dict or str)."""
    if isinstance(ctx, dict):
        if "text" in ctx and ctx["text"]:
            return ctx["text"]
        # fall back to whatever else is present
        return " ".join(str(ctx.get(k, "")) for k in ("title", "link")).strip()
    return str(ctx) if ctx is not None else ""


def contains_answer(passage: str, answers: List[str]) -> bool:
    p_norm = normalize(passage)
    return any(normalize(ans) in p_norm for ans in answers)


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"Input file not found: {IN_PATH}")

    data = json.loads(IN_PATH.read_text())

    for item in data:
        answers  = item["correct answer"]      # list of strings
        contexts = item["context"]

        # stable sort: passages with answer first, original order otherwise
        contexts.sort(
            key=lambda ctx: not contains_answer(passage_text(ctx), answers)
        )
        # inplace replacement already done because list.sort() mutates

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved answer-sorted dataset → {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
