import json, sys, asyncio
from pathlib import Path
from tqdm.asyncio import tqdm as atqdm
from mxbai_rerank import MxbaiRerankV2 

# ----------------------------------------------------------------------
IN_PATH  = Path("/home/zs7353/RobustRAG/data/realtimeqa.json")
OUT_PATH = Path("/home/zs7353/RobustRAG/data/realtimeqa_sorted.json")
MODEL_ID = "/scratch/gpfs/zs7353/mxbai-rerank-large-v2"
# ----------------------------------------------------------------------


def passage_text(ctx) -> str:
    if isinstance(ctx, dict):
        if ctx.get("text"):
            return ctx["text"]
        return (ctx.get("title", "") + " " + ctx.get("link", "")).strip()
    return str(ctx) if ctx is not None else ""


def main():
    if not IN_PATH.exists():
        sys.exit(f"Input not found: {IN_PATH}")

    data = json.loads(IN_PATH.read_text())

    model = MxbaiRerankV2(
        MODEL_ID
    )

    for item in atqdm(data, desc="reranking"):
        query = item["question"]
        docs  = [passage_text(c) for c in item["context"]]
        if not docs:
            continue

        ranked = model.rank(
            query,
            docs,
            top_k=len(docs),
            return_documents=False,
            sort=True,
        )

        # reorder the original context list
        item["context"] = [item["context"][r.index] for r in ranked]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("Saved sorted dataset →", OUT_PATH.resolve())


if __name__ == "__main__":
    main()