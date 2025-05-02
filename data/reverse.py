import json
from pathlib import Path

INPUT_FILE  = Path("/home/zs7353/RobustRAG/data/realtimeqa_sorted.json")     # path to your original file
OUTPUT_FILE = Path("/home/zs7353/RobustRAG/data/realtimeqa_reversed.json")   # where you’d like the result saved

def reverse_top10(lst):
    """Return a list where the first 10 elements are reversed in-place."""
    if len(lst) <= 10:
        # Fewer than (or exactly) 10 docs: just reverse everything
        return list(reversed(lst))
    # Split: reverse the first 10, leave the rest unchanged
    return list(reversed(lst[:10])) + lst[10:]

def main() -> None:
    # 1) load the existing data ----------------------------------------------
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)      # data is a list of question‐level dicts
    
    # 2) reverse the order of documents in every “context” list --------------
    for item in data:
        if isinstance(item.get("context"), list):
            item["context"] = reverse_top10(item["context"])
    
    # 3) write the transformed data back out ---------------------------------
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
