import json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def contains_answer(context_item, answers):
    title = context_item.get("title", "").lower()
    text = context_item.get("text", "").lower()
    return any(answer.lower() in title or answer.lower() in text for answer in answers)

def make_all_relevant(dataset):
    json_path = f"./{dataset}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fractions = []
    modified_data = []
    for item in data:
        context_list = item.get("context", [])
        correct_answers = item.get("correct answer", [])
        if not context_list:
            continue

        # Step 1: Determine which contexts match (once)
        match_flags = [contains_answer(c, correct_answers) for c in context_list]
        matching_contexts = [context_list[i] for i, is_match in enumerate(match_flags) if is_match]

        # Step 2: Calculate and store matching fraction
        match_count = len(matching_contexts)
        fraction = match_count / len(context_list)
        fractions.append(fraction)

        # Step 3: Create modified version using round-robin replacement
        if matching_contexts:
            replacement_contexts = []
            rr_index = 0
            for is_match in match_flags:
                if is_match:
                    replacement_contexts.append(context_list[rr_index])
                else:
                    replacement_contexts.append(matching_contexts[rr_index % match_count])
                rr_index += 1

            modified_item = deepcopy(item)
            modified_item["context"] = replacement_contexts
        else:
            # No replacements available; leave item unchanged
            modified_item = deepcopy(item)

        modified_data.append(modified_item)

     # Save modified dataset
    with open(f"./{dataset}_allrel.json", "w", encoding="utf-8") as out_file:
        json.dump(modified_data, out_file, indent=2, ensure_ascii=False)

make_all_relevant("realtimeqa")
make_all_relevant("open_nq")
make_all_relevant("simpleqa")
make_all_relevant("triviaqa")