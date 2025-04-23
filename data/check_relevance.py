import json
import numpy as np
import matplotlib.pyplot as plt

def contains_answer(context_item, answers):
    title = context_item.get("title", "").lower()
    text = context_item.get("text", "").lower()
    return any(answer.lower() in title or answer.lower() in text for answer in answers)

def compute_fraction_with_answer(dataset):
    json_path = f"./data/{dataset}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fractions = []

    for item in data:
        context_list = item.get("context", [])
        correct_answers = item.get("correct answer", [])

        if not context_list:
            print(f"Question has empty context.")
            continue

        match_count = sum(contains_answer(c, correct_answers) for c in context_list)
        fraction = match_count / len(context_list)
        fractions.append(fraction)

    results = np.array(fractions)
    if results.size > 0:
        print(f"# items: {len(results)}")
        print(f"min fraction: {np.min(results):.2f}")
        print(f"max fraction: {np.max(results):.2f}")
        print(f"mean fraction: {np.mean(results):.2f}")
        print(f"SD: {np.std(results):.2f}")

        plt.figure(figsize=(8, 5))
        counts, bins, patches = plt.hist(fractions, bins=30, range=(0, 1), edgecolor='black', alpha=0.75, density=True)
        bin_width = bins[1] - bins[0]
        for patch, count in zip(patches, counts):
            height = count * bin_width * 100  # percentage height
            patch.set_height(height)

        plt.gca().set_ylim([0, 50])
        plt.gca().set_yticks(np.linspace(0, 50, 6))
        plt.gca().set_yticklabels([f"{int(x)}%" for x in np.linspace(0, 50, 6)])
        
        plt.title("Distribution of fractions of contexts that contain answers")
        plt.xlabel("Fraction of contexts with a correct answer (verbatim)")
        plt.ylabel("Percentage of questions")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./data/{dataset}_relevance_dist.png")
    else:
        print("No valid items to evaluate.")


compute_fraction_with_answer("realtimeqa")
compute_fraction_with_answer("open_nq")
compute_fraction_with_answer("simpleqa")
compute_fraction_with_answer("triviaqa")