import json
import numpy as np
import matplotlib.pyplot as plt

def contains_answer(context_item, answers):
    title = context_item.get("title", "").lower()
    text = context_item.get("text", "").lower()
    return any(answer.lower() in title or answer.lower() in text for answer in answers)

def compute_fraction_with_answer(dataset):
    json_path = f"./{dataset}.json"
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

        # Sort fractions and compute CDF
        num_questions = len(fractions)
        sorted_fracs = np.sort(fractions)
        cdf = np.arange(1, num_questions + 1) / num_questions * 100  # percentage

        # Plot CDF
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_fracs, cdf, marker='o', linestyle='-')
        plt.title(f"CDF of Matching Context Fractions (n={num_questions})")
        plt.xlabel("Fraction of Contexts with a Correct Answer")
        plt.ylabel("Cumulative % of Questions")
        plt.grid(True)
        plt.ylim([0, 100])
        plt.tight_layout()
        plt.savefig(f"./{dataset}_relevance_dist.png")
    else:
        print("No valid items to evaluate.")


compute_fraction_with_answer("realtimeqa")
compute_fraction_with_answer("open_nq")
compute_fraction_with_answer("simpleqa")
compute_fraction_with_answer("triviaqa")