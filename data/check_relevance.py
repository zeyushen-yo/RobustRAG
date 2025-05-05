import json
import numpy as np
import matplotlib.pyplot as plt

def init_plot_font_size():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=MEDIUM_SIZE)

def contains_answer(context_item, answers):
    title = context_item.get("title", "").lower()
    text = context_item.get("text", "").lower()
    return any(answer.lower() in title or answer.lower() in text for answer in answers)

def compute_fraction_with_answer(dataset, d, name):
    json_path = f"./{dataset}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fractions = []
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
        init_plot_font_size()
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_fracs, cdf, marker='o', linestyle='-')

        # Compute mean and median
        mean_val = np.mean(fractions)
        median_val = np.median(fractions)
        plt.axvline(mean_val, color='red', linestyle='--')
        plt.text(mean_val + 0.01, 5, f'Mean = {mean_val:.2f}', color='red', rotation=90, va='bottom')

        plt.plot([0, median_val], [50, 50], linestyle=':', color='blue')  # horizontal part
        plt.plot([median_val, median_val], [0, 50], linestyle=':', color='blue')  # vertical part
        plt.text(median_val + 0.01, 5, f'Median = {median_val:.2f}', color='blue', rotation=90, va='bottom')

        plt.title(f"{name} (n={len(data)})")
        plt.xlabel("Fraction of contexts with a correct answer")
        plt.ylabel("Cumulative % of questions")
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 100])
        plt.tight_layout()
        plt.savefig(f"./fig_rel_dist/{d}/{dataset}_relevance_dist.png")
    else:
        print("No valid items to evaluate.")

compute_fraction_with_answer("realtimeqa", d="orig", name="RQA")
compute_fraction_with_answer("open_nq", d="orig", name="NQ")
compute_fraction_with_answer("simpleqa", d="orig", name="SQA")
compute_fraction_with_answer("triviaqa", d="orig", name="TQA")
compute_fraction_with_answer("realtimeqa_allrel", d="allrel", name="RQA")
compute_fraction_with_answer("open_nq_allrel", d="allrel", name="NQ")
compute_fraction_with_answer("simpleqa_allrel", d="allrel", name="SQA")
compute_fraction_with_answer("triviaqa_allrel", d="allrel", name="TQA")
compute_fraction_with_answer("realtimeqa_allrel_perturb", d="allrel_perturb", name="RQA")
compute_fraction_with_answer("open_nq_allrel_perturb", d="allrel_perturb", name="NQ")
compute_fraction_with_answer("simpleqa_allrel_perturb", d="allrel_perturb", name="SQA")
compute_fraction_with_answer("triviaqa_allrel_perturb", d="allrel_perturb", name="TQA")