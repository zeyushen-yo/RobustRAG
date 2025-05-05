import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_plot_font_size():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=10)
    plt.rc('figure', titlesize=SMALL_SIZE)

init_plot_font_size   
plt.rcParams.update({"font.size": 14})

map_label = {
    "vanilla": "Vanilla",
    "none": "Vanilla",
    "astuterag": "AstuteRAG",
    "instructrag_icl": "InstructRAG-ICL",
    "sampleMIS": "Sampling+MIS",
    "MIS": "MIS",
    "keyword": "RobustRAG\n(Keyword)",
}

dataset_labels = {
    "realtimeqa_sorted": "RQA",
    "realtimeqa_allrel_perturb": "RQA (cleaned)",
    "open_nq_sorted": "NQ",
    "open_nq_allrel_perturb": "NQ (cleaned)",
}

def get_accuracy(df):
    """Return accuracy from the first row of a results CSV."""
    return df.iloc[0]["acc"]

# -------- Parameters you’ll most likely tweak ----------
fixed_attack = "PIA"          # e.g. "PIA" or "Poison"
fixed_attack_position = 9    # e.g. 0, 24, 49
d_abbrev = "rqa"
#d_abbrev = "nq"
# -------------------------------------------------------

plots = {
    "results_with_baselines": {
        "datasets": [
            "realtimeqa_sorted",
            "realtimeqa_allrel_perturb",
            #"open_nq_sorted",
            #"open_nq_allrel_perturb"
            #"triviaqa_sorted"
        ],
        "models": ["llama3b", "mistral7b"],
        "defenses": [
            {"name": "none"},
            #{"name": "astuterag"},
            #{"name": "instructrag_icl"},
            {"name": "keyword", "params": {"gamma": [1.0]}},
            {"name": "MIS"},
            #{"name": "sampleMIS", "params": {"gamma": [0.9], "T": [20], "m": [2]}},
        ],
    },
}

# One color per dataset (extend / edit as you like)
dataset_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

for plot_name, cfg in plots.items():
    datasets  = cfg["datasets"]
    models    = cfg["models"]
    defenses  = cfg["defenses"]

    for model in models:
        # Map: defense -> list of accuracies (one per dataset, same order as datasets)
        acc_map = {
            (d["name"] if d["name"] != "none" else "vanilla"): [0.0] * len(datasets)
            for d in defenses
        }

        # ---------- Load results ----------
        for ds_idx, dataset in enumerate(datasets):
            for defense in defenses:
                # Build candidate file paths
                paths = []
                key = defense["name"] if defense["name"] != "none" else "vanilla"

                if defense["name"] == "none":
                    paths.append(
                        f"./output/{dataset}-{model}-none-rep1-top10-attack{fixed_attack}"
                        f"-attackpos{fixed_attack_position}.csv"
                    )

                elif defense["name"] in {"astuterag", "instructrag_icl", "MIS"}:
                    paths.append(
                        f"./output/{dataset}-{model}-{defense['name']}"
                        f"-rep1-top10-attack{fixed_attack}"
                        f"-attackpos{fixed_attack_position}.csv"
                    )

                elif defense["name"] == "keyword":
                    for gamma in defense["params"]["gamma"]:
                        paths.append(
                            f"./output/{dataset}-{model}-keyword-0.3-3.0-gamma{gamma}"
                            f"-rep1-top10-attack{fixed_attack}"
                            f"-attackpos{fixed_attack_position}.csv"
                        )

                elif defense["name"] == "sampleMIS":
                    for gamma in defense["params"]["gamma"]:
                        for T in defense["params"]["T"]:
                            for m in defense["params"]["m"]:
                                paths.append(
                                    f"./output/{dataset}-{model}-"
                                    f"-rep1-top50-attack{fixed_attack}"
                                    f"-attackpos{fixed_attack_position}.csv"
                                )
                else:
                    continue

                # Use the first path that exists
                for p in paths:
                    if os.path.exists(p):
                        df = pd.read_csv(p)
                        if not df.empty:
                            acc_map[key][ds_idx] = get_accuracy(df)
                        break
                    else:
                        print(f"File not found: {p}")

        # ---------- Plot grouped-bar chart ----------
        defense_labels = list(acc_map.keys())
        num_defenses   = len(defense_labels)
        num_datasets   = len(datasets)

        x = np.arange(num_defenses)                         # center of each group
        total_group_width = 0.8
        bar_width = total_group_width / num_datasets        # width of each bar
        offsets = (np.arange(num_datasets) - (num_datasets-1)/2) * bar_width

        fig, ax = plt.subplots(figsize=(6, 5.5))

        for ds_idx, dataset in enumerate(datasets):
            heights = [acc_map[d][ds_idx] for d in defense_labels]
            ax.bar(
                x + offsets[ds_idx],
                heights,
                bar_width * 0.9,          # small gap between bars
                label=dataset_labels.get(dataset),
                color=dataset_colors[ds_idx % len(dataset_colors)],
                edgecolor="black",
                linewidth=0.5,
            )

        # Formatting
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Defense Method")
        ax.set_title(
            f"Model: {model} | Attack: {fixed_attack} @ pos {fixed_attack_position+1}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([map_label.get(d) for d in defense_labels], rotation=15)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title="Datasets")

        # Save
        out_dir = "./figs_accuracy_perdataset/"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{model}_{plot_name}_grouped_accuracy_{d_abbrev}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        print(f"Saved: {os.path.join(out_dir, fname)}")
        plt.close()
