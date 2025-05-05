import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools

def get_metrics(df):
    row = df.iloc[0]
    return row["acc"], row["asr"]

plt.rcParams.update({'font.size': 14})

plots = {
    "results_with_baselines": {
        "datasets": ["realtimeqa_sorted", "open_nq_sorted", "triviaqa_sorted"],
        "models": ["llama3b", "mistral7b"],
        "attack_positions": [0, 4, 9],
        "attacks": ["PIA"],
        "defenses": [
            {"name": "none"},
            {"name": "astuterag"},
            {"name": "MIS"},
            {"name": "instructrag_icl"},
            {"name": "keyword", "params": {"gamma": [1.0]}},
        ],
    },
}

dataset_labels = {
    "realtimeqa_sorted": "RQA",
    "open_nq_sorted": "NQ",
    "triviaqa_sorted": "TQA",
}

map_label = {
    "none": "Vanilla",
    "astuterag": "AstuteRAG",
    "instructrag_icl": "InstructRAG-ICL",
    "sampleMIS": "Sampling+MIS",
    "MIS": "MIS",
    "keyword": "RobustRAG (Keyword)",
}

style_map = {
    "none"          : ("tab:blue"  , "o"),   # Vanilla
    "astuterag"     : ("tab:orange", "v"),   # AstuteRAG
    "keyword"       : ("tab:green" , "^"),   # RobustRAG (Keyword)
    "instructrag_icl":("tab:red"   , "s"),   # InstructRAG-ICL
    "sampleMIS"     : ("tab:purple", "D"),   # Sampling+MIS
    "MIS"           : ("tab:brown" , "p"),   # MIS
}

# Define colors and markers
colors = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
]
markers = [
    "o", "v", "^", "<", ">", "s", "p", "*", "h", "D"
]

# Create (color, marker) combinations
style_combinations = itertools.cycle(
    [(c, m) for m in markers for c in colors]
)
key_style_map = {}

for plot_name, cfg in plots.items():
    datasets         = cfg["datasets"]
    models           = cfg["models"]
    attack_positions = cfg["attack_positions"]
    attacks          = cfg["attacks"]
    defenses         = cfg["defenses"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:
            fig, axes = plt.subplots(
                nrows=1, ncols=len(attacks),
                figsize=(5*len(attacks), 5),
                constrained_layout=True
            )
            fig.subplots_adjust(top=0.82)
            #fig.suptitle(f"Model: {model}, Dataset: {dataset}", y=1.1, fontsize=20)

            # Ensure axes is always iterable
            if len(attacks) == 1:
                axes = [axes]

            for i, attack in enumerate(attacks):
                ax_acc = axes[i]

                all_data = {}

                for attack_position in attack_positions:
                    for defense in defenses:
                        # --- Compose file path & key -------------------------------------------------
                        if defense["name"] == "none":
                            key = "none"
                            file_path = (
                                f"./output/{dataset}-{model}-none-rep1-top10-attack{attack}"
                                f"-attackpos{attack_position}.csv"
                            )
                        elif defense["name"] in ["astuterag", "instructrag_icl", "MIS"]:
                            key = defense["name"]
                            file_path = (
                                f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}"
                                f"-attackpos{attack_position}.csv"
                            )
                        elif defense["name"] == "keyword":
                            for gamma in defense["params"]["gamma"]:
                                key = "keyword"
                                file_path = (
                                    f"./output/{dataset}-{model}-keyword-0.3-3.0-gamma{gamma}"
                                    f"-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                                )
                        elif defense["name"] == "sampleMIS":
                            for gamma in defense["params"]["gamma"]:
                                for T in defense["params"]["T"]:
                                    for m in defense["params"]["m"]:
                                        key = "sampleMIS"
                                        file_path = (
                                            f"./output/{dataset}-{model}-"
                                            f"-rep1-top10-attack{attack}"
                                            f"-attackpos{attack_position}.csv"
                                        )
                        else:
                            continue  # skip unknown defenses

                        # ---------------------------------------------------------------------------
                        if os.path.exists(file_path):
                            df = pd.read_csv(file_path)
                            if not df.empty:
                                acc, asr = get_metrics(df)
                                all_data.setdefault(key, []).append((attack_position, acc))

                # ---- Plot -----------------------------------------------------------------------
                for label_key, values in all_data.items():
                    #if label_key not in key_style_map:
                    #    color, marker = next(style_combinations)
                    #    key_style_map[label_key] = (color, marker)
                    #else:
                    #    color, marker = key_style_map[label_key]

                    x = [pos for pos, _ in values]
                    y_acc = [acc for _, acc in values]

                    color, marker = style_map[label_key]

                    ax_acc.plot(
                        x, y_acc,
                        label=map_label[label_key],
                        marker=marker, linestyle='--',
                        color=color, markersize=8
                    )

                # ---- Format accuracy subplot ----------------------------------------------------
                ax_acc.set_title(f"{model}; {attack}; {dataset_labels[dataset]}")
                ax_acc.set_xlabel("Attack Position")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_ylim(0, 1)
                ax_acc.grid(True)
                ax_acc.set_xticks(attack_positions)

            # ============ SINGLE, HORIZONTAL LEGEND (top, across figure) =========================
            # Grab handles/labels from the first axis
            handles, labels = axes[0].get_legend_handles_labels()

            # Re-order so that “Sampling+MIS” is last
            if "MIS" in labels:
                sm_idx = labels.index("MIS")
                # Pop the Sampling+MIS entry and append it
                sm_handle = handles.pop(sm_idx)
                sm_label  = labels.pop(sm_idx)
                handles.append(sm_handle)
                labels.append(sm_label)

            # Create a horizontal legend spanning the figure width
            fig.legend(handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 1.14),
                ncol=3, frameon=False, fontsize=12)

            # ---- Save figure -------------------------------------------------------------------
            out_dir = "./figs_accuracy_new/"
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{model}_{dataset}_{plot_name}_accuracy_top10.png"
            plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
            print(f"Saved figure: {os.path.join(out_dir, fname)}")
            plt.close()
