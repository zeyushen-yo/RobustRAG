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
        #"datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
        "datasets": ["realtimeqa", "realtimeqa_allrel_perturb"],
        "models": ["llama3b", "mistral7b"],
        "attack_positions": [0, 4, 9],
        "attacks": ["PIA", "Poison"],
        "defenses": [
            {"name": "none"},
            {"name": "astuterag"},
            {"name": "MIS"},
            {"name": "instructrag_icl"},
            {"name": "keyword", "params": { "gamma": [1.0] }},
            #{"name": "sampling_keyword", "params": {"gamma": [0.8, 1.0], "T": [10], "m": [1]}},
        ],
    },
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

for plot_name in plots:
    datasets = plots[plot_name]["datasets"]
    models = plots[plot_name]["models"]
    attack_positions = plots[plot_name]["attack_positions"]
    attacks = plots[plot_name]["attacks"]
    defenses = plots[plot_name]["defenses"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:
            fig, axes = plt.subplots(
                nrows=len(attacks), ncols=2, 
                figsize=(12, 5 * len(attacks)),
                constrained_layout=True
            )
            fig.suptitle(f"{model}-{dataset}", fontsize=20)

            if len(attacks) == 1:
                axes = [axes]  # ensure axes is always a list of pairs

            for i, attack in enumerate(attacks):
                ax_robust = axes[i][0]
                ax_acc = axes[i][1]

                all_data = {}

                for attack_position in attack_positions:
                    for defense in defenses:
                        if defense["name"] == "none":
                            key = "vanilla"
                            file_path = f"./output/{dataset}-{model}-none-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                        elif defense["name"] in ["astuterag", "instructrag_icl", "MIS"]:
                            key = defense["name"]
                            file_path = f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                        elif defense["name"] == "keyword":
                            for gamma in defense["params"]["gamma"]:
                                key = f"keyword ($\\gamma$={gamma})"
                                file_path = f"./output/{dataset}-{model}-keyword-0.3-3.0-gamma{gamma}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                        elif defense["name"] == "sampling_keyword":
                            for gamma in defense["params"]["gamma"]:
                                for T in defense["params"]["T"]:
                                    for m in defense["params"]["m"]:
                                        key = f"sampling_keyword (T={T},m={m}) ($\\gamma$={gamma})"
                                        file_path = f"./output/{dataset}-{model}-sampling_keyword-T{T}-m{m}-gamma{gamma}-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                        else:
                            continue  # skip unknown defenses

                        # Read the CSV and collect the results
                        if os.path.exists(file_path):
                            df = pd.read_csv(file_path)
                            if len(df) > 0:
                                all_data.setdefault(key, []).append((attack_position, *get_metrics(df)))
                        else:
                            print(f"File not found: {file_path}")

                for label, values in all_data.items():
                    if label not in key_style_map:
                        color, marker = next(style_combinations)
                        key_style_map[label] = (color, marker)
                    else:
                        color, marker = key_style_map[label]

                    x = [pos for pos, acc, asr in values]
                    y_acc = [acc for pos, acc, asr in values]
                    y_robust = [1 - asr for pos, acc, asr in values]

                    ax_acc.plot(
                        x, y_acc, label=label, marker=marker, linestyle='--',
                        color=color, markersize=8
                    )
                    ax_robust.plot(
                        x, y_robust, label=label, marker=marker, linestyle='-',
                        color=color, markersize=8
                    )

                # Format robustness plot
                ax_robust.set_title(f"Robustness - Attack: {attack}")
                ax_robust.set_xlabel("Attack Position")
                ax_robust.set_ylabel("Robustness (1 - ASR)")
                ax_robust.set_ylim(0, 1)
                ax_robust.grid(True)
                ax_robust.set_xticks(attack_positions)

                # Format accuracy plot
                ax_acc.set_title(f"Accuracy - Attack: {attack}")
                ax_acc.set_xlabel("Attack Position")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_ylim(0, 1)
                ax_acc.grid(True)
                ax_acc.set_xticks(attack_positions)

                if i == 0:
                    ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            os.makedirs("./figs_accuracy_robustness/", exist_ok=True)
            plt.savefig(f"./figs_accuracy_robustness/{model}_{dataset}_{plot_name}_splitaccrobust.png", bbox_inches='tight')
            print(f"Saved figure: ./figs_accuracy_robustness/{model}_{dataset}_{plot_name}_splitaccrobust.png")
            plt.close()
