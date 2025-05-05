import pandas as pd
import os
import matplotlib.pyplot as plt


def get_metrics(df):
    #if len(df) > 1:
        #print("Warning: More than one row in DataFrame, using the first row for metrics.")
    row = df.iloc[0]
    return row["acc"], row["asr"]

plt.rcParams.update({'font.size': 14})

plots = {
    "sampling_with_baselines": {
        "datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
        "models": ["llama3b", "mistral7b"],
        "attack_positions": [0, 4, 9],
        "attacks": ["PIA", "Poison"],
        "defenses": [
            {"name": "none"},
            {"name": "astuterag"},
            {"name": "instructrag_icl"},
            {"name": "MIS"},
            {"name": "keyword", "params": { "gamma": [0.8, 1.0],}},
            {"name": "sampling_keyword", "params": {"gamma": [0.8, 1.0],"T": [10],"m": [1]}},
            {"name": "sampling", "params": {"gamma": [0.8, 1.0],"T": [10],"m": [1]}},
        ],
    },
    #"sampling_keyword_m1_vary_t": {
    #    "datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
    #    "models": ["llama3b", "mistral7b"],
    #    "attack_positions": [0, 4, 9],
    #    "attacks": ["PIA", "Poison"],
    #    "defenses": [
    #        {"name": "none"},
    #        {"name": "sampling_keyword", "params": {"gamma": [1.0],"T": [1, 3, 5, 10],"m": [1]}},
    #    ],
    #},
    #"sampling_m1_vary_t": {
    #    "datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
    #    "models": ["llama3b", "mistral7b"],
    #    "attack_positions": [0, 4, 9],
    #    "attacks": ["PIA", "Poison"],
    #    "defenses": [
    #        {"name": "none"},
    #        {"name": "sampling", "params": {"gamma": [1.0],"T": [1, 3, 5, 10],"m": [1]}},
    #    ],
    #},
    #"sampling_keyword_t10_vary_m": {
    #    "datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
    #    "models": ["llama3b", "mistral7b"],
    #    "attack_positions": [0, 4, 9],
    #    "attacks": ["PIA", "Poison"],
    #    "defenses": [
    #        {"name": "none"},
    #        {"name": "sampling_keyword", "params": {"gamma": [1.0],"T": [10],"m": [1, 3, 5]}},
    #    ],
    #},
    #"sampling_t10_vary_m": {
    #    "datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
    #    "models": ["llama3b", "mistral7b"],
    #    "attack_positions": [0, 4, 9],
    #    "attacks": ["PIA", "Poison"],
    #    "defenses": [
    #        {"name": "none"},
    #        {"name": "sampling", "params": {"gamma": [1.0],"T": [10],"m": [1, 3, 5]}},
    #    ],
    #},
}

markers = [
    ".",   # point
    ",",   # pixel
    "o",   # circle
    "v",   # triangle_down
    "^",   # triangle_up
    "<",   # triangle_left
    ">",   # triangle_right
    "1",   # tri_down
    "2",   # tri_up
    "3",   # tri_left
    "4",   # tri_right
    "s",   # square
    "p",   # pentagon
    "*",   # star
    "h",   # hexagon1
    "H",   # hexagon2
    "+",   # plus
    "x",   # x
    "D",   # diamond
    "d",   # thin_diamond
    "|",   # vertical line
    "_",   # horizontal line
]

# Assign a unique marker for each key across all plots
key_marker_map = {}

for plot_name in plots:
    datasets = plots[plot_name].get("datasets")
    models = plots[plot_name].get("models")
    attack_positions = plots[plot_name].get("attack_positions")
    attacks = plots[plot_name].get("attacks")
    defenses = plots[plot_name].get("defenses")

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:
            fig, axes = plt.subplots(nrows=len(attacks), ncols=len(attack_positions), figsize=(4*len(attacks), 4*len(attack_positions)), constrained_layout=True)
            fig.suptitle(f"Accuracy vs. Robustness for {model}-{dataset}", fontsize=18)

            for i, attack in enumerate(attacks):
                for j, attack_position in enumerate(attack_positions):
                    if len(attack_positions) == 1:
                        ax = axes[i]
                    elif len(attacks) == 1:
                        ax = axes[j]
                    else:
                        ax = axes[i, j]

                    name = f"{model}-{dataset}-{attack}-attackpos{attack_position}"
                    print(f"\tplotting: {name}")

                    all_dfs = {}
                    for defense in defenses:
                        for param in defense.get("params", [{}]):
                            if defense["name"] == "none":
                                key = f"{defense['name']}"
                                key_readable = "vanilla"
                                file_path = f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                                if os.path.exists(file_path):
                                    all_dfs[key_readable] = pd.read_csv(file_path)
                                else:
                                    print(f"File not found: {file_path}")
                            elif defense["name"] == "keyword":
                                for gamma in defense["params"]["gamma"]:
                                    key = f"{defense['name']}-0.3-3.0-gamma{gamma}"
                                    key_readable = f"keyword ($\\gamma$={gamma})"
                                    file_path = f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                                    if os.path.exists(file_path):
                                        all_dfs[key_readable] = pd.read_csv(file_path)
                                    else:
                                        print(f"File not found: {file_path}")
                            elif defense["name"] == "sampling_keyword":
                                for gamma in defense["params"]["gamma"]:
                                    for T in defense["params"]["T"]:
                                        for m in defense["params"]["m"]:
                                            key = f"{defense['name']}-T{T}-m{m}-gamma{gamma}-a0.3-b3.0"
                                            key_readable = f"sampling_keyword (T={T},m={m}) ($\\gamma$={gamma})"
                                            file_path = f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                                            if os.path.exists(file_path):
                                                all_dfs[key_readable] = pd.read_csv(file_path)
                                            else:
                                                print(f"File not found: {file_path}")
                            elif defense["name"] == "sampling":
                                for gamma in defense["params"]["gamma"]:
                                    for T in defense["params"]["T"]:
                                        for m in defense["params"]["m"]:
                                            key = f"{defense['name']}-{T}-{m}-gamma{gamma}"
                                            key_readable = f"sampling (T={T},m={m}) ($\\gamma$={gamma})"
                                            file_path = f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                                            if os.path.exists(file_path):
                                                all_dfs[key_readable] = pd.read_csv(file_path)
                                            else:
                                                print(f"File not found: {file_path}")
                            elif defense["name"] in ["astuterag", "instructrag_icl", "MIS"]:
                                key = f"{defense['name']}"
                                key_readable = defense['name']
                                file_path = f"./output/{dataset}-{model}-{key}-rep1-top10-attack{attack}-attackpos{attack_position}.csv"
                                if os.path.exists(file_path):
                                    all_dfs[key_readable] = pd.read_csv(file_path)
                                else:
                                    print(f"File not found: {file_path}")


                    all_plot_data = {}
                    for key, df in all_dfs.items():
                        if len(df) > 0:
                            all_plot_data[key] = get_metrics(df)

                    for (label, (acc, asr)) in all_plot_data.items():
                        if label not in key_marker_map:
                            key_marker_map[label] = markers[len(key_marker_map) % len(markers)]
                        marker = key_marker_map[label]
                        robustness = 1 - asr
                        ax.scatter(acc, robustness, label=label, s=100, marker=marker)

                    ax.set_title(f"Attack: {attack}, Pos: {attack_position}")
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.set_xlabel("Accuracy")
                    ax.set_ylabel("Robustness (1 - ASR)")
                    ax.grid(True)
                    if i == 0 and j == len(attack_positions) - 1:
                        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

            plt.savefig(f"./figs_pareto/{model}_{dataset}_{plot_name}.png")
            print(f"\tSaved figure: ./figs_pareto/{model}_{dataset}_{plot_name}.png")
            plt.close()
