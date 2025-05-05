import pandas as pd
import os
import matplotlib.pyplot as plt

def get_metrics(df):
    row = df.iloc[0]
    return row["acc"], row["asr"]

plt.rcParams.update({'font.size': 14})

plots = {
    "sampling_keyword_t10_vary_m": {
        "datasets": ["realtimeqa", "realtimeqa_allrel", "realtimeqa_allrel_perturb"],
        "models": ["llama3b", "mistral7b"],
        "attack_positions": [9],
        "attacks": ["PIA", "Poison"],
        "defenses": [
            {"name": "sampling_keyword", "params": {"gamma": [0.8, 1.0], "T": [10], "m": [1, 3, 5]}},
        ],
    }
}

markers = [
    ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4",
    "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"
]

key_marker_map = {}

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
            fig.suptitle(f"{model}-{dataset} (attackpos: {attack_positions[0]})", fontsize=20)

            if len(attacks) == 1:
                axes = [axes]  # Ensure axes is iterable

            for i, attack in enumerate(attacks):
                ax_robust = axes[i][0]
                ax_acc = axes[i][1]

                all_data = {}

                for defense in defenses:
                    defense_name = defense["name"]
                    params = defense.get("params", {})

                    gammas = params.get("gamma", [1.0])
                    Ts = params.get("T", [10])
                    ms = params.get("m", [])

                    for gamma in gammas:
                        for T in Ts:
                            for m in ms:
                                key = f"{defense_name} (T={T}, $\\gamma$={gamma})"
                                file_path = f"./output/{dataset}-{model}-sampling_keyword-T{T}-m{m}-gamma{gamma}-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_positions[0]}.csv"

                                if os.path.exists(file_path):
                                    df = pd.read_csv(file_path)
                                    all_data.setdefault(key, []).append((m, *get_metrics(df)))
                                else:
                                    print(f"File not found: {file_path}")

                for label, values in all_data.items():
                    if label not in key_marker_map:
                        key_marker_map[label] = markers[len(key_marker_map) % len(markers)]
                    marker = key_marker_map[label]

                    x = [m for m, acc, asr in values]
                    y_acc = [acc for m, acc, asr in values]
                    y_robust = [1 - asr for m, acc, asr in values]

                    ax_acc.plot(x, y_acc, label=label, marker=marker, linestyle='--')
                    ax_robust.plot(x, y_robust, label=label, marker=marker, linestyle='-')

                # Format robustness plot
                ax_robust.set_title(f"Robustness - Attack: {attack}")
                ax_robust.set_xlabel("m")
                ax_robust.set_ylabel("Robustness (1 - ASR)")
                ax_robust.set_ylim(0, 1)
                ax_robust.grid(True)

                # Format accuracy plot
                ax_acc.set_title(f"Accuracy - Attack: {attack}")
                ax_acc.set_xlabel("m")
                ax_acc.set_ylabel("Accuracy")
                ax_acc.set_ylim(0, 1)
                ax_acc.grid(True)

                if i == 0:
                    ax_acc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            os.makedirs("./figs_vary_m/", exist_ok=True)
            plt.savefig(f"./figs_vary_m/{model}_{dataset}_{plot_name}_varym_splitaccrobust.png", bbox_inches='tight')
            print(f"Saved figure: ./figs_vary_m/{model}_{dataset}_{plot_name}_varym_splitaccrobust.png")
            plt.close()
