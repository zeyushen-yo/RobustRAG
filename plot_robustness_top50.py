import pandas as pd
import numpy as np
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
        "attack_positions": [0, 24, 49],
        "attacks": ["PIA"],
        "defenses": [
            {"name": "none"},
            {"name": "astuterag"},
            {"name": "sampleMIS", "params": {"gamma": [0.9], "T": [20], "m": [2]}},
            {"name": "instructrag_icl"},
            {"name": "keyword", "params": {"gamma": [1.0]}},
        ],
    },
}

dataset_labels = {
    "realtimeqa_sorted": "RQA",
    "open_nq_sorted"   : "NQ",
    "triviaqa_sorted"  : "TQA",
}

map_label = {
    "none"           : "Vanilla",
    "astuterag"      : "AstuteRAG",
    "instructrag_icl": "InstructRAG-ICL",
    "sampleMIS"      : "Sampling+MIS",
    "keyword"        : "RobustRAG (Keyword)",
}

style_map = {
    "none"           : ("tab:blue"  , "o"),
    "astuterag"      : ("tab:orange", "v"),
    "keyword"        : ("tab:green" , "^"),
    "instructrag_icl": ("tab:red"   , "s"),
    "sampleMIS"      : ("tab:purple", "D"),
}

# ---------------------------------------------------------------------------
for plot_name, cfg in plots.items():
    datasets         = cfg["datasets"]
    models           = cfg["models"]
    attack_positions = cfg["attack_positions"]
    attacks          = cfg["attacks"]
    defenses         = cfg["defenses"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:
            # one value-file per model • dataset
            txt_dir  = "./figs_robustness_new/values"
            os.makedirs(txt_dir, exist_ok=True)
            txt_path = os.path.join(txt_dir, f"{model}_{dataset}_robustness_top50.txt")
            header_written = False

            fig, axes = plt.subplots(
                nrows=1, ncols=len(attacks),
                figsize=(5*len(attacks), 5),
                constrained_layout=True
            )
            if len(attacks) == 1:
                axes = [axes]

            for i, attack in enumerate(attacks):
                ax_acc   = axes[i]
                all_data = {}

                for attack_position in attack_positions:
                    for defense in defenses:
                        # ------------- build file path/key -----------------
                        if defense["name"] == "none":
                            key = "none"
                            file_path = (
                                f"./output/{dataset}-{model}-none-rep1-top50-attack{attack}"
                                f"-attackpos{attack_position}.csv"
                            )

                        elif defense["name"] in ["astuterag", "instructrag_icl"]:
                            key = defense["name"]
                            file_path = (
                                f"./output/{dataset}-{model}-{key}-rep1-top50-attack{attack}"
                                f"-attackpos{attack_position}.csv"
                            )

                        elif defense["name"] == "keyword":
                            for gamma in defense["params"]["gamma"]:
                                key = "keyword"
                                file_path = (
                                    f"./output/{dataset}-{model}-keyword-0.3-3.0-gamma{gamma}"
                                    f"-rep1-top50-attack{attack}-attackpos{attack_position}.csv"
                                )

                        elif defense["name"] == "sampleMIS":
                            for gamma in defense["params"]["gamma"]:
                                for T in defense["params"]["T"]:
                                    for m in defense["params"]["m"]:
                                        key = "sampleMIS"
                                        file_path = (
                                            f"./output/{dataset}-{model}-"
                                            f"-rep1-top50-attack{attack}"
                                            f"-attackpos{attack_position}.csv"
                                        )
                        else:
                            continue
                        # ----------------------------------------------------

                        if os.path.exists(file_path):
                            df = pd.read_csv(file_path)
                            if not df.empty:
                                acc, asr = get_metrics(df)
                                all_data.setdefault(key, []).append((attack_position, asr))

                # ------------- write / append to value file ---------------
                if all_data:
                    mode = "w" if not header_written else "a"
                    with open(txt_path, mode, encoding="utf-8") as f:
                        if not header_written:
                            f.write("defense\tattack\tattack_position\robustness\n")
                            header_written = True
                        for dkey, vals in all_data.items():
                            for pos, asr in vals:
                                f.write(f"{map_label[dkey]}\t{attack}\t{pos+1}\t{asr:.4f}\n")
                # -----------------------------------------------------------

                # --------------------- plotting -----------------------------
                for label_key, values in all_data.items():
                    x      = [pos+1 for pos, _ in values]
                    y_acc  = [acc for _, acc in values]
                    color, marker = style_map[label_key]

                    ax_acc.plot(
                        x, y_acc,
                        label=map_label[label_key],
                        marker=marker, linestyle='--',
                        color=color, markersize=8
                    )

                ax_acc.set_title(f"{model}; {attack}; {dataset_labels[dataset]}")
                ax_acc.set_xlabel("Attack Position")
                ax_acc.set_ylabel("Attack Success Rate (ASR)")
                ax_acc.set_ylim(0, 1)
                ax_acc.grid(True)
                ax_acc.set_xticks(np.array(attack_positions) + 1)

            # -------- legend & save figure -------------------------------
            handles, labels = axes[0].get_legend_handles_labels()
            if "Sampling+MIS" in labels:
                sm_idx = labels.index("Sampling+MIS")
                handles.append(handles.pop(sm_idx))
                labels.append(labels.pop(sm_idx))

            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 1.14),
                       ncol=3, frameon=False, fontsize=12)

            out_dir = "./figs_robustness_new/"
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{model}_{dataset}_{plot_name}_robustness_top50.png"
            plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
            print(f"Saved figure: {os.path.join(out_dir, fname)}")
            print(f"   → values written to: {txt_path}")
            plt.close()
