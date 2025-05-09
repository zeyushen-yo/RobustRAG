#!/usr/bin/env python3
"""
Plot accuracy curves (mean ± 95 % CI) for Robust RAG experiments.

Changes relative to the original script
---------------------------------------
1. `get_metrics` now looks at the **first ≤ 5 rows** of every CSV file
   and returns `(mean_acc, ci_acc, mean_asr, ci_asr)`, where the CI is
   the 95 % confidence-interval half-width (±).
2. Each accuracy curve is accompanied by a **shaded CI ribbon** that uses
   the same colour with `alpha=0.2`.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Metrics helpers
# ----------------------------------------------------------------------
Z = 1.96  # z-score for 95 % confidence

def _mean_and_ci(series: pd.Series) -> tuple[float, float]:
    """Return (mean, 95 % CI half-width) for a Series with ≤ 5 elements."""
    n    = len(series)
    mean = series.mean()
    if n < 2:
        return mean, 0.0                     # CI undefined for a single row
    sem  = series.std() / np.sqrt(n)   # standard error of the mean
    return mean, Z * sem

def get_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Compute mean & 95 % CI of ACC and ASR using the first ≤ 5 rows.
    Returns: mean_acc, ci_acc, mean_asr, ci_asr
    """
    subset = df.iloc[:5]                     # first 5 (or fewer) reps
    mean_acc, ci_acc = _mean_and_ci(subset["acc"])
    mean_asr, ci_asr = _mean_and_ci(subset["asr"])
    return mean_acc, ci_acc, mean_asr, ci_asr
# ----------------------------------------------------------------------

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE)
plt.rc('ytick', labelsize=BIGGER_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)

# ----------------------------------------------------------------------
# 2) Configuration
# ----------------------------------------------------------------------
plots = {
    "results_with_baselines": {
        #"datasets": ["realtimeqa_sorted", "open_nq_sorted"],
        "datasets": ["realtimeqa_sorted", "open_nq_sorted", "triviaqa_sorted"],
        "models":   ["mistral7b"],
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
    "none"           : "Vanilla RAG",
    "astuterag"      : "AstuteRAG",
    "instructrag_icl": "InstructRAG (ICL)",
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

# ----------------------------------------------------------------------
# 3) Plotting loop
# ----------------------------------------------------------------------
for plot_name, cfg in plots.items():
    datasets         = cfg["datasets"]
    models           = cfg["models"]
    attack_positions = cfg["attack_positions"]
    attacks          = cfg["attacks"]
    defenses         = cfg["defenses"]

    print("\nPlotting for:", plot_name)
    for dataset in datasets:
        for model in models:

            # ── where to save raw numbers ───────────────────────────────
            values_dir = "./figs_robustness_new_ci/values"
            os.makedirs(values_dir, exist_ok=True)
            txt_path   = os.path.join(values_dir,
                                      f"{model}_{dataset}_accuracy_top50.txt")
            header_written = False

            # ── one subplot per attack type ─────────────────────────────
            fig, axes = plt.subplots(
                nrows=1, ncols=len(attacks),
                figsize=(5 * len(attacks), 5),
                constrained_layout=True,
            )
            if len(attacks) == 1:            # keep iterable
                axes = [axes]

            for i, attack in enumerate(attacks):
                ax_acc   = axes[i]
                all_data = {}                # keyed by defense

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # iterate over attack positions & defense configs
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                for attack_position in attack_positions:
                    for defense in defenses:

                        # ---------- build file path + legend key ----------
                        if defense["name"] == "none":
                            key = "none"
                            file_path = (
                                f"./output/{dataset}-{model}-none-rep5-top50-"
                                f"attack{attack}-attackpos{attack_position}.csv"
                            )

                        elif defense["name"] in ["astuterag", "instructrag_icl"]:
                            key = defense["name"]
                            file_path = (
                                f"./output/{dataset}-{model}-{key}-rep5-top50-"
                                f"attack{attack}-attackpos{attack_position}.csv"
                            )

                        elif defense["name"] == "keyword":
                            for gamma in defense["params"]["gamma"]:
                                key = "keyword"
                                file_path = (
                                    f"./output/{dataset}-{model}-keyword-0.3-3.0-"
                                    f"gamma{gamma}-rep5-top50-attack{attack}-"
                                    f"attackpos{attack_position}.csv"
                                )

                        elif defense["name"] == "sampleMIS":
                            for gamma in defense["params"]["gamma"]:
                                for T in defense["params"]["T"]:
                                    for m in defense["params"]["m"]:
                                        key = "sampleMIS"
                                        file_path = (
                                            f"./output/{dataset}-{model}-"
                                            f"-rep5-top50-"
                                            f"attack{attack}-attackpos{attack_position}.csv"
                                        )

                        else:
                            continue
                        # -------------------------------------------------

                        if os.path.exists(file_path):
                            df = pd.read_csv(file_path)
                            if not df.empty:
                                mean_acc, ci_acc, mean_asr, ci_asr = get_metrics(df)
                                all_data.setdefault(key, []).append(
                                    (attack_position, mean_asr, ci_asr)
                                )

                # ── write numeric values to disk ────────────────────────
                if all_data:
                    mode = "w" if not header_written else "a"
                    with open(txt_path, mode, encoding="utf-8") as f:
                        if not header_written:
                            f.write("defense\tattack\tattack_position\asr\n")
                            header_written = True
                        for dkey, vals in all_data.items():
                            for pos, mean_asr, _ in vals:
                                f.write(f"{map_label[dkey]}\t{attack}\t{pos+1}\t{mean_asr:.4f}\n")

                # ── plotting curves + CIs ───────────────────────────────
                for label_key, values in all_data.items():
                    x_pos   = [pos + 1    for pos, _, _  in values]
                    y_acc   = [m_acc      for _,  m_acc, _ in values]
                    y_ci    = [ci         for _,  _,     ci in values]

                    color, marker = style_map[label_key]

                    # central line
                    ax_acc.plot(x_pos, y_acc,
                                label=map_label[label_key],
                                marker=marker, linestyle="--",
                                color=color, markersize=8)

                    # CI ribbon
                    lower = np.array(y_acc) - np.array(y_ci)
                    upper = np.array(y_acc) + np.array(y_ci)
                    ax_acc.fill_between(x_pos, lower, upper,
                                        color=color, alpha=0.20, linewidth=0)

                # axes cosmetics
                ax_acc.set_title(f"{model}; {attack}; {dataset_labels[dataset]}")
                ax_acc.set_xlabel("Attack Position")
                ax_acc.set_ylabel("ASR (mean ± 95 % CI)")
                ax_acc.set_ylim(0, 1)
                ax_acc.grid(True)
                ax_acc.set_xticks(np.array(attack_positions) + 1)

            # ── consolidate legend & save figure ────────────────────────
            handles, labels = axes[0].get_legend_handles_labels()
            if "Sampling+MIS" in labels:            # push this entry last
                sm_idx = labels.index("Sampling+MIS")
                handles.append(handles.pop(sm_idx))
                labels.append(labels.pop(sm_idx))

            fig.legend(handles, labels,
                       loc="upper center", bbox_to_anchor=(0.5, 1.20),
                       ncol=2, frameon=False)

            out_dir = "./figs_robustness_new_ci/"
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{model}_{dataset}_{plot_name}_robustness_top50.png"
            fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
            print(f"Saved figure: {os.path.join(out_dir, fname)}")
            print(f"   → values written to: {txt_path}")
            plt.close(fig)
