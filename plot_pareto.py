import pandas as pd
import matplotlib.pyplot as plt

# Helper function to extract defended_acc and defended_asr
def get_defended_metrics(df, gamma, rank):
    row = df[(df["gamma"] == gamma) & (df["rank"] == rank)].iloc[0]
    return row["defended_acc"], row["defended_asr"]

def get_metrics(df):
    row = df.iloc[0]
    return row["acc"], row["asr"]

def get_undefended_metrics(df, gamma, rank):
    row = df[(df["gamma"] == gamma) & (df["rank"] == rank)].iloc[0]
    return row["undefended_acc"], row["undefended_asr"]

plt.rcParams.update({'font.size': 14})

plots = [
    "sampling_keyword_m1_vary_t_unweighted",
    "sampling_keyword_t10_vary_m_unweighted",

    "sampling_keyword_m1_vary_t_weighted",
    "sampling_keyword_t10_vary_m_weighted",

    "sampling_with_baselines"
]
for plot_name in plots:
    print("\nPlotting for:", plot_name)
    for dataset in ["realtimeqa", "realtimeqa_allrel"]:
        for model in ["tai_llama8b", "tai_mistral7b"]:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15), constrained_layout=True)
            fig.suptitle(f"Accuracy vs. Robustness for {model}-{dataset}", fontsize=18)

            #for i, attack in enumerate(["PIA", "Poison", "none"]):
            for i, attack in enumerate(["PIA", "Poison", "none"]):
                #for j, attack_position in enumerate([0,4,9]):
                for j, attack_position in enumerate([0,9]):
                    ax = axes[i, j]
                    name = f"{model}-{dataset}-{attack}-attackpos{attack_position}"
                    print(f"\tplotting: {name}")

                    vanilla_df =    pd.read_csv(f"./output/{dataset}-{model}-none-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    keyword0_8_df =    pd.read_csv(f"./output/{dataset}-{model}-keyword-0.3-3.0-gamma0.8-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    keyword1_0_df =    pd.read_csv(f"./output/{dataset}-{model}-keyword-0.3-3.0-gamma1.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    
                    samplingkeyword0_8_df20_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T20-m1-gamma0.8-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    samplingkeyword1_0_df20_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T20-m1-gamma1.0-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    samplingkeyword0_8_df10_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T10-m1-gamma0.8-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    samplingkeyword1_0_df10_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T10-m1-gamma1.0-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    samplingkeyword0_8_df5_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T5-m1-gamma0.8-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    samplingkeyword1_0_df5_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T5-m1-gamma1.0-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    samplingkeyword0_8_df3_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T3-m1-gamma0.8-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    samplingkeyword1_0_df3_1 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T3-m1-gamma1.0-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    samplingkeyword0_8_df10_3 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T10-m3-gamma0.8-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    samplingkeyword1_0_df10_3 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T10-m3-gamma1.0-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    samplingkeyword0_8_df10_5 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T10-m5-gamma0.8-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    samplingkeyword1_0_df10_5 = pd.read_csv(f"./output/{dataset}-{model}-sampling_keyword-T10-m5-gamma1.0-a0.3-b3.0-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    #astuterag_df =  pd.read_csv(f"./output/{dataset}-{model}-astuterag-rep1-top10-attack{attack}-attackpos{attack_position}.csv")
                    #instructrag_icl_df = pd.read_csv(f"./output/{dataset}-{model}-instructrag_icl-rep1-top10-attack{attack}-attackpos{attack_position}.csv")

                    vanilla = {
                        "vanilla": get_metrics(vanilla_df),
                    }
                    baselines = {
                        #"astuterag": get_metrics(astuterag_df),
                        #"instructrag_icl": get_metrics(instructrag_icl_df),
                    }

                    all_data = {
                        "sampling_keyword_m1_vary_t_unweighted": {
                            "sampling+keyword (T=20,m=1) ($\gamma$=1.0)": get_metrics(samplingkeyword1_0_df20_1),
                            "sampling+keyword (T=10,m=1) ($\gamma$=1.0)": get_metrics(samplingkeyword1_0_df10_1),
                            "sampling+keyword (T=5,m=1) ($\gamma$=1.0)":  get_metrics(samplingkeyword1_0_df5_1),
                            "sampling+keyword (T=3,m=1) ($\gamma$=1.0)":  get_metrics(samplingkeyword1_0_df3_1),
                        },
                        "sampling_keyword_t10_vary_m_unweighted": {
                            "sampling+keyword (T=10,m=1) ($\gamma$=1.0)":  get_metrics(samplingkeyword1_0_df10_1),
                            "sampling+keyword (T=10,m=3) ($\gamma$=1.0)":  get_metrics(samplingkeyword1_0_df10_3),
                            "sampling+keyword (T=10,m=5) ($\gamma$=1.0)":  get_metrics(samplingkeyword1_0_df10_5),
                        },

                        "sampling_keyword_m1_vary_t_weighted": {
                            "sampling+keyword (T=20,m=1) ($\gamma$=0.8)": get_metrics(samplingkeyword0_8_df20_1),
                            "sampling+keyword (T=10,m=1) ($\gamma$=0.8)": get_metrics(samplingkeyword0_8_df10_1),
                            "sampling+keyword (T=5,m=1) ($\gamma$=0.8)":  get_metrics(samplingkeyword0_8_df5_1),
                            "sampling+keyword (T=3,m=1) ($\gamma$=0.8)":  get_metrics(samplingkeyword0_8_df3_1),
                        },
                        "sampling_keyword_t10_vary_m_weighted": {
                            "sampling+keyword (T=10,m=1) ($\gamma$=0.8)":  get_metrics(samplingkeyword0_8_df10_1),
                            "sampling+keyword (T=10,m=3) ($\gamma$=0.8)":  get_metrics(samplingkeyword0_8_df10_3),
                            "sampling+keyword (T=10,m=5) ($\gamma$=0.8)":  get_metrics(samplingkeyword0_8_df10_5),
                        },
                        "sampling_with_baselines": {
                            "sampling+keyword (T=10,m=1) ($\gamma$=1.0)": get_metrics(samplingkeyword1_0_df10_1),
                            "sampling+keyword (T=10,m=1) ($\gamma$=0.8)":  get_metrics(samplingkeyword0_8_df10_1),
                            "keyword ($\gamma$=1.0)": get_metrics(keyword1_0_df),
                            "keyword ($\gamma$=0.8)": get_metrics(keyword0_8_df),
                        },
                    }

                    data = all_data[plot_name]
                    data.update(vanilla)

                    # Plotting
                    #plt.figure(figsize=(8, 6))
                    for label, (acc, asr) in data.items():
                        robustness = 1 - asr
                        ax.scatter(acc, robustness, label=label, s=100)
                        #if attack != "none":
                        #    plt.text(acc + 0.001, robustness - 0.001, label, fontsize=9, rotation=0)

                    ax.set_title(f"Attack: {attack}, Pos: {attack_position}")
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.set_xlabel("Accuracy")
                    ax.set_ylabel("Robustness (1 - ASR)")
                    ax.grid(True)
                    if i == 0 and j == 1:  # place legend once
                        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

            plt.savefig(f"./figs_pareto/{model}_{dataset}_{plot_name}.png")
            plt.close()
