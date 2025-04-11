import pandas as pd
import matplotlib.pyplot as plt

# Helper function to extract defended_acc and defended_asr
def get_defended_metrics(df, gamma, rank):
    row = df[(df["gamma"] == gamma) & (df["rank"] == rank)].iloc[0]
    return row["defended_acc"], row["defended_asr"]

def get_undefended_metrics(df, gamma, rank):
    row = df[(df["gamma"] == gamma) & (df["rank"] == rank)].iloc[0]
    return row["undefended_acc"], row["undefended_asr"]

model = "llama3b"
dataset = "realtimeqa"

for attack in ["PIA", "Poison", "none"]:
    for attack_position in [0,4,9]:
        name = f"{model}-{dataset}-{attack}-attackpos{attack_position}"
        print(f"plotting: {name}")
        keyword_df = pd.read_csv(f"./output/{dataset}-{model}-keyword-0.3-3.0-rep5-top10-attack{attack}.csv")
        sampling_df33 = pd.read_csv(f"./output/{dataset}-{model}-sampling-3-3-emb-rep5-top10-attack{attack}.csv")
        sampling_df35 = pd.read_csv(f"./output/{dataset}-{model}-sampling-3-5-emb-rep5-top10-attack{attack}.csv")
        sampling_df39 = pd.read_csv(f"./output/{dataset}-{model}-sampling-3-9-emb-rep5-top10-attack{attack}.csv")

        astuterag_df = pd.read_csv(f"./output/{dataset}-{model}-astuterag-rep5-top10-attack{attack}.csv")
        instructrag_icl_df = pd.read_csv(f"./output/{dataset}-{model}-instructrag_icl-rep5-top10-attack{attack}.csv")
        
        data = {
            "keyword weighted ($\gamma=0.8$)": get_defended_metrics(keyword_df, gamma=0.8, rank=attack_position),
            "keyword unweighted ($\gamma=1.0$)": get_defended_metrics(keyword_df, gamma=1.0, rank=attack_position),

            "sampling weighted ($\gamma=0.8$, T=3, M=5)": get_defended_metrics(sampling_df35, gamma=0.8, rank=attack_position),
            "sampling unweighted ($\gamma=1.0$, T=3, M=5)": get_defended_metrics(sampling_df35, gamma=1.0, rank=attack_position),

            "astuterag": get_defended_metrics(astuterag_df, gamma=1.0, rank=attack_position),
            "instructrag_icl": get_defended_metrics(instructrag_icl_df, gamma=1.0, rank=attack_position),

            "vanilla": get_undefended_metrics(keyword_df, gamma=1.0, rank=attack_position),
        }

        # Plotting
        plt.figure(figsize=(8, 6))
        for label, (acc, asr) in data.items():
            robustness = 1 - asr
            plt.scatter(acc, robustness, label=label, s=100)
            #if attack != "none":
            #    plt.text(acc + 0.001, robustness - 0.001, label, fontsize=9, rotation=0)

        plt.title(name)
        plt.xlabel("Accuracy")
        plt.ylabel("Robustness (1 - ASR)")
        plt.grid(True)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
        plt.subplots_adjust(top=0.9)  # adjust top margin (0.9 means 90% of the figure height)

        plt.savefig(f"./figs_pareto/{name}.png")
