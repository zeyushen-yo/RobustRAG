import pandas as pd
import matplotlib.pyplot as plt

# Helper function to extract defended_acc and defended_asr
def get_metrics(df, gamma, rank):
    row = df[(df["gamma"] == gamma) & (df["rank"] == rank)].iloc[0]
    return row["defended_acc"], row["defended_asr"]

model = "llama3b"
dataset = "realtimeqa"

for attack in ["none", "PIA", "Poison"]:
    for attack_position in [0,4,9]:
        name = f"{model}-{dataset}-{attack}-attackpos{attack_position}"
        print(f"plotting: {name}")
        keyword_df = pd.read_csv(f"./output/{dataset}-{model}-keyword-0.3-3.0-rep5-top10-attack{attack}.csv")
        sampling_df = pd.read_csv(f"./output/{dataset}-{model}-sampling-rep5-top10-attack{attack}.csv")

        # Define data points
        data = {
            "keyword weighted ($\gamma=0.8$)": get_metrics(keyword_df, gamma=0.8, rank=attack_position),
            "keyword unweighted ($\gamma=1.0$)": get_metrics(keyword_df, gamma=1.0, rank=attack_position),
            "sampling weighted ($\gamma=0.8$)": get_metrics(sampling_df, gamma=0.8, rank=attack_position),
            "sampling unweighted ($\gamma=1.0$)": get_metrics(sampling_df, gamma=1.0, rank=attack_position),
        }

        # Plotting
        plt.figure(figsize=(8, 6))
        for label, (acc, asr) in data.items():
            robustness = 1 - asr
            plt.scatter(acc, robustness, label=label, s=100)
            plt.text(acc + 0.001, robustness + 0.001, label, fontsize=9)

        plt.title(name)
        plt.xlabel("Defended Accuracy")
        plt.ylabel("Robustness (1 - ASR)")
        plt.grid(True)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./figs_pareto/{name}.png")