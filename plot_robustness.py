import pandas as pd
import matplotlib.pyplot as plt

exp_list = [
    "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attacknone",
    "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPIA",
    "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPoison",

    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attacknone",
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPIA",
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPoison",

    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attacknone",
    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attackPIA",
]

for exp in exp_list:
    if "attacknone" in exp:
        continue
    print(f"Plotting {exp}")
    
    in_file = f"./output/{exp}.csv"

    df = pd.read_csv(in_file)

    # Plotting
    plt.figure(figsize=(10, 6))

    for gamma in df["gamma"].unique():
        subset = df[df["gamma"] == gamma]
        
        label_defended = f"Gamma {gamma} (new)" if gamma != 1 else f"Gamma {gamma} (RobustRAG)"
        plt.plot(subset["rank"], [1 - asr for asr in subset["defended_asr"]], marker='x', label=label_defended)
        
        if gamma == 1:
            label_undefended = f"Vanilla"
            plt.plot(subset["rank"], [1 - asr for asr in subset["undefended_asr"]], marker='o', label=label_undefended)

    plt.xlabel("Rank")
    plt.ylabel("Robustness (1 - ASR)")
    plt.ylim([0.5,1])
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.title(exp)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./figs_robustness/{exp}.png")