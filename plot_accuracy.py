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
    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attackPoison",

    "realtimeqa-llama3b-sampling-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-rep5-top10-attackPoison",
]

for exp in exp_list:
    print(f"Plotting {exp}")
    
    in_file = f"./output/{exp}.csv"

    df = pd.read_csv(in_file)

    # Plotting
    plt.figure(figsize=(10, 6))

    for gamma in df["gamma"].unique():
        subset = df[df["gamma"] == gamma]
        
        label_defended = ""
        if gamma == 1:
            if "sampling" not in exp:
                label_defended += "keyword unweighted"
            else:
                label_defended += "sampling unweighted"
        elif gamma != 1:
            if "sampling" not in exp:
                label_defended += "keyword weighted"
            else:
                label_defended += "sampling weighted"
        label_defended += f" ($\gamma$={gamma})"

        plt.plot(subset["rank"], subset["defended_acc"], marker='x', label=label_defended)
    
        if gamma == 1:
            label_undefended = f"Vanilla"
            plt.plot(subset["rank"], subset["undefended_acc"], marker='o', label=label_undefended)

    plt.xlabel("Rank")
    plt.ylabel("Accuracy")
    plt.ylim([0,1])
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.title(exp)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./figs_accuracy/{exp}.png")