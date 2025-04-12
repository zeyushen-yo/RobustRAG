import pandas as pd
import matplotlib.pyplot as plt

exp_list = [
    # "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attacknone",
    # "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPIA",
    # "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPoison",

    # "simpleqa-gpt-4o-keyword-0.3-3.0-rep5-top10-attacknone",
    # "simpleqa-gpt-4o-keyword-0.3-3.0-rep5-top10-attackPIA",
    # "simpleqa-gpt-4o-keyword-0.3-3.0-rep5-top10-attackPoison",

    # "simpleqa-llama3b-decoding-0.0-rep2-top10-attackPIA",
    # "simpleqa-llama3b-decoding-0.0-rep2-top10-attackPoison",

    # "simpleqa-llama3b-keyword-0.3-3.0-rep5-top10-attacknone",
    # "simpleqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPIA",
    # "simpleqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPoison",

    # "simpleqa-mistral7b-decoding-0.0-rep2-top10-attackPIA",
    # "simpleqa-mistral7b-decoding-0.0-rep2-top10-attackPoison",

    # "simpleqa-mistral7b-decoding-0.0-rep2-top10-attackPIA",
    # "simpleqa-mistral7b-decoding-0.0-rep2-top10-attackPoison",

    # "simpleqa-mistral7b-keyword-0.3-3.0-rep5-top10-attacknone",
    # "simpleqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPIA",
    # "simpleqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPoison",

    # "realtimeqa-llama3b-graph-rep5-top10-attacknone",
    # "realtimeqa-llama3b-graph-rep5-top10-attackPIA",
    # "realtimeqa-llama3b-graph-rep5-top10-attackPoison",

    # "realtimeqa-mistral7b-graph-rep5-top10-attacknone",
    # "realtimeqa-mistral7b-graph-rep5-top10-attackPIA",
    # "realtimeqa-mistral7b-graph-rep5-top10-attackPoison",

    # "simpleqa-llama3b-graph-rep2-top10-attacknone",
    # "simpleqa-llama3b-graph-rep2-top10-attackPIA",
    # "simpleqa-llama3b-graph-rep2-top10-attackPoison",

    "simpleqa-llama3b--rep2-top10-attackPIA",

    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attacknone",
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPIA",
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPoison",

    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attacknone",
    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attackPIA",
    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attackPoison",

    "realtimeqa-llama3b-sampling-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-3-5-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-3-5-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-3-5-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-3-3-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-3-3-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-3-3-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-9-3-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-9-3-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-9-3-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-9-5-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-9-5-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-9-5-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-3-5-emb-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-3-5-emb-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-3-5-emb-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-3-7-emb-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-3-7-emb-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-3-7-emb-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-3-9-emb-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-3-9-emb-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-3-9-emb-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-3-3-emb-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-3-3-emb-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-3-3-emb-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-5-3-emb-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-5-3-emb-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-5-3-emb-rep5-top10-attackPoison",

    "realtimeqa-llama3b-sampling-7-3-emb-rep5-top10-attacknone",
    "realtimeqa-llama3b-sampling-7-3-emb-rep5-top10-attackPIA",
    "realtimeqa-llama3b-sampling-7-3-emb-rep5-top10-attackPoison",

    "realtimeqa-llama3b-astuterag-rep5-top10-attacknone",
    "realtimeqa-llama3b-astuterag-rep5-top10-attackPIA",
    "realtimeqa-llama3b-astuterag-rep5-top10-attackPoison",

    "realtimeqa-llama3b-instructrag_icl-rep5-top10-attacknone",
    "realtimeqa-llama3b-instructrag_icl-rep5-top10-attackPIA",
    "realtimeqa-llama3b-instructrag_icl-rep5-top10-attackPoison",
]

new_exp_list = [
    "open_nq-llama3b-astuterag-rep5-top10-attacknone",
    "open_nq-llama3b-astuterag-rep5-top10-attackPIA",
    "open_nq-llama3b-astuterag-rep5-top10-attackPoison",

    "open_nq-llama3b-instructrag_icl-rep5-top10-attacknone",
    "open_nq-llama3b-instructrag_icl-rep5-top10-attackPIA",
    "open_nq-llama3b-instructrag_icl-rep5-top10-attackPoison",

    "open_nq-mistral7b-astuterag-rep5-top10-attacknone",
    #"open_nq-mistral7b-astuterag-rep5-top10-attackPIA",
    "open_nq-mistral7b-astuterag-rep5-top10-attackPoison",

    "open_nq-mistral7b-instructrag_icl-rep5-top10-attacknone",
    "open_nq-mistral7b-instructrag_icl-rep5-top10-attackPIA",
    "open_nq-mistral7b-instructrag_icl-rep5-top10-attackPoison",
]

plt.rcParams.update({'font.size': 14})

for exp in new_exp_list:
    print(f"Plotting {exp}")
    
    in_file = f"./output/{exp}.csv"

    df = pd.read_csv(in_file)

    # Plotting
    plt.figure(figsize=(10, 5))

    for gamma in df["gamma"].unique():
        subset = df[df["gamma"] == gamma]
        defense_method = exp.split("-")[2]
        label_defended = f"{defense_method}"
        if defense_method in ["sampling", "keyword"]:
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
    plt.close()