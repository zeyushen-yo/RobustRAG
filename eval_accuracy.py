import pandas as pd
import re

exp_list = [
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attacknone",
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPIA",
    "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attackPoison",

    "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attacknone",
    "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPIA",
    "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPoison",

    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attacknone",
    "realtimeqa-gpt-4o-keyword-0.3-3.0-rep2-top10-attackPIA",
]


for exp in exp_list:
    print(f"Processing {exp}")
    realtimeqa_total_samples = 100
    rep = 2 if "gpt" in exp else 5
    total_cnt = realtimeqa_total_samples * rep

    file_name = f"./log/{exp}.log"

    # Read the file content
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Prepare list for data
    rows = []

    # Parse the log
    i = 0
    while i < len(lines):
        if "Params: Gamma:" in lines[i]:
            gamma_match = re.search(r'Gamma:\s*([0-9.]+)', lines[i])
            rank_match = re.search(r'rank:\s*(\d+)', lines[i])
            if gamma_match and rank_match and i + 2 < len(lines):
                gamma = float(gamma_match.group(1))
                rank = int(rank_match.group(1))
                undefended_match = re.search(r'\d+', lines[i + 1])
                defended_match = re.search(r'\d+', lines[i + 2])
                if undefended_match and defended_match:
                    undefended_cnt = int(undefended_match.group())
                    defended_cnt = int(defended_match.group())
                    rows.append({
                        "gamma": gamma,
                        "rank": rank,
                        "undefended_acc": undefended_cnt/total_cnt,
                        "defended_acc": defended_cnt/total_cnt
                    })
            i += 3
        else:
            i += 1

    # Create DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(f"./output/{exp}_accuracy.csv")

    # Pretty print to command line
    # print(df.to_string(index=False))