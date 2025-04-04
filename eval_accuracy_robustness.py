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
]

for exp in exp_list:
    print(f"Processing {exp}")
    total_samples = 100
    rep = 2 if "gpt" in exp else 5
    total_cnt = total_samples * rep

    file_name = f"./log/{exp}.log"

    with open(file_name, 'r') as file:
        lines = file.readlines()

    rows = []
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
                    row = {
                        "gamma": gamma,
                        "rank": rank,
                        "undefended_acc": int(undefended_match.group()) / total_cnt,
                        "defended_acc": int(defended_match.group()) / total_cnt,
                    }

                    # If it's not an "attacknone" experiment, read ASR counts
                    if "attacknone" not in exp and i + 5 < len(lines):
                        undef_asr_match = re.search(r'\d+', lines[i + 4])
                        def_asr_match = re.search(r'\d+', lines[i + 5])
                        if undef_asr_match and def_asr_match:
                            row["undefended_asr"] = int(undef_asr_match.group()) / total_samples
                            row["defended_asr"] = int(def_asr_match.group()) / total_cnt
                        else:
                            row["undefended_asr"] = 0
                            row["defended_asr"] = 0
                    else:
                        row["undefended_asr"] = 0
                        row["defended_asr"] = 0

                    rows.append(row)
            i += 6 if "attacknone" not in exp else 3
        else:
            i += 1

    df = pd.DataFrame(rows)
    df.to_csv(f"./output/{exp}.csv", index=False)