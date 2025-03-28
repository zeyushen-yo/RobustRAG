import pandas as pd
import re

# Define the file name
#exp = "realtimeqa-llama3b-keyword-0.3-3.0-rep5-top10-attacknone"
#exp = "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attacknone"
exp = "realtimeqa-mistral7b-keyword-0.3-3.0-rep5-top10-attackPIA"
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
                    "undefended_acc": undefended_cnt/500,
                    "defended_acc": defended_cnt/500
                })
        i += 3
    else:
        i += 1

# Create DataFrame
df = pd.DataFrame(rows)
df.to_csv(f"./output/{exp}_accuracy.csv")

# Pretty print to command line
print(df.to_string(index=False))