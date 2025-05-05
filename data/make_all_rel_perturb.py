import json
import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from openai import OpenAI
from tqdm import tqdm
import time
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def contains_answer(context_item, answers):
    title = context_item.get("title", "").lower()
    text = context_item.get("text", "").lower()
    return any(answer.lower() in title or answer.lower() in text for answer in answers)

async def reword_single(id, context, answer, semaphore):
    prompt = (
        f"Here is a passage that contains the answer '{answer}'\n\n"
        f"Passage: {context}\n\n"
        "Please reword the passage slightly so it still contains keywords in the answer but is phrased differently."
        "Only return the reworded passage without any additional text.\n\n"
        "Reworded passage: "
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.7,
                n=1,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during rewording ({id}): {e}")
            return context  # fallback if error

async def reword_batch(id, contexts, answers):
    semaphore = asyncio.Semaphore(5)  # allow only 5 at a time

    tasks = [
        asyncio.create_task(reword_single(id, context, answer, semaphore))
        for context, answer in zip(contexts, answers)
    ]
    reworded = await asyncio.gather(*tasks)
    return reworded

def make_all_relevant(dataset):
    print(f"Processing dataset: {dataset}")
    json_path = f"./{dataset}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified_data = []
    count = 0
    for item in tqdm(data, desc="Processing items"):
        count += 1
        context_list = item.get("context", [])
        correct_answers = item.get("correct answer", [])

        if not context_list or not correct_answers:
            print("\tSkipping item due to missing context or answers: {count}")
            continue
    
        match_flags = [contains_answer(c, correct_answers) for c in context_list]
        matching_contexts = [context_list[i] for i, is_match in enumerate(match_flags) if is_match]
        num_matches = len(matching_contexts)

        if num_matches == 0:
            print("\tNo matching contexts found, skipping item.")
            modified_data.append(deepcopy(item))
            continue

        reworded_contexts = [None] * len(context_list)
        rr_index = 0
        base_texts = []
        base_contexts = []
        answers_to_use = []
        indices_to_reword = []

        for i, is_match in enumerate(match_flags):
            if is_match:
                context_list[i]["purturbed"]= False
                reworded_contexts[i] = context_list[i]
            else:
                base_context = matching_contexts[rr_index % num_matches]
                base_text = base_context.get("text", "")
                answer_to_use = correct_answers[rr_index % len(correct_answers)]
                base_texts.append(base_text)
                reworded_contexts[i] = deepcopy(base_context) # reworded context will be updated later
                answers_to_use.append(answer_to_use)
                indices_to_reword.append(i)
                rr_index += 1

        # Batch rewording
        if len(indices_to_reword) > 0:
            reworded_texts = asyncio.run(reword_batch(count, base_texts, answers_to_use))        
            for i, new_text in zip(indices_to_reword, reworded_texts):
                new_context = reworded_contexts[i]
                new_context["text"] = new_text
                new_context["purturbed"] = True

        modified_item = deepcopy(item)
        modified_item["context"] = reworded_contexts
        modified_data.append(modified_item)

    with open(f"./{dataset}_allrel_perturb.json", "w", encoding="utf-8") as out_file:
        json.dump(modified_data, out_file, indent=2, ensure_ascii=False)

make_all_relevant("realtimeqa")
make_all_relevant("open_nq")
make_all_relevant("simpleqa")
make_all_relevant("triviaqa")