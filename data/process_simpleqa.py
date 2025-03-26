import csv
import json
import os
import random
import re
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def call_gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1024):
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print("GPT call error:", e)
        return ""


def get_formatted_expanded_answer(correct_answer):
    prompt_expanded = (
        f"Given the answer '{correct_answer}', generate exactly two alternative ways of expressing the same answer that are factually correct and convey the same meaning. "
        "Format your answer exactly as follows on separate lines:\n"
        "<expanded_answer_1>Your first alternative answer here</expanded_answer_1>\n"
        "<expanded_answer_2>Your second alternative answer here</expanded_answer_2>\n"
        "Do not include any additional text."
    )
    while True:
        response = call_gpt(prompt_expanded, max_tokens=1024)
        match = re.search(
            r"<expanded_answer_1>(.*?)</expanded_answer_1>.*?<expanded_answer_2>(.*?)</expanded_answer_2>",
            response,
            re.DOTALL
        )
        if match:
            ans1 = match.group(1).strip()
            ans2 = match.group(2).strip()
            if ans1 and ans2:
                return [ans1, ans2]
        print("Formatted expanded answer not received. Reprompting...")


def get_formatted_incorrect_answer(correct_answer):
    prompt_incorrect = (
        f"Generate an incorrect answer that is similar to but different from the correct answer '{correct_answer}' for a multiple choice question. "
        "Format your answer exactly as follows on a single line:\n"
        "<incorrect_answer>Your incorrect answer here</incorrect_answer>\n"
        "Do not include any additional text."
    )
    while True:
        response = call_gpt(prompt_incorrect, max_tokens=1024)
        match = re.search(r"<incorrect_answer>(.*?)</incorrect_answer>", response, re.DOTALL)
        if match:
            ans = match.group(1).strip()
            if ans:
                return ans
        print("Formatted incorrect answer not received. Reprompting...")


def get_formatted_incorrect_context(incorrect_answer):
    prompt_incorrect_context = (
        f"Generate exactly 5 plausible but factually incorrect context passages that support the incorrect answer '{incorrect_answer}'. "
        "Format your answer exactly as follows on separate lines:\n"
        "<incorrect_context_1>Your first context snippet here</incorrect_context_1>\n"
        "<incorrect_context_2>Your second context snippet here</incorrect_context_2>\n"
        "<incorrect_context_3>Your third context snippet here</incorrect_context_3>\n"
        "<incorrect_context_4>Your fourth context snippet here</incorrect_context_4>\n"
        "<incorrect_context_5>Your fifth context snippet here</incorrect_context_5>\n"
        "Do not include any additional text."
    )
    while True:
        response = call_gpt(prompt_incorrect_context, max_tokens=1024)
        pattern = (
            r"<incorrect_context_1>(.*?)</incorrect_context_1>.*?"
            r"<incorrect_context_2>(.*?)</incorrect_context_2>.*?"
            r"<incorrect_context_3>(.*?)</incorrect_context_3>.*?"
            r"<incorrect_context_4>(.*?)</incorrect_context_4>.*?"
            r"<incorrect_context_5>(.*?)</incorrect_context_5>"
        )
        match = re.search(pattern, response, re.DOTALL)
        if match:
            contexts = [match.group(i).strip() for i in range(1, 6)]
            if all(contexts):
                return contexts
        print("Formatted incorrect context not received. Reprompting...")


def process_csv(input_csv):
    rows = []
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    # Randomly select 500 problems if there are more than 500 available
    if len(rows) > 500:
        selected_rows = random.sample(rows, 500)
    else:
        selected_rows = rows
    
    data = []
    for i, row in enumerate(tqdm(selected_rows, desc="Processing problems")):
        question = row.get("problem", "").strip()
        correct_answer = row.get("answer", "").strip()
        
        # Get strictly formatted expanded answers
        expanded_answer = get_formatted_expanded_answer(correct_answer)
        
        # Get strictly formatted incorrect answer
        incorrect_answer = get_formatted_incorrect_answer(correct_answer)
        
        # Get strictly formatted incorrect context passages
        incorrect_context = get_formatted_incorrect_context(incorrect_answer)
        
        item = {
            "id": i,
            "question": question,
            "correct answer": [correct_answer],
            "expanded answer": expanded_answer,
            "incorrect answer": incorrect_answer,
            "incorrect_context": incorrect_context,
            "context": []  # to be filled later using google_search.py
        }
        
        data.append(item)
    return data


def main():
    input_csv = "/scratch/gpfs/zs7353/SimpleQA/simple_qa_test_set.csv"   # adjust file name as needed
    output_json = "data/simple_qa.json"
    
    qa_data = process_csv(input_csv)
    with open(output_json, "w", encoding='utf-8') as out_file:
        json.dump(qa_data, out_file, indent=4)
    
    print(f"Generated {len(qa_data)} QA items and saved to {output_json}")

if __name__ == "__main__":
    main()
