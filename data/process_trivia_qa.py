import json
import os
import random
import re
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

def call_gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=1024):
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

def get_formatted_expanded_answer(correct_answer, question):
    prompt_expanded = (
        f"Given the answer '{correct_answer}' to the question '{question}', generate exactly two alternative ways of expressing the same answer that are factually correct and convey the same meaning. "
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

def get_formatted_incorrect_answer(correct_answer, question):
    prompt_incorrect = (
        f"Generate an incorrect answer that answers the question '{question}' and is related to but different from the correct answer '{correct_answer}'."
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

def get_formatted_incorrect_context(incorrect_answer, question):
    prompt_incorrect_context = (
        f"Generate exactly 5 plausible but factually incorrect context passages that support the incorrect answer '{incorrect_answer}' to the question '{question}'. "
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

# --- Processing the original JSON file ---

# Load the original JSON file
with open('/home/zs7353/RobustRAG/data/unfiltered-web-dev.json', 'r') as infile:
    original_data = json.load(infile)

processed_entries = []

for idx, entry in enumerate(tqdm(original_data.get("Data", [])[:500],
         desc="Processing entries",
         unit="entry")):
    question = entry.get("Question", "")
    
    # Use the original answer value as the correct answer. Wrap it in a list.
    correct_ans_value = entry.get("Answer", {}).get("Value", "")
    correct_answer_field = [correct_ans_value]
    
    # Generate expanded answer alternatives.
    # (Alternatively, you might choose to simply use the "NormalizedAliases" field.)
    expanded_answer = get_formatted_expanded_answer(correct_ans_value, question)
    
    # Generate an incorrect answer using the helper function.
    incorrect_answer = get_formatted_incorrect_answer(correct_ans_value, question)
    
    # Generate the incorrect context passages.
    incorrect_context = get_formatted_incorrect_context(incorrect_answer, question)
    
    # Process the search results to fill the context field.
    context = []
    for sr in entry.get("SearchResults", []):
        context_item = {
            "title": sr.get("Title", ""),
            "text": sr.get("Description", ""),
            "link": sr.get("Url", "")
        }
        context.append(context_item)
    
    # Create the new formatted entry
    new_entry = {
        "id": idx,
        "question": question,
        "correct answer": correct_answer_field,
        "expanded answer": expanded_answer,
        "incorrect answer": incorrect_answer,
        "incorrect_context": incorrect_context,
        "context": context
    }
    
    processed_entries.append(new_entry)

# Optionally, save the processed data into a new JSON file
with open('/home/zs7353/RobustRAG/data/processed-data.json', 'w') as outfile:
    json.dump(processed_entries, outfile, indent=4)

print("Processing complete. Processed data saved to '/home/zs7353/RobustRAG/data/triviaqa.json'.")