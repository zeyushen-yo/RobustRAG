import os
import re
import logging
from openai import AzureOpenAI

logger = logging.getLogger('RRAG-main')

sandbox_api_key = os.environ['AI_SANDBOX_KEY']
sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_version = "2024-02-01"

judge_prompt = '''Here is the complete answer:

<Start of Answer>
{answer}
<End of Answer>

an LLM provides to the following question:

<Start of Question>
{question}
<End of Question>

Extract the ultimate answer provided by the LLM to the question, without any additional analysis, thinking, internal notes, etc. Provide your answer in the following format:

<Answer>
[Your Answer].
</Answer>

Example: Given the following complete answer:

<Start of Answer>
The first document provides irrelevant information to the question. The second document says Jack wins the prize at 2011 but seems incorrect.
The third document says Jack wins the prize at 2005 and seems more trustworthy. Thus, my answer is 2005.
<End of Answer>

an LLM provides to the following question:

<Start of Question>
When did Jack win the Nobel Prize?
<End of Question>

The answer you should provide is the following:

<Answer>
2005.
</Answer>
'''

def llm_judge(model, question, answer):
    final_prompt = judge_prompt.format(question=question, answer=answer)
    response = get_gpt_output(model, final_prompt)
    final_response = extract_from_text(response, "Answer")
    logger.debug(f"Final response after post-processing by LLM judge: {final_response}")
    return final_response

def get_gpt_output(model, prompt, temperature=0):
    messages = [{"role": "user", "content": prompt}]
    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint=sandbox_endpoint,
        api_version=sandbox_api_version
    )
    response = client.chat.completions.create(
               model=model,
               temperature=temperature, 
               max_tokens=1000, 
               top_p=0.5,
               messages=messages)
    return response.choices[0].message.content

def extract_from_text(text, tag):
    # extract stuffs inside <tag>...</tag>
    try:
        pattern = fr'<{tag}>\s*(.*?)\s*</{tag}>'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""
    except Exception as e:
        print("LLM Judge error extracting from text:", e)
        return ""