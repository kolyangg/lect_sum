#!/usr/bin/env python
import os
import argparse
import ast
import json
import requests
from langchain_ollama.llms import OllamaLLM

# Model selection
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
API_MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Fixed system prompt (in Russian)
SYSTEM_PROMPT = (
    "Твоя задача выделить из данного текста термины, связанные с Reinforcement Learning и представить ответ "
    "в формате списка python (на русском языке, где возможно), ничего больше. Отсортируй список по степени важности "
    "для конкретной рассматриваемой на занятии темы."
)

def get_response_local(query, system_prompt):
    """Send query using the local DeepSeek model via OllamaLLM."""
    model = OllamaLLM(model=LOCAL_MODEL_NAME)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    return model.invoke({"messages": messages})

def get_response_api(query, system_prompt):
    """Send query using the OpenRouter API model."""
    if not OPENROUTER_API_KEY:
        return "[Error] OPENROUTER_API_KEY not found in environment variables."
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    data = {"model": API_MODEL_NAME, "messages": messages}
    response = requests.post(API_URL, headers=headers, json=data)
    try:
        response_json = response.json()
        if response.status_code != 200:
            return f"[Error] API request failed: {response_json.get('error', 'Unknown error')}"
        if "choices" not in response_json or not response_json["choices"]:
            return "[Error] No response choices returned from API."
        return response_json["choices"][0]["message"]["content"]
    except json.JSONDecodeError:
        return "[Error] Failed to parse API response as JSON."

def is_valid_list(response_str):
    """Checks whether the response string can be parsed as a Python list."""
    try:
        result = ast.literal_eval(response_str)
        return isinstance(result, list)
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract RL-related terms from notes using the DeepSeek model."
    )
    parser.add_argument("--notes", required=True, help="Path to input text file with notes")
    parser.add_argument("--output", required=True, help="Path to output text file to save the response")
    parser.add_argument("--retry", type=int, default=3, help="Number of retry attempts (default: 3)")
    parser.add_argument("--api", action="store_true", help="Use OpenRouter API model instead of local model")
    args = parser.parse_args()

    # Load query from notes file
    with open(args.notes, "r", encoding="utf-8") as f:
        query_text = f.read().strip()

    if not query_text:
        print("The notes file is empty. Exiting.")
        return

    attempt = 0
    response = ""
    while attempt <= args.retry:
        if attempt > 0:
            print(f"Retry attempt {attempt}...")
        if args.api:
            response = get_response_api(query_text, SYSTEM_PROMPT)
        else:
            response = get_response_local(query_text, SYSTEM_PROMPT)
        response_stripped = response.strip()
        if response_stripped and is_valid_list(response_stripped):
            break
        attempt += 1

    # Save the final response to the output file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(response)
    
    print(f"Model response saved to {args.output}")

if __name__ == "__main__":
    main()
