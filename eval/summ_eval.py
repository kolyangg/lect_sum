#!/usr/bin/env python
import os
import re
import json
import argparse
import requests
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM

# Model selection
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
# DEFAULT_API_MODEL_NAME = "deepseek/deepseek-r1:free"
DEFAULT_API_MODEL_NAME = "deepseek/deepseek-r1"
MISTRAL_MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Lower temperature for more deterministic outputs
DETERMINISTIC_TEMPERATURE = 0.0

def chat_with_model(user_input, chat_history, system_prompt):
    model = OllamaLLM(model=LOCAL_MODEL_NAME, temperature=DETERMINISTIC_TEMPERATURE)
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    response = model.invoke({"messages": messages})
    return response

def chat_with_api(user_input, chat_history, system_prompt, api_model_name):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    data = {
        "model": api_model_name,
        "messages": messages,
        "temperature": DETERMINISTIC_TEMPERATURE
    }

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

def clean_text(text):
    return re.sub(r"PART_\d+:", "", text).strip()

def remove_latex(text):
    # Remove entire LaTeX equations enclosed in $...$ or $$...$$ (including multiline)
    # This pattern uses DOTALL so that . matches newlines
    pattern = r"\$\$.*?\$\$|\$.*?\$"
    return re.sub(pattern, "", text, flags=re.DOTALL)

def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries with DeepSeek similarity scoring.")
    parser.add_argument("--source_json", required=True, help="Input JSON file containing the source lecture transcriptions")
    parser.add_argument("--summary_json", required=True, help="Input JSON file containing the generated summaries")
    parser.add_argument("--results_json", required=True, help="Output JSON file to store evaluation scores")
    parser.add_argument("--system_prompt", required=True, help="System prompt as string or path to file")
    parser.add_argument("--api", action="store_true", help="Use OpenRouter API model")
    parser.add_argument("--mistral", action="store_true", help="Use Mistral model via OpenRouter")
    parser.add_argument("--ignore_latex", action="store_true", help="Remove LaTeX formulas from summaries before processing")
    parser.add_argument("--retry", type=int, default=3, help="Number of retry attempts for empty responses")
    args = parser.parse_args()

    with open(args.source_json, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    with open(args.summary_json, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    if args.api and not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    if os.path.isfile(args.system_prompt):
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = args.system_prompt

    api_model_name = MISTRAL_MODEL_NAME if args.mistral else DEFAULT_API_MODEL_NAME

    chat_history = []
    results = {}

    for key in tqdm(source_data, desc="Evaluating summaries"):
        source = clean_text(source_data[key])
        if not re.sub(r"[\s\n]+", "", source):
            print(f"Skipping key {key} due to empty or whitespace-only content.")
            continue

        summary = summary_data.get(key, "")
        if args.ignore_latex:
            summary = remove_latex(summary)

        user_input = (
            f"Compare the following source and summary texts, and return a JSON with two keys: \n"
            f"'similarity_1': percentage of source text information present in summary,\n"
            f"'similarity_2': percentage of summary text present in source text.\n"
            f"Both should be between 0 and 1.\n"
            f"Return ONLY a JSON dict like: {{'similarity_1': 0.75, 'similarity_2': 0.82}}.\n\n"
            f"SOURCE:\n{source}\n\nSUMMARY:\n{summary}"
        )

        attempt = 0
        while attempt < args.retry:
            if args.api:
                response = chat_with_api(user_input, chat_history, system_prompt, api_model_name)
            else:
                response = chat_with_model(user_input, chat_history, system_prompt)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            if len(chat_history) > 2:
                chat_history = chat_history[-2:]

            try:
                score = json.loads(response)
                results[key] = score
                print(f"{key}: {score}")
                break
            except Exception as e:
                attempt += 1
                print(f"Failed to parse response for key {key}: {response} | attempt {attempt}")
                if attempt >= args.retry:
                    results[key] = {"similarity_1": None, "similarity_2": None}

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Evaluation complete. Results saved to", args.results_json)

if __name__ == "__main__":
    main()
