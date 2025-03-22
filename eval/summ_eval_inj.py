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
DEFAULT_API_MODEL_NAME = "deepseek/deepseek-r1:free"
MISTRAL_MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Lower temperature for more deterministic outputs
DETERMINISTIC_TEMPERATURE = 0.0

"""
This script now accepts --question which can be a path to a txt file.
"""

def chat_with_model(user_input, chat_history, system_prompt):
    model = OllamaLLM(model=LOCAL_MODEL_NAME, temperature=DETERMINISTIC_TEMPERATURE)
    messages = [
        {"role": "system", "content": system_prompt}
    ] + chat_history + [
        {"role": "user", "content": user_input}
    ]
    return model.invoke({"messages": messages})


def chat_with_api(user_input, chat_history, system_prompt, api_model_name):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt}
    ] + chat_history + [
        {"role": "user", "content": user_input}
    ]

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


def clean_text(text: str) -> str:
    return re.sub(r"PART_\d+:", "", text).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question (in Russian) for each key in a source JSON file, using local or API-based LLM."
    )
    parser.add_argument("--source_json", required=True,
                        help="Input JSON file containing text for each key.")
    parser.add_argument("--results_json", required=True,
                        help="Output JSON file to store answers.")
    parser.add_argument("--system_prompt", required=True,
                        help="System prompt as string or path to file.")

    # The question can now be a file path or a direct string
    parser.add_argument("--question", required=True,
                        help="Вопрос в виде строки или путь к файлу.")

    parser.add_argument("--api", action="store_true",
                        help="Use OpenRouter API model instead of local Ollama.")
    parser.add_argument("--mistral", action="store_true",
                        help="Use Mistral model instead of default if using API.")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retry attempts if we get an empty or invalid response.")

    args = parser.parse_args()

    # Load the source data.
    with open(args.source_json, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    # Load the system prompt.
    if os.path.isfile(args.system_prompt):
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = args.system_prompt

    # Check if the question is a file.
    if os.path.isfile(args.question):
        with open(args.question, "r", encoding="utf-8") as f:
            question_text = f.read().strip()
    else:
        question_text = args.question

    # Decide which model to use if using the API.
    api_model_name = MISTRAL_MODEL_NAME if args.mistral else DEFAULT_API_MODEL_NAME

    if args.api and not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not found.")

    results = {}
    chat_history = []

    for key in tqdm(source_data, desc="Asking question"):
        text = clean_text(source_data[key])
        if not re.sub(r"[\s\n]+", "", text):
            print(f"Skipping key {key} due to empty or whitespace-only content.")
            continue

        user_input = (
            f"Контекст:\n{text}\n\n"
            f"Вопрос:\n{question_text}\n\n"
            f"Пожалуйста, ответьте одной фразой или абзацем, без использования информации вне этого контекста."
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

            if not response or response.strip().startswith("[Error]"):
                attempt += 1
                print(f"Empty or error response for key {key}, attempt {attempt}: {response}")
                if attempt >= args.retry:
                    results[key] = ""
            else:
                results[key] = response.strip()
                print(f"{key}: {response}")
                break

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Готово. Ответы сохранены в", args.results_json)


if __name__ == "__main__":
    main()
