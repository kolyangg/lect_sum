#!/usr/bin/env python
import os
import re
import json
import argparse
import requests
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM

# Model selection constants
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
# DEFAULT_API_MODEL_NAME = "deepseek/deepseek-r1:free"
DEFAULT_API_MODEL_NAME = "deepseek/deepseek-r1"
MISTRAL_MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# Lower temperature for more deterministic outputs
DETERMINISTIC_TEMPERATURE = 0.0

def chat_with_model(user_input, chat_history, system_prompt):
    model = OllamaLLM(model=LOCAL_MODEL_NAME, temperature=DETERMINISTIC_TEMPERATURE)
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    return model.invoke({"messages": messages})

# def chat_with_api(user_input, chat_history, system_prompt, api_model_name):
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
#     data = {
#         "model": api_model_name,
#         "messages": messages,
#         "temperature": DETERMINISTIC_TEMPERATURE
#     }
#     response = requests.post(API_URL, headers=headers, json=data)
#     try:
#         response_json = response.json()
#         if response.status_code != 200:
#             return f"[Error] API request failed: {response_json.get('error', 'Unknown error')}"
#         if "choices" not in response_json or not response_json["choices"]:
#             return "[Error] No response choices returned from API."
#         return response_json["choices"][0]["message"]["content"]
#     except json.JSONDecodeError:
#         return "[Error] Failed to parse API response as JSON."


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
    
    try:
        # Add a timeout of 90 seconds
        response = requests.post(API_URL, headers=headers, json=data, timeout=90)
    except requests.exceptions.Timeout:
        print("Request timed out after 90 seconds.")
        return ""  # Return an empty string so that the retry logic will trigger
    
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
        description="For each key in a source JSON file, ask all three questions (provided in a quest JSON file) using its context. "
                    "Print the model's answers vs. correct answers along with the count of correct answers, and save results to a JSON file."
    )
    parser.add_argument("--source_json", required=True,
                        help="Input JSON file containing text for each key (context).")
    parser.add_argument("--quest_json", required=True,
                        help="JSON file with questions and correct answers. Each key should match a key in the source JSON and contain subkeys '1', '2', '3' with a 'question' and 'answer'.")
    parser.add_argument("--results_json", required=True,
                        help="Output JSON file to store the grading results.")
    parser.add_argument("--system_prompt", default="Пожалуйста, ответьте тремя буквами (а, б, в, г) через запятую без дополнительных комментариев.",
                        help="System prompt as string or path to file.")
    parser.add_argument("--api", action="store_true",
                        help="Use OpenRouter API model instead of local Ollama.")
    parser.add_argument("--mistral", action="store_true",
                        help="Use Mistral model instead of default if using API.")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retry attempts if we get an empty or invalid response.")
    args = parser.parse_args()

    # Load the source context data.
    with open(args.source_json, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    # Load the questions and correct answers.
    with open(args.quest_json, "r", encoding="utf-8") as f:
        quest_data = json.load(f)

    # Load system prompt from file if applicable.
    if os.path.isfile(args.system_prompt):
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = args.system_prompt

    # Decide which API model to use, if needed.
    api_model_name = MISTRAL_MODEL_NAME if args.mistral else DEFAULT_API_MODEL_NAME
    if args.api and not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not found.")

    results = {}
    total_correct = 0
    total_questions = 0

    # For each key in the source data, ask all three questions in one prompt.
    for key in tqdm(source_data, desc="Processing keys"):
        context_text = clean_text(source_data[key])
        if not re.sub(r"[\s\n]+", "", context_text):
            print(f"Skipping key {key} due to empty or whitespace-only content.")
            continue

        if key not in quest_data:
            print(f"Skipping key {key} because no corresponding quest data is available.")
            continue

        # Build combined prompt with all three questions.
        combined_prompt = f"Контекст:\n{context_text}\n\nВопросы:\n"
        # Sort questions by their key ("1", "2", "3") assuming numeric keys.
        for q in sorted(quest_data[key].keys(), key=int):
            question_text = quest_data[key][q]["question"]
            combined_prompt += f"{q}. {question_text}\n"
        # combined_prompt += "\nПожалуйста, ответьте тремя буквами (а, б, в, г) через запятую без дополнительных комментариев."
        combined_prompt += "\nПожалуйста, ответьте тремя буквами (а, б, в, г, д) через запятую без дополнительных комментариев."

        chat_history = []
        
        attempt = 0
        answer_given_list = []
        response = ""
        while attempt < args.retry:
            # Reset chat_history for each attempt if you don't want previous attempts included.
            chat_history = []
            if args.api:
                response = chat_with_api(combined_prompt, chat_history, system_prompt, api_model_name)
            else:
                response = chat_with_model(combined_prompt, chat_history, system_prompt)
            
            # Optionally, if you want to accumulate context across retries (for example, to help the model "remember" previous attempts),
            # you can instead clear the chat_history outside the retry loop or not clear it at all.
            # Here we show it being reset to avoid previous attempts from influencing the new one.
            
            # Process the response (for example, remove any special markers or prefixes)
            response = re.sub(r"^\*\*Ответ:\*\*\s*", "", response)
            
            # Optionally, add the current attempt to chat_history if you want it to be part of the context
            # chat_history.append({"role": "user", "content": combined_prompt})
            # chat_history.append({"role": "assistant", "content": response})
            
            # Extract answer letters
            # answer_given_list = re.findall(r"[абвг]", response)
            answer_given_list = re.findall(r"[абвгд]", response)
            if len(answer_given_list) == 3:
                break
            else:
                attempt += 1
                print(f"Key {key} attempt {attempt}: Incomplete answer received: {response.strip()}")
                if attempt >= args.retry:
                    answer_given_list = ["", "", ""]



        # Compare answers with correct answers.
        key_results = {}
        correct_count = 0
        for idx, q in enumerate(sorted(quest_data[key].keys(), key=int), start=0):
            correct_ans = quest_data[key][q]["answer"]
            given_ans = answer_given_list[idx] if idx < len(answer_given_list) else ""
            if given_ans == correct_ans:
                correct_count += 1
            key_results[q] = {
                "answer_given": given_ans,
                "correct_answer": correct_ans
            }
        results[key] = key_results

        total_correct += correct_count
        total_questions += 3

        # Print result for this key.
        print(f"\nKey: {key}")
        for q in sorted(quest_data[key].keys(), key=int):
            print(f"Вопрос {q}: Дано: {results[key][q]['answer_given']}, Правильно: {results[key][q]['correct_answer']}")
        print(f"Итого правильно для {key}: {correct_count}/3")

    print(f"\nОбщее количество правильных ответов: {total_correct}/{total_questions}")

    # Save the grading results.
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Готово. Результаты сохранены в", args.results_json)

if __name__ == "__main__":
    main()
