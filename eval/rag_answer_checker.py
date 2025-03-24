#!/usr/bin/env python
import os
import json
import argparse
import requests
import re
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM

# Model selection
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
DEFAULT_API_MODEL_NAME = "deepseek/deepseek-r1"
MISTRAL_MODEL_NAME = "mistralai/mistral-small-3.1-24b-instruct"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

def clean_model_response(text):
    # Remove Markdown code fences (```json and ```)
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "")
    return cleaned.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate correctness of generated answers compared to correct answers.")
    parser.add_argument("--qa_json", required=True, help="Input JSON file with generated and correct answers.")
    parser.add_argument("--results_json", required=True, help="Output JSON file to store evaluation results with correctness flag.")
    parser.add_argument("--system_prompt", required=True, help="Path to system prompt template text file or a prompt string.")
    parser.add_argument("--api", action="store_true", help="Use OpenRouter API instead of local model.")
    parser.add_argument("--mistral", action="store_true", help="Use Mistral model via OpenRouter")
    parser.add_argument("--retry", type=int, default=3, help="Number of retry attempts for empty responses")
    args = parser.parse_args()

    if os.path.isfile(args.system_prompt):
        with open(args.system_prompt, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = args.system_prompt

    with open(args.qa_json, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    api_model_name = MISTRAL_MODEL_NAME if args.mistral else DEFAULT_API_MODEL_NAME

    results = {}
    total_correct = 0
    total_questions = 0
    chat_history = []

    prompt_template = (
        "Сравните сгенерированный ответ и правильный ответ. "
        "Если правильный ответ равен 'NA', то сгенерированный ответ должен указывать, что информация отсутствует (например, 'Не найдено'). "
        "Если правильный ответ не равен 'NA', то сгенерированный ответ считается правильным, если он совпадает по смыслу с правильным ответом. "
        "Верните ТОЛЬКО JSON-объект вида: {{'correct': true}} или {{'correct': false}} без дополнительных пояснений.\n\n"
        "Сгенерированный ответ: {generated_answer}\n"
        "Правильный ответ: {correct_answer}"
    )

    for key, qa_obj in tqdm(qa_data.items(), desc="Evaluating QA", total=len(qa_data)):
        generated_answer = qa_obj.get("generated_answer", "")
        correct_answer = qa_obj.get("correct_answer", "")

        user_input = prompt_template.format(generated_answer=generated_answer, correct_answer=correct_answer)

        attempt = 0
        eval_response = ""
        while attempt < args.retry:
            if args.api:
                eval_response = chat_with_api(user_input, chat_history, system_prompt, api_model_name)
            else:
                eval_response = chat_with_model(user_input, chat_history, system_prompt)
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": eval_response})
            if len(chat_history) > 2:
                chat_history = chat_history[-2:]
            # Clean the response from Markdown code fences.
            eval_response_clean = clean_model_response(eval_response)
            try:
                eval_result = json.loads(eval_response_clean)
                if "correct" in eval_result:
                    results[key] = eval_result
                    if eval_result["correct"]:
                        total_correct += 1
                    total_questions += 1
                    print(f"{key}: {eval_result}")
                    break
                else:
                    raise ValueError("Key 'correct' not found in response.")
            except Exception as e:
                attempt += 1
                print(f"Failed to parse response for key {key}: {eval_response} | attempt {attempt}")
                if attempt >= args.retry:
                    results[key] = {"correct": None}
                    total_questions += 1

    summary = {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "accuracy": total_correct / total_questions if total_questions > 0 else 0
    }
    results["_summary"] = summary

    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Evaluation complete. Results saved to", args.results_json)

if __name__ == "__main__":
    main()
