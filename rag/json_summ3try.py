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
API_MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Lower temperature for more deterministic outputs
DETERMINISTIC_TEMPERATURE = 0.0

def chat_with_model(user_input, chat_history, system_prompt):
    """
    Chat with the local DeepSeek model in a continuous session.
    Constructs a messages list with a system prompt, the accumulated chat history,
    then appends the new user input.
    """
    # Initialize the model with a lower temperature.
    model = OllamaLLM(model=LOCAL_MODEL_NAME, temperature=DETERMINISTIC_TEMPERATURE)
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    response = model.invoke({"messages": messages})
    return response

def chat_with_api(user_input, chat_history, system_prompt):
    """
    Chat with the OpenRouter API model.
    Uses the API endpoint, API_MODEL_NAME, and OPENROUTER_API_KEY to send a request.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": user_input}]
    data = {
        "model": API_MODEL_NAME,
        "messages": messages,
        "temperature": DETERMINISTIC_TEMPERATURE  # Use lower temperature here as well
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

def clean_transcript(text):
    """
    Remove the "PART_1:" marker from the text.
    (Other markers such as "PART_2:" remain, acting as natural paragraph separators.)
    """
    return text.replace("PART_1:", "").strip()

def is_empty_transcript(text):
    """
    Determine if the transcript text contains any meaningful content.
    This function removes all markers (e.g. PART_1:, PART_2:) and checks if any non‐whitespace remains.
    """
    cleaned = re.sub(r"PART_\d+:\s*", "", text).strip()
    return cleaned == ""

def load_system_prompt(prompt_arg):
    """
    Load system prompt either from a file (if prompt_arg is a valid file path)
    or use the provided string directly.
    """
    if os.path.isfile(prompt_arg):
        with open(prompt_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    return prompt_arg

def main():
    parser = argparse.ArgumentParser(
        description="Batch process lecture transcriptions with DeepSeek to produce structured summaries."
    )
    parser.add_argument("--final_json", default="RL_json1.json",
                        help="Input JSON file containing transcription chunks (default: RL_json1.json)")
    parser.add_argument("--system_prompt", required=True,
                        help="System prompt as a string or a path to a text file with the prompt")
    parser.add_argument("--json_structured", required=True,
                        help="Output JSON file to store structured summaries")
    parser.add_argument("--api", action="store_true", help="Use OpenRouter API model instead of local Ollama model")
    parser.add_argument("--retry", type=int, default=3, help="Number of retry attempts for empty API responses")
    args = parser.parse_args()

    # Load the input JSON data
    with open(args.final_json, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Load the system prompt (from file or directly)
    system_prompt = load_system_prompt(args.system_prompt)

    # If using the API, verify that the API key is available.
    if args.api and not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not found.")

    # Initialize chat_history to store only the last conversation turn.
    chat_history = []
    output_data = {}  # Dictionary to store LLM responses keyed by the input keys.

    # Process each key in the input JSON using a progress bar.
    for key in tqdm(data, desc="Processing keys"):
        transcript = data[key]
        if is_empty_transcript(transcript):
            print(f"Skipping key {key} due to empty content.")
            continue

        # Clean the transcript by removing the "PART_1:" marker.
        user_input = clean_transcript(transcript)

        # Use API with retry logic or the local model.
        if args.api:
            response = chat_with_api(user_input, chat_history, system_prompt)
            attempt = 0
            while response == "" and attempt < args.retry:
                attempt += 1
                print(f"Retrying for key {key}: attempt {attempt}")
                response = chat_with_api(user_input, chat_history, system_prompt)
        else:
            response = chat_with_model(user_input, chat_history, system_prompt)

        # Update the chat history: keep only the immediate previous conversation turn.
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        if len(chat_history) > 2:
            chat_history = chat_history[-2:]

        # Store the response under the corresponding key.
        output_data[key] = response

    # Write the structured summaries to the output JSON file.
    with open(args.json_structured, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=2)

    print("Processing complete. Structured summaries saved to", args.json_structured)

if __name__ == "__main__":
    main()
