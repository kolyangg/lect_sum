# interface to chat with DeepSeek model:
import os
import argparse
import json
import requests
import streamlit as st
from langchain_ollama.llms import OllamaLLM

# Model selection
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
API_MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def chat_with_ollama(user_input, chat_history, system_prompt):
    """Chat with local Ollama model."""
    model = OllamaLLM(model=LOCAL_MODEL_NAME)

    # Build chat history
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [
        {"role": "user", "content": user_input}
    ]

    return model.invoke({"messages": messages})


def chat_with_api(user_input, chat_history, system_prompt):
    """Chat with OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": system_prompt}] + chat_history + [
        {"role": "user", "content": user_input}
    ]

    data = {"model": API_MODEL_NAME, "messages": messages}

    response = requests.post(API_URL, headers=headers, json=data)

    try:
        response_json = response.json()

        # Handle API errors
        if response.status_code != 200:
            return f"[Error] API request failed: {response_json.get('error', 'Unknown error')}"

        if "choices" not in response_json or not response_json["choices"]:
            return "[Error] No response choices returned from API."

        return response_json["choices"][0]["message"]["content"]

    except json.JSONDecodeError:
        return "[Error] Failed to parse API response as JSON."


def main():
    parser = argparse.ArgumentParser(description="General Chatbot with Streamlit")
    parser.add_argument("--api", action="store_true", help="Use OpenRouter API instead of local Ollama")
    args = parser.parse_args()

    st.title("🗨️ AI Chatbot with Custom System Prompt")

    # System prompt input field
    system_prompt = st.text_area("Enter System Prompt:", value="You are a helpful AI assistant.")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input field
    user_input = st.text_input("Ask a question:")

    if user_input:
        st.chat_message("user").write(user_input)

        # Choose local or API model
        if args.api:
            if not OPENROUTER_API_KEY:
                st.error("API Key not found! Set OPENROUTER_API_KEY in environment variables.")
                return
            response = chat_with_api(user_input, st.session_state.chat_history, system_prompt)
        else:
            response = chat_with_ollama(user_input, st.session_state.chat_history, system_prompt)

        # Store in chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.chat_message("assistant").write(response)


if __name__ == "__main__":
    main()