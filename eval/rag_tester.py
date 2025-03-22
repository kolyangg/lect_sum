#!/usr/bin/env python
import os
import argparse
import json
import requests
import pickle
import hashlib
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

# Ensure nltk tokenizer is available
nltk.download("punkt")

# Directories for caching FAISS index and temporary files
FAISS_DB_PATH = "_faiss_cache"
os.makedirs(FAISS_DB_PATH, exist_ok=True)
DATA_DIRECTORY = "_data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Model names and API details
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
API_MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def fix_formulas(text):
    """
    Convert LaTeX delimiters so that formulas are rendered correctly.
    """
    return (
        text.replace(r"\[", "$$")
            .replace(r"\]", "$$")
            .replace(r"\(", "$")
            .replace(r"\)", "$")
    )

def extract_final_answer(answer):
    """
    Extract only the final answer portion from the response.
    If the answer contains a marker like "### **Ответ", only the text after that marker is kept.
    """
    marker = "### **Ответ"
    if marker in answer:
        parts = answer.split(marker, 1)
        final_answer = parts[1].strip()
        # Remove a leading colon if present.
        if final_answer.startswith(":"):
            final_answer = final_answer[1:].strip()
        return final_answer
    return answer

def hash_file(file_path):
    """
    Generate a unique MD5 hash for a file based on its contents.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_and_split_markdown_file(file_path):
    """
    Load a Markdown file and split its contents into chunks.
    """
    print("[INFO] Loading Markdown file...")
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_sections = []
    for doc in documents:
        sections = text_splitter.split_documents([doc])
        for section in sections:
            section.metadata["source"] = doc.metadata.get("source", file_path)
        all_sections.extend(sections)
    print(f"[INFO] Loaded and split {len(all_sections)} sections from Markdown file.")
    return all_sections

def load_and_split_pdf(file_path):
    """
    Load a PDF file and split its contents.
    """
    print("[INFO] Loading PDF file...")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

def load_documents(file_path):
    """
    Load documents based on file extension.
    """
    if file_path.endswith(".md"):
        return load_and_split_markdown_file(file_path)
    elif file_path.endswith(".pdf"):
        return load_and_split_pdf(file_path)
    else:
        print("[ERROR] Unsupported file format.")
        return []

def build_cached_faiss_retriever(documents, file_hash, use_cache=True):
    """
    Build (or load cached) FAISS embeddings and return a retriever.
    """
    faiss_index_path = os.path.join(FAISS_DB_PATH, f"{file_hash}.faiss")
    docstore_path = os.path.join(FAISS_DB_PATH, f"{file_hash}_docs.pkl")
    embeddings = OllamaEmbeddings(model=LOCAL_MODEL_NAME)
    if use_cache and os.path.exists(faiss_index_path) and os.path.exists(docstore_path):
        print("[INFO] Loaded cached embeddings from FAISS.")
        vector_store = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        with open(docstore_path, "rb") as f:
            vector_store.docstore = pickle.load(f)
    else:
        print("[INFO] Generating embeddings...")
        vector_store = FAISS.from_documents(documents, embeddings)
        print("[INFO] Saving embeddings for future use.")
        vector_store.save_local(faiss_index_path)
        with open(docstore_path, "wb") as f:
            pickle.dump(vector_store.docstore, f)
    return vector_store.as_retriever()

def build_bm25_retriever(documents):
    """
    Build a BM25 retriever for keyword-based search.
    """
    return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize)

def answer_question_with_ollama(question, retrieved_docs, template):
    """
    Generate an answer using the local Ollama model with the given prompt template.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=LOCAL_MODEL_NAME)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def answer_question_with_api(question, retrieved_docs, template):
    """
    Generate an answer using the OpenRouter API.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": API_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an assistant for question-answering tasks."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}\nAnswer:"}
        ]
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

def main():
    parser = argparse.ArgumentParser(description="Simplified RAG System Script")
    parser.add_argument("--prompt_file", required=True, help="Path to system prompt template text file.")
    parser.add_argument("--questions_file", required=True, help="Path to JSON file with questions (dict of key: question).")
    parser.add_argument("--context_file", required=True, help="Path to context file (Markdown or PDF).")
    parser.add_argument("--output_file", required=True, help="Path to output JSON file with answers.")
    parser.add_argument("--use_api", action="store_true", help="Use OpenRouter API instead of local model.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached embeddings if available.")
    args = parser.parse_args()

    # Load system prompt template from the text file.
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        template = f.read().strip()
    print("[INFO] Loaded system prompt template.")

    # Load questions dictionary from the JSON file.
    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions_dict = json.load(f)
    print(f"[INFO] Loaded {len(questions_dict)} questions.")

    # Load and split the context file.
    documents = load_documents(args.context_file)
    if not documents:
        print("[ERROR] No documents loaded. Exiting.")
        return

    # Build retrievers.
    file_hash = hash_file(args.context_file)
    semantic_retriever = build_cached_faiss_retriever(documents, file_hash, use_cache=args.use_cache)
    bm25_retriever = build_bm25_retriever(documents)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    results = {}
    # Process each question with a progress bar.
    for key, question in tqdm(questions_dict.items(), desc="Processing questions", total=len(questions_dict)):
        retrieved_docs = hybrid_retriever.invoke(question)
        if args.use_api:
            if not OPENROUTER_API_KEY:
                print("[ERROR] OPENROUTER_API_KEY not set. Exiting.")
                return
            answer = answer_question_with_api(question, retrieved_docs, template)
        else:
            answer = answer_question_with_ollama(question, retrieved_docs, template)
        answer = fix_formulas(answer)
        answer = extract_final_answer(answer)
        results[key] = answer

    # Save the answers in a JSON file with the same keys.
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved answers to {args.output_file}")

if __name__ == "__main__":
    main()
