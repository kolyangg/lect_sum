#!/usr/bin/env python
import os
import argparse
import json
import requests
import pickle
import hashlib
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

nltk.download("punkt")

# Directories for caching FAISS index and temporary files
FAISS_DB_PATH = "_faiss_cache"
os.makedirs(FAISS_DB_PATH, exist_ok=True)
DATA_DIRECTORY = "_data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Model names and API details
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
API_MODEL_NAME = "deepseek/deepseek-r1"
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

def build_cached_faiss_retriever(documents, file_hash, embeddings, use_cache=True):
    """
    Build (or load cached) FAISS embeddings and return a retriever.
    """
    faiss_index_path = os.path.join(FAISS_DB_PATH, f"{file_hash}.faiss")
    docstore_path = os.path.join(FAISS_DB_PATH, f"{file_hash}_docs.pkl")
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

# If --hf_emb is provided, use HuggingFace embedder instead of OllamaEmbeddings.
def get_embeddings(use_hf=False):
    if use_hf:
        from transformers import AutoTokenizer, AutoModel
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class HfEmbeddings:
            def __init__(self, model_name="BAAI/llm-embedder"):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(device)
                self.model.eval()

            def embed_documents(self, documents):
                embeddings = []
                with torch.no_grad():
                    for doc in documents:
                        inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
                        outputs = self.model(**inputs)
                        token_embeddings = outputs.last_hidden_state.squeeze(0)
                        embedding = token_embeddings.mean(dim=0)
                        embeddings.append(embedding.cpu().numpy().tolist())
                return embeddings

            def __call__(self, text):
                # Allow the instance to be called like a function.
                if isinstance(text, list):
                    return self.embed_documents(text)
                else:
                    return self.embed_documents([text])[0]

        print("[INFO] Using HuggingFace embedder.")
        return HfEmbeddings()
    else:
        from langchain_ollama import OllamaEmbeddings
        print("[INFO] Using OllamaEmbeddings.")
        return OllamaEmbeddings(model=LOCAL_MODEL_NAME)


def main():
    parser = argparse.ArgumentParser(description="Simplified RAG System Script")
    parser.add_argument("--prompt_file", required=True, help="Path to system prompt template text file.")
    parser.add_argument("--questions_file", required=True, help="Path to JSON file with questions in new structure (each key maps to an object with 'question' and 'answer').")
    parser.add_argument("--context_file", required=True, help="Path to context file (Markdown or PDF).")
    parser.add_argument("--output_file", required=True, help="Path to output JSON file with answers.")
    parser.add_argument("--use_api", action="store_true", help="Use OpenRouter API instead of local model.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached embeddings if available.")
    parser.add_argument("--bm25_weight", type=float, default=0.5, help="Weight for BM25 retriever; semantic retriever weight will be (1 - bm25_weight).")
    parser.add_argument("--hf_emb", action="store_true", help="Use HuggingFace embedder instead of current embedder.")
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        template = f.read().strip()
    print("[INFO] Loaded system prompt template.")

    with open(args.questions_file, "r", encoding="utf-8") as f:
        questions_dict = json.load(f)
    print(f"[INFO] Loaded {len(questions_dict)} questions.")

    documents = load_documents(args.context_file)
    if not documents:
        print("[ERROR] No documents loaded. Exiting.")
        return

    file_hash = hash_file(args.context_file)
    embeddings = get_embeddings(use_hf=args.hf_emb)
    semantic_retriever = build_cached_faiss_retriever(documents, file_hash, embeddings, use_cache=args.use_cache)
    bm25_retriever = build_bm25_retriever(documents)
    # Set retriever weights: semantic weight = (1 - bm25_weight), bm25 weight = bm25_weight.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[1 - args.bm25_weight, args.bm25_weight]
    )

    results = {}
    for key, q_obj in tqdm(questions_dict.items(), desc="Processing questions", total=len(questions_dict)):
        question_text = q_obj.get("question", "")
        retrieved_docs = hybrid_retriever.invoke(question_text)
        if args.use_api:
            if not OPENROUTER_API_KEY:
                print("[ERROR] OPENROUTER_API_KEY not set. Exiting.")
                return
            answer = answer_question_with_api(question_text, retrieved_docs, template)
        else:
            answer = answer_question_with_ollama(question_text, retrieved_docs, template)
        answer = fix_formulas(answer)
        answer = extract_final_answer(answer)
        print(f"[INFO] Key: {key}, Generated Answer: {answer}, Correct Answer: {q_obj.get('answer', None)}")
        results[key] = {"generated_answer": answer, "correct_answer": q_obj.get("answer", None)}

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved answers to {args.output_file}")

if __name__ == "__main__":
    main()
