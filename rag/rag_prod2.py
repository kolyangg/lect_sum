#!/usr/bin/env python
import os
import argparse
import json
import requests
import pickle
import hashlib
import re
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from nltk.tokenize import word_tokenize
import nltk
import ast

# Ensure nltk tokenizer is available
nltk.download("punkt")

# Define directories for FAISS cache and uploaded files
FAISS_DB_PATH = "_faiss_cache"
os.makedirs(FAISS_DB_PATH, exist_ok=True)
DATA_DIRECTORY = "_data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Define RAG prompt (Markdown-friendly Russian)
TEMPLATE = """
Вы — полезный AI-ассистент, который предоставляет ответы в **Markdown-формате**.
- Используйте **жирный шрифт** для важных понятий.
- Используйте `inline code` для названий функций и программных терминов.
- Используйте нотацию LaTeX (`$...$` для встроенных формул, `$$...$$` для блочных формул) для корректного форматирования математических выражений.
- **Не используйте китайские символы в ответе.**

---

### **Контекст**
{context}

### **Вопрос**
{question}

### **Ответ в Markdown-формате:**
"""

# Model selection
LOCAL_MODEL_NAME = "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q6_K_L"
API_MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def fix_formulas(text):
    """
    Convert LaTeX delimiters so that formulas are rendered correctly by Streamlit.
    This workaround replaces:
    - "\[" with "$$" and "\]" with "$$"
    - "\(" with "$" and "\)" with "$"
    """
    new_text = (
        text.replace(r"\[", "$$")
            .replace(r"\]", "$$")
            .replace(r"\(", "$")
            .replace(r"\)", "$")
    )
    return new_text


def hash_file(file_path):
    """Generate a unique hash for a file based on its contents."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def extract_images_from_markdown(text):
    """Extracts base64-encoded images from Markdown text.
    Returns a list of base64 image strings.
    """
    image_pattern = r"!\[.*?\]\((data:image\/.*?;base64,.*?)\)"
    return re.findall(image_pattern, text)


def load_and_split_markdown_file(file_path):
    """Loads and splits a Markdown file using recursive splitting and extracts images."""
    st.write("[INFO] Loading Markdown file...")
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()

    # Set source metadata and extract images for each document.
    for doc in documents:
        if "source" not in doc.metadata:
            doc.metadata["source"] = file_path
        images = extract_images_from_markdown(doc.page_content)
        doc.metadata["images"] = images

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_sections = []
    for doc in documents:
        sections = text_splitter.split_documents([doc])
        for section in sections:
            section.metadata["source"] = doc.metadata["source"]
            section.metadata["images"] = doc.metadata.get("images", [])
        all_sections.extend(sections)
    st.write(f"[INFO] Loaded and split {len(all_sections)} sections from Markdown file.")
    return all_sections


def load_and_split_pdf(file_path):
    """Loads and splits PDF documents using a recursive character splitter."""
    st.write("[INFO] Loading PDF file...")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def load_documents(file_path):
    """Dispatches document loading and splitting based on file extension."""
    if file_path.endswith(".md"):
        return load_and_split_markdown_file(file_path)
    elif file_path.endswith(".pdf"):
        return load_and_split_pdf(file_path)
    else:
        st.error("Unsupported file format.")
        return []


def build_cached_faiss_retriever(documents, file_hash, use_cache):
    """Builds or loads a FAISS retriever with caching."""
    faiss_index_path = os.path.join(FAISS_DB_PATH, f"{file_hash}.faiss")
    docstore_path = os.path.join(FAISS_DB_PATH, f"{file_hash}_docs.pkl")
    embeddings = OllamaEmbeddings(model=LOCAL_MODEL_NAME)

    if use_cache and os.path.exists(faiss_index_path) and os.path.exists(docstore_path):
        st.write("✅ Loaded cached embeddings from FAISS.")
        vector_store = FAISS.load_local(faiss_index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        with open(docstore_path, "rb") as f:
            vector_store.docstore = pickle.load(f)
    else:
        st.write("🔄 Generating embeddings... (This may take a while)")
        progress_bar = st.progress(0)
        vector_store = FAISS.from_documents(documents, embeddings)
        progress_bar.progress(100)
        st.write("✅ Embeddings saved for future use.")
        vector_store.save_local(faiss_index_path)
        with open(docstore_path, "wb") as f:
            pickle.dump(vector_store.docstore, f)
    return vector_store.as_retriever()


def build_bm25_retriever(documents):
    """Builds a BM25 retriever for keyword-based search."""
    return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize)


def find_relevant_images(retrieved_docs):
    """Finds images in the retrieved Markdown sections.
    Returns a list of unique base64 image strings.
    """
    images = []
    for doc in retrieved_docs:
        if "images" in doc.metadata and doc.metadata["images"]:
            images.extend(doc.metadata["images"])
    return list(set(images))


def force_find_images(answer_text, file_path, debug=False):
    """
    Parses the answer text for image filenames (e.g. 'img-23.jpeg') and searches
    the raw Markdown file for all matching base64 image embeddings.
    Returns a list of unique base64 image strings.
    """
    if debug:
        st.write("[DEBUG] Answer text for force_find_images:", answer_text)
    pattern = r"(img-\d+\.\w+)"
    matches = re.findall(pattern, answer_text)
    images_found = []
    if matches:
        if debug:
            st.write("[DEBUG] Found target image filenames in answer:", matches)
        with open(file_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        for target in matches:
            pattern_img = fr"!\[{re.escape(target)}\]\((data:image\/[^\)]+)\)"
            found = re.findall(pattern_img, md_text)
            if found:
                if debug:
                    st.write(f"[DEBUG] Found image(s) for {target}: {found}")
                images_found.extend(found)
            else:
                if debug:
                    st.write(f"[DEBUG] No image found for {target}.")
    else:
        if debug:
            st.write("[DEBUG] No image filenames found in answer text.")
    return list(set(images_found))


def answer_question_with_ollama(question, documents):
    """Generates an answer using the local Ollama model."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = OllamaLLM(model=LOCAL_MODEL_NAME)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


def answer_question_with_api(question, documents):
    """Generates an answer using the hosted DeepSeek R1 API."""
    context = "\n\n".join([doc.page_content for doc in documents])
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


def save_response_to_markdown(response_text, response_file, images):
    """Saves the answer in a Markdown file, including any relevant images."""
    st.write(f"[INFO] Saving response to {response_file}...")
    with open(response_file, "w", encoding="utf-8") as f:
        f.write("# Response\n\n")
        f.write(response_text + "\n\n")
        if images:
            f.write("## Related Images\n\n")
            for img in images:
                if not img.startswith("data:"):
                    img = "data:image/jpeg;base64," + img
                f.write(f"![Embedded Image]({img})\n\n")
    st.write("[INFO] Response successfully saved.")


def main():
    parser = argparse.ArgumentParser(
        description="RAG System with Image Retrieval, Forced Image Option, Term Selection, and Follow-up Questions"
    )
    parser.add_argument("--terms_list", required=False,
                        help="Path to a text file containing a Python list of terms")
    args = parser.parse_args()

    st.title("📄 RAG System with Image Retrieval & Term Selection")
    st.write("Upload a Markdown file to ask questions and optionally include relevant images from the document.")

    # Editable system prompt field (default provided)
    default_prompt = "Основываясь на информации из учебника, объясни этой термин с примерами и предоставь релевантное изображение, если такое есть: "
    editable_prompt = st.text_area("Настройте системный скрипт:", value=default_prompt, height=100)

    # Load terms if provided
    terms = []
    if args.terms_list:
        try:
            with open(args.terms_list, "r", encoding="utf-8") as f:
                terms_str = f.read()
            terms = ast.literal_eval(terms_str)
            if not isinstance(terms, list):
                st.error("The terms list file does not contain a Python list.")
                terms = []
        except Exception as e:
            st.error(f"Failed to parse terms list file: {e}")
            terms = []

    # Sidebar options (default True)
    use_api = st.sidebar.checkbox("Use OpenRouter API", value=True)
    use_cache = st.sidebar.checkbox("Use Cached Embeddings", value=True)
    include_img = st.sidebar.checkbox("Include Images in Output", value=True)
    force_img = st.sidebar.checkbox("Force Display Images", value=True)
    output_md = st.sidebar.text_input("Output File (optional, e.g., output.md)", value="")

    uploaded_file = st.file_uploader("Upload Markdown File", type=["md"], accept_multiple_files=False)

    if uploaded_file:
        # Save uploaded file to DATA_DIRECTORY
        file_path = os.path.join(DATA_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_hash = hash_file(file_path)
        documents = load_documents(file_path)
        sections = load_and_split_markdown_file(file_path)

        semantic_retriever = build_cached_faiss_retriever(sections, file_hash, use_cache)
        bm25_retriever = build_bm25_retriever(sections)
        hybrid_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

        st.subheader("Выберите термин из списка:")
        # Display terms as buttons in a grid with 4 columns.
        if terms:
            cols = st.columns(4)
            for idx, term in enumerate(terms):
                col = cols[idx % 4]
                if col.button(term, key=f"term_{idx}"):
                    st.session_state.selected_term = term

        custom_term = st.text_input("Или введите свой собственный термин:", value="")

        selected_term = st.session_state.get("selected_term", "")
        final_term = custom_term.strip() if custom_term.strip() != "" else selected_term
        question = editable_prompt + " " + final_term

        st.write("### " + question)

        if question.strip() != editable_prompt:
            st.write("⏳ Processing your query...")
            retrieved_docs = hybrid_retriever.invoke(question)

            # Generate answer using chosen method
            if use_api:
                if not OPENROUTER_API_KEY:
                    st.error("API Key not found! Please set OPENROUTER_API_KEY in environment variables.")
                    return
                answer = answer_question_with_api(question, retrieved_docs)
            else:
                answer = answer_question_with_ollama(question, retrieved_docs)
            
            # Fix formulas in the answer before displaying
            answer = fix_formulas(answer)
            
            # Determine images to display
            images_to_display = []
            if force_img:
                images_to_display = force_find_images(answer, file_path)
            elif include_img:
                images_found = find_relevant_images(retrieved_docs)
                images_to_display = images_found

            # Save output if output_md is provided
            if output_md:
                output_file = os.path.join(DATA_DIRECTORY, output_md)
                save_response_to_markdown(answer, output_file, images_to_display)
                st.write("Response saved in:", output_file)

            st.markdown("### 📄 Response:")
            st.markdown(answer)
            if images_to_display:
                st.markdown("### 📷 Related Images:")
                for img in images_to_display:
                    if not img.startswith("data:"):
                        img = "data:image/jpeg;base64," + img
                    st.image(img)

            # Follow-up question section
            st.markdown("### Задайте уточняющий вопрос:")
            followup = st.text_input("Введите уточняющий вопрос:", key="followup")
            if st.button("Отправить уточняющий вопрос"):
                if followup.strip() != "":
                    followup_question = editable_prompt + " " + followup.strip()
                    st.write("⏳ Processing your follow-up query...")
                    retrieved_docs_followup = hybrid_retriever.invoke(followup_question)
                    if use_api:
                        followup_answer = answer_question_with_api(followup_question, retrieved_docs_followup)
                    else:
                        followup_answer = answer_question_with_ollama(followup_question, retrieved_docs_followup)
                    
                    # Fix formulas in the follow-up answer
                    followup_answer = fix_formulas(followup_answer)
                    
                    # Determine images for follow-up answer
                    images_to_display_followup = []
                    if force_img:
                        images_to_display_followup = force_find_images(followup_answer, file_path)
                    elif include_img:
                        images_found_followup = find_relevant_images(retrieved_docs_followup)
                        images_to_display_followup = images_found_followup

                    st.markdown("### 📄 Follow-up Response:")
                    st.markdown(followup_answer)
                    if images_to_display_followup:
                        st.markdown("### 📷 Related Images:")
                        for img in images_to_display_followup:
                            if not img.startswith("data:"):
                                img = "data:image/jpeg;base64," + img
                            st.image(img)


if __name__ == "__main__":
    main()
