#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
from mistralai import Mistral

def ask_question_about_pdf_for_terms(pdf_file):
    """
    Uploads the given PDF file to the Mistral API and asks a question about its content.
    The question is:
      "This is a lecture on Reinforcement Learning. Provide in Python list format a list of all the specific technical terms mentioned in the document."
    
    Parameters:
      pdf_file (str or Path): Path to the PDF file.
    
    Returns:
      str: The content of the response from Mistral.
    """
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file '{pdf_file}' not found.")

    # Retrieve the API key from the environment.
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY environment variable is not set.")

    # Specify the model to use.
    model = "mistral-small-latest"

    # Initialize the Mistral client.
    client = Mistral(api_key=api_key)

    # Upload the PDF file for OCR processing.
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_path.stem,
            "content": pdf_path.read_bytes(),
        },
        purpose="ocr",
    )

    # Get a signed URL for the uploaded PDF (expires in 1 minute).
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    # Define the question to extract specific technical terms.
    question_text = (
        "This is a lecture on Reinforcement Learning. "
        "Provide in Python list format a list of all the specific technical terms mentioned in the document."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text},
                {"type": "document_url", "document_url": signed_url.url}
            ]
        }
    ]

    # Call the chat completion API.
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    response_text = chat_response.choices[0].message.content
    return response_text

def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF lecture on Reinforcement Learning using Mistral API and extract specific technical terms as a Python list."
    )
    parser.add_argument("--pdf_file", type=str, required=True,
                        help="Path to the PDF file containing the lecture.")
    parser.add_argument("--output_txt", type=str, default="terms_output.txt",
                        help="Path for the output text file (default: terms_output.txt).")
    args = parser.parse_args()

    try:
        answer = ask_question_about_pdf_for_terms(args.pdf_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the answer to the output text file.
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(answer)
    print(f"Output saved to {args.output_txt}")

if __name__ == "__main__":
    main()
