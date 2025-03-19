#!/usr/bin/env python
import os
import re
import json
import argparse
from pathlib import Path

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image markdown placeholders with embedded base64 image data.
    For each image id, replace occurrences of:
       ![<img_id>](<img_id>)
    with:
       ![<img_id>]({base64_string})
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def save_pdf_as_md(pdf_response, output_path: str, many_pages: bool = False):
    """
    Save the OCR'd PDF pages as markdown.
    
    - If many_pages is True, treat output_path as a directory and save each page
      as a separate markdown file (named page_1.md, page_2.md, etc.).
    - Otherwise, combine all pages into one markdown file saved at output_path.
    
    The pdf_response is expected to have a "pages" key (if a dict) or an attribute "pages".
    Each page is expected to contain a markdown string and a list of images
    (each image having an "id" and "image_base64").
    """
    # Extract pages from pdf_response
    if isinstance(pdf_response, dict):
        pages = pdf_response.get("pages", [])
    else:
        pages = pdf_response.pages

    if many_pages:
        # Ensure output_path exists as a directory.
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i, page in enumerate(pages):
            if isinstance(page, dict):
                images_dict = {img.get("id"): img.get("image_base64") for img in page.get("images", [])}
                md_text = replace_images_in_markdown(page.get("markdown", ""), images_dict)
            else:
                images_dict = {img.id: img.image_base64 for img in page.images}
                md_text = replace_images_in_markdown(page.markdown, images_dict)
            file_path = os.path.join(output_path, f"page_{i+1}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_text)
        print(f"Markdown files saved in directory: {output_path}")
    else:
        combined_md = ""
        for i, page in enumerate(pages):
            if isinstance(page, dict):
                images_dict = {img.get("id"): img.get("image_base64") for img in page.get("images", [])}
                md_text = replace_images_in_markdown(page.get("markdown", ""), images_dict)
            else:
                images_dict = {img.id: img.image_base64 for img in page.images}
                md_text = replace_images_in_markdown(page.markdown, images_dict)
            combined_md += md_text + "\n\n---\n\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_md)
        print(f"Markdown file saved as: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process a PDF file with Mistral OCR and save the result as markdown."
    )
    parser.add_argument("--pdf_file", type=str, required=True,
                        help="Path to the PDF file to process.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output markdown file (if --many_pages is False) or output directory (if --many_pages is True).")
    parser.add_argument("--many_pages", action="store_true", default=False,
                        help="If set, save a markdown file for each page; otherwise, combine all pages into one markdown file.")
    args = parser.parse_args()

    # Get Mistral API key from environment
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        return

    # Initialize the Mistral client.
    from mistralai import Mistral, DocumentURLChunk
    client = Mistral(api_key=api_key)

    pdf_path = Path(args.pdf_file)
    if not pdf_path.is_file():
        print(f"Error: PDF file '{args.pdf_file}' not found.")
        return

    # Upload the PDF for OCR processing.
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_path.stem,
            "content": pdf_path.read_bytes(),
        },
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True
    )

    # Save the markdown output according to the --many_pages flag.
    save_pdf_as_md(pdf_response, args.output, many_pages=args.many_pages)

if __name__ == "__main__":
    main()
