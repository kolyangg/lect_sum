import json
import re
import argparse

def remove_consecutive_sentence_duplicates(text):
    """
    Split the text into sentences using punctuation (. ! ?),
    then remove consecutive duplicates.
    """
    # Split on punctuation that ends a sentence followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cleaned_sentences and s == cleaned_sentences[-1]:
            continue
        cleaned_sentences.append(s)
    return " ".join(cleaned_sentences)

def remove_repeated_phrases_in_sentence(sentence):
    """
    Splits a sentence by commas and removes consecutive duplicate phrases.
    """
    parts = sentence.split(',')
    cleaned_parts = []
    for part in parts:
        part_clean = part.strip()
        if part_clean:
            if cleaned_parts and part_clean == cleaned_parts[-1]:
                continue
            cleaned_parts.append(part_clean)
    # Join parts with a comma and a space.
    return ", ".join(cleaned_parts)

def clean_text(text):
    """
    Cleans a transcript text by:
      1. Removing repeated phrases within each sentence.
      2. Reassembling the text and removing consecutive duplicate sentences.
    """
    # Split text into sentences using punctuation delimiters.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    for sentence in sentences:
        if sentence.strip():
            cleaned_sentence = remove_repeated_phrases_in_sentence(sentence)
            cleaned_sentences.append(cleaned_sentence)
    # Rejoin sentences and then remove duplicate sentences.
    joined_text = " ".join(cleaned_sentences)
    cleaned_text = remove_consecutive_sentence_duplicates(joined_text)
    return cleaned_text

def process_json(input_json, output_json):
    """
    Loads the JSON file with transcript texts, cleans each text,
    and writes the cleaned mapping to a new JSON file.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = {}
    for key, text in data.items():
        cleaned_text = clean_text(text)
        cleaned_data[key] = cleaned_text

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"Cleaned JSON saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove hallucinated repeating sentences/phrases from transcript JSON."
    )
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to input JSON file (e.g., output.json)")
    parser.add_argument("--output_json", type=str, default="output_cleaned.json",
                        help="Path to output JSON file (default: output_cleaned.json)")
    args = parser.parse_args()
    process_json(args.input_json, args.output_json)
