#!/usr/bin/env python
import json
import argparse

def fix_formulas(text):
    """
    Replaces LaTeX delimiters in the input text so that formulas are rendered correctly.
    Converts:
      - "\[" to "$$" and "\]" to "$$"
      - "\(" to "$" and "\)" to "$"
    """
    return (text.replace(r"\[", "$$")
                .replace(r"\]", "$$")
                .replace(r"\(", "$")
                .replace(r"\)", "$"))

def process_item(item):
    """
    Recursively process items in the JSON structure:
      - If item is a string, fix its formulas.
      - If it's a list, process each element.
      - If it's a dict, process each key-value pair.
    Otherwise, return the item unchanged.
    """
    if isinstance(item, str):
        return fix_formulas(item)
    elif isinstance(item, list):
        return [process_item(subitem) for subitem in item]
    elif isinstance(item, dict):
        return {key: process_item(value) for key, value in item.items()}
    else:
        return item

def process_json(input_file, output_file):
    """Load input JSON, process all values to fix formulas, and save to output JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = process_item(data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"Processed JSON saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Fix LATEX formulas in a JSON file.")
    parser.add_argument("--input_json", required=True, help="Path to the input JSON file")
    parser.add_argument("--output_json", required=True, help="Path to the output JSON file")
    args = parser.parse_args()
    process_json(args.input_json, args.output_json)

if __name__ == "__main__":
    main()
