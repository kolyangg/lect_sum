#!/usr/bin/env python
import os
import re
import json
import argparse

def extract_json_from_text(text):
    """
    Extracts JSON content from a string by looking for a markdown code block tagged with "json".
    Returns the JSON string if found, else None.
    """
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a JSON file with extraneous text and markdown code blocks to extract valid JSON content."
    )
    parser.add_argument("--input_json", required=True,
                        help="Input JSON file with keys mapping to text that contains JSON code blocks.")
    parser.add_argument("--output_json", required=True,
                        help="Output JSON file to store the cleaned JSON objects.")
    args = parser.parse_args()

    # Load the input JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = {}
    for key, value in data.items():
        # Try to extract the JSON portion from the string value.
        json_str = extract_json_from_text(value)
        if json_str:
            try:
                # Parse the JSON string into a Python object.
                parsed_json = json.loads(json_str)
                cleaned_data[key] = parsed_json
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for key {key}: {e}")
        else:
            print(f"No JSON code block found for key {key}; skipping.")
    
    # Write the cleaned JSON data to the output file.
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Cleaned JSON written to {args.output_json}")

if __name__ == "__main__":
    main()
