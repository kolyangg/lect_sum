#!/usr/bin/env python
import re
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Split question and answer into separate subkeys."
    )
    parser.add_argument("--input_file", required=True,
                        help="Input text file containing key, question, and answer entries.")
    parser.add_argument("--output_file", required=True,
                        help="Output JSON file to store the result.")
    args = parser.parse_args()

    # Read the entire file as text.
    with open(args.input_file, "r", encoding="utf-8") as infile:
        content = infile.read()

    # Use a regex to capture:
    #   1. The key: exactly 2 digits between quotes.
    #   2. The question text: everything inside quotes up to the next quote.
    #   3. The answer text: everything inside quotes after the comma.
    #
    # For example, it matches a pattern like:
    # "01": "Сколько блюд съел обжора в примере с парадоксом обжоры?", "10"
    pattern = r'"(\d{2})":\s*"([^"]+)"\s*,\s*"([^"]+)"'
    matches = re.findall(pattern, content)

    if not matches:
        print("No matches found. Please check the input file format.")
        return

    output_dict = {}
    for key, question, answer in matches:
        output_dict[key] = {"question": question, "answer": answer}

    # Write the result to the output JSON file.
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_dict, outfile, ensure_ascii=False, indent=2)

    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
