#!/usr/bin/env python
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate text from JSON values into a single text file."
    )
    parser.add_argument("--json_structured", required=True,
                        help="Input JSON file containing structured summaries")
    parser.add_argument("--output", default="output.txt",
                        help="Output text file (default: output.txt)")
    args = parser.parse_args()

    # Load JSON data
    with open(args.json_structured, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Collect all text values from JSON and join them with a space
    all_text = " ".join(value for value in data.values() if isinstance(value, str))

    # Save the concatenated text to the output file
    with open(args.output, "w", encoding="utf-8") as outfile:
        outfile.write(all_text)

    print(f"Saved concatenated text to {args.output}")

if __name__ == "__main__":
    main()
