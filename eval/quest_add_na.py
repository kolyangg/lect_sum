#!/usr/bin/env python
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Append a new option to each question in the JSON file."
    )
    parser.add_argument("--input_json", required=True,
                        help="Input JSON file with questions and correct answers.")
    parser.add_argument("--output_json", required=True,
                        help="Output JSON file to store the updated questions.")
    args = parser.parse_args()

    # Load the input JSON file.
    with open(args.input_json, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Define the new option to add.
    new_option = "\nд) Ответ на этой вопрос отсутствует в контексте"

    # Iterate over each key and update each question.
    for key in data:
        for q in data[key]:
            if "question" in data[key][q]:
                data[key][q]["question"] += new_option

    # Save the updated JSON to the output file.
    with open(args.output_json, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

    print(f"Updated JSON saved to {args.output_json}")

if __name__ == "__main__":
    main()
