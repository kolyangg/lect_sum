#!/usr/bin/env python
import os
import json
import argparse

def aggregate_correct_answers(output_json, source_files):
    results = {}
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        correct_count = 0
        # Iterate over each top-level key (e.g. processed_frame_014825.jpg)
        for frame_key, qa in data.items():
            # For each question in the frame
            for q_num, answer_obj in qa.items():
                if answer_obj.get("answer_given") == answer_obj.get("correct_answer"):
                    correct_count += 1
        # Use only the basename of the file as the key in the output.
        results[os.path.basename(file_path)] = correct_count
    # Write aggregated results to output JSON.
    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    print(f"Aggregated results saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate correct answers from multiple source JSON files."
    )
    parser.add_argument("--output_json", required=True, help="Name of the output JSON file")
    parser.add_argument("source_files", nargs="+", help="One or more source JSON files")
    args = parser.parse_args()
    aggregate_correct_answers(args.output_json, args.source_files)

if __name__ == "__main__":
    main()
