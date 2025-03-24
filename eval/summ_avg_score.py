#!/usr/bin/env python
import os
import json
import argparse

def aggregate_similarity_per_file(output_json, source_files):
    results = {}
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_similarity_1 = 0.0
        total_similarity_2 = 0.0
        count = 0
        for key, metrics in data.items():
            total_similarity_1 += metrics.get("similarity_1", 0)
            total_similarity_2 += metrics.get("similarity_2", 0)
            count += 1
        if count > 0:
            avg_similarity_1 = total_similarity_1 / count
            avg_similarity_2 = total_similarity_2 / count
        else:
            avg_similarity_1 = 0
            avg_similarity_2 = 0
        results[os.path.basename(file_path)] = {
            "average_similarity_1": avg_similarity_1,
            "average_similarity_2": avg_similarity_2
        }
    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    print(f"Aggregated similarity metrics saved to {output_json}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate average similarity_1 and similarity_2 metrics per file from multiple JSON files."
    )
    parser.add_argument("--output_json", required=True, help="Name of the output JSON file")
    parser.add_argument("source_files", nargs="+", help="One or more source JSON files")
    args = parser.parse_args()
    aggregate_similarity_per_file(args.output_json, args.source_files)

if __name__ == "__main__":
    main()
