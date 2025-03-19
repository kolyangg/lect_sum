#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
from PIL import Image

def get_images_dict(images_folder):
    """
    Scan images_folder for files with names like processed_frame_*.jpg,
    extract the frame number, and return a dict mapping frame number (int) to file path.
    """
    pattern = re.compile(r'processed_frame_(\d+)\.jpg')
    images_dict = {}
    files = glob.glob(os.path.join(images_folder, "processed_frame_*.jpg"))
    for file in files:
        basename = os.path.basename(file)
        m = pattern.search(basename)
        if m:
            frame_num = int(m.group(1))
            images_dict[frame_num] = file
    return images_dict

def get_good_frames(img_class_file, images_folder):
    """
    Read the classification file (e.g., images_classified.txt) and return a set of frame numbers
    for which the classification is "good".
    Expected file format: "processed_frame_XXXXXX.jpg: classification"
    """
    good_frames = set()
    pattern = re.compile(r'processed_frame_(\d+)\.jpg')
    with open(img_class_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            image_name, classification = line.split(":", 1)
            if classification.strip().lower() == "good":
                m = pattern.search(image_name)
                if m:
                    frame_num = int(m.group(1))
                    good_frames.add(frame_num)
    return good_frames

def create_transcript_json(json_file,
                           images_folder,
                           img_class_file=None,
                           group_duplicates=False,
                           unique_img_file=None,
                           start_cut=0,
                           end_cut=0,
                           transcript_json_file="output_transcript.json"):
    """
    Reads a JSON transcript mapping (frame -> text), optionally filters by 'good' frames (if img_class_file provided),
    optionally groups duplicates (if group_duplicates=True with unique_img_file),
    and finally writes out a JSON file mapping 'processed_frame_XXXXXX.jpg' to the transcript text.

    :param json_file: Path to the JSON file with transcript mapping (frame -> text).
    :param images_folder: Path to the folder containing image files.
    :param img_class_file: Optional path to image classification file (to filter frames).
    :param group_duplicates: Whether to group frames by a 'unique image' file.
    :param unique_img_file: Path to the unique image file (required if group_duplicates=True).
    :param start_cut: Number of slides to ignore from the beginning.
    :param end_cut: Number of slides to ignore from the end.
    :param transcript_json_file: Output path for the final JSON mapping.
    """
    # Load transcript mapping and ensure keys are ints
    with open(json_file, "r", encoding="utf-8") as f:
        transcript_mapping = json.load(f)
    transcript_mapping = {int(k): v for k, v in transcript_mapping.items()}

    # Get images dict mapping frame number to image file path
    images_dict = get_images_dict(images_folder)

    # Prepare the final JSON mapping
    transcript_mapping_json = {}

    if group_duplicates:
        if unique_img_file is None:
            print("Error: When group_duplicates is True, you must supply --unique_img_file")
            return

        # Read unique_img.txt, expected to be a JSON dict: { "processed_frame_xxxxxx.jpg": group_index, ... }
        with open(unique_img_file, "r", encoding="utf-8") as f:
            unique_img_dict = json.load(f)

        # Group transcript frames by group index
        group_to_frames = {}
        for frame, text in transcript_mapping.items():
            fname = f"processed_frame_{frame:06d}.jpg"
            if fname in unique_img_dict:
                group = unique_img_dict[fname]
                group_to_frames.setdefault(group, []).append((frame, text))

        # Sort groups, then apply start_cut and end_cut
        sorted_groups = sorted(group_to_frames.keys())
        if end_cut > 0:
            effective_groups = sorted_groups[start_cut:-end_cut] if len(sorted_groups) > (start_cut + end_cut) else []
        else:
            effective_groups = sorted_groups[start_cut:]

        # For each group, pick the representative frame (lowest frame number),
        # merge transcript text for all frames in that group
        for group in effective_groups:
            occurrences = sorted(group_to_frames[group], key=lambda x: x[0])
            rep_frame = occurrences[0][0]

            # Check if representative frame has an actual image
            if rep_frame not in images_dict:
                print(f"Warning: No image found for frame {rep_frame} in group {group}")
                continue

            # Merge transcript text for all frames in the group
            merged_text_parts = []
            for idx, (frame, text) in enumerate(occurrences, start=1):
                merged_text_parts.append(f"PART_{idx}: {text.strip()}")
            merged_text = "\n".join(merged_text_parts)

            # Map the representative image to the merged text
            rep_filename = f"processed_frame_{rep_frame:06d}.jpg"
            transcript_mapping_json[rep_filename] = merged_text

    else:
        # Normal processing: if classification file is provided, filter for "good" frames; otherwise, use all
        if img_class_file:
            good_frames = get_good_frames(img_class_file, images_folder)
        else:
            good_frames = set(images_dict.keys())

        # Sort frames that appear in both transcript_mapping and good_frames
        sorted_frames = sorted([frame for frame in transcript_mapping.keys() if frame in good_frames])
        if end_cut > 0:
            effective_frames = sorted_frames[start_cut:-end_cut] if len(sorted_frames) > (start_cut + end_cut) else []
        else:
            effective_frames = sorted_frames[start_cut:]

        # For each frame, map "processed_frame_XXXXXX.jpg" to text
        for frame in effective_frames:
            if frame not in images_dict:
                print(f"Warning: No image found for frame {frame}")
                continue
            text = transcript_mapping[frame].strip()
            rep_filename = f"processed_frame_{frame:06d}.jpg"
            transcript_mapping_json[rep_filename] = text

    # Finally, save the transcript mapping JSON
    with open(transcript_json_file, "w", encoding="utf-8") as jf:
        json.dump(transcript_mapping_json, jf, indent=4, ensure_ascii=False)
    print(f"Transcript JSON mapping saved as {transcript_json_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a JSON mapping of images to transcript text. "
                    "Optionally group duplicate images and filter by classification."
    )
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file (e.g., output.json) with transcript mapping.")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing image files.")
    parser.add_argument("--img_class_file", type=str, required=False, default=None,
                        help="Optional path to the image classification file (e.g., images_classified.txt). "
                             "If not provided, all images in images_folder are used (if in transcript).")
    parser.add_argument("--group_duplicates", action="store_true",
                        help="If set, group duplicate images using unique_img.txt and merge transcript parts.")
    parser.add_argument("--unique_img_file", type=str, default=None,
                        help="Path to the unique image file (unique_img.txt), required if group_duplicates is set.")
    parser.add_argument("--start_cut", type=int, default=0,
                        help="Number of slides to ignore from the beginning (default: 0).")
    parser.add_argument("--end_cut", type=int, default=0,
                        help="Number of slides to ignore from the end (default: 0).")
    parser.add_argument("--output_transcript_json", type=str, default="output_transcript.json",
                        help="Path for the output transcript JSON file (default: output_transcript.json).")

    args = parser.parse_args()

    create_transcript_json(
        json_file=args.json_file,
        images_folder=args.images_folder,
        img_class_file=args.img_class_file,
        group_duplicates=args.group_duplicates,
        unique_img_file=args.unique_img_file,
        start_cut=args.start_cut,
        end_cut=args.end_cut,
        transcript_json_file=args.output_transcript_json
    )

if __name__ == "__main__":
    main()
