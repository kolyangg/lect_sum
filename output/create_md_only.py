#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
from PIL import Image

IMG_SCALE_FACTOR = 0.75

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

def get_good_frames(img_class_file):
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

def create_markdown(json_file, images_folder, img_class_file=None, unique_img_file=None,
                    output_md="output.md", start_cut=0, end_cut=0):
    """
    Reads a JSON transcript mapping (keys are filenames like "processed_frame_XXXXXX.jpg"),
    optionally filters frames by "good" classification, then:
      - If a unique image file is provided, uses it to pick one representative image per group
        (the image with the lowest frame number in that group).
      - Orders the resulting images by frame number.
    Finally, writes a Markdown file with each image (scaled by IMG_SCALE_FACTOR) and its transcript text.
    """
    # Load transcript mapping and convert keys from "processed_frame_XXXXXX.jpg" to int.
    with open(json_file, "r", encoding="utf-8") as f:
        transcript_mapping = json.load(f)

    pattern = re.compile(r'processed_frame_(\d+)\.jpg')
    def extract_frame_number(key):
        m = pattern.search(key)
        if m:
            return int(m.group(1))
        else:
            raise ValueError(f"Invalid key format: {key}")

    transcript_mapping = {extract_frame_number(k): v for k, v in transcript_mapping.items()}

    # Get available images.
    images_dict = get_images_dict(images_folder)

    # Determine good frames (if classification is provided).
    if img_class_file:
        good_frames = get_good_frames(img_class_file)
    else:
        good_frames = set(images_dict.keys())

    # Filter transcript mapping for frames that are both good and have an image.
    filtered_frames = [frame for frame in transcript_mapping if frame in good_frames and frame in images_dict]

    if unique_img_file:
        # Load the unique image file.
        # Expected to be a JSON mapping: {"processed_frame_XXXXXX.jpg": group_id, ...}
        with open(unique_img_file, "r", encoding="utf-8") as f:
            unique_img_dict = json.load(f)

        # Group frames by group id from unique_img_dict.
        group_to_frames = {}
        for frame in filtered_frames:
            fname = f"processed_frame_{frame:06d}.jpg"
            if fname in unique_img_dict:
                group = unique_img_dict[fname]
                group_to_frames.setdefault(group, []).append((frame, transcript_mapping[frame]))

        # For each group, choose the representative frame (lowest frame number).
        rep_frames = []
        for group, items in group_to_frames.items():
            rep = min(items, key=lambda x: x[0])  # (frame, text)
            rep_frames.append(rep)

        # Sort the representative frames by frame number.
        rep_frames.sort(key=lambda x: x[0])
        # Apply start_cut and end_cut.
        if end_cut > 0 and len(rep_frames) > (start_cut + end_cut):
            rep_frames = rep_frames[start_cut:-end_cut]
        else:
            rep_frames = rep_frames[start_cut:]
    else:
        # No unique image file: simply sort filtered frames.
        filtered_frames.sort()
        if end_cut > 0 and len(filtered_frames) > (start_cut + end_cut):
            filtered_frames = filtered_frames[start_cut:-end_cut]
        else:
            filtered_frames = filtered_frames[start_cut:]

    with open(output_md, "w", encoding="utf-8") as md:
        if unique_img_file:
            for frame, text in rep_frames:
                img_path = images_dict[frame]
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                    continue
                new_width = int(width * IMG_SCALE_FACTOR)
                new_height = int(height * IMG_SCALE_FACTOR)
                md.write(f'<img src="{img_path}" width="{new_width}" height="{new_height}" alt="Frame {frame}" />\n\n')
                md.write(text.strip() + "\n\n")
                md.write("---\n\n")
        else:
            for frame in filtered_frames:
                text = transcript_mapping[frame].strip()
                img_path = images_dict[frame]
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                    continue
                new_width = int(width * IMG_SCALE_FACTOR)
                new_height = int(height * IMG_SCALE_FACTOR)
                md.write(f'<img src="{img_path}" width="{new_width}" height="{new_height}" alt="Frame {frame}" />\n\n')
                md.write(text + "\n\n")
                md.write("---\n\n")
    print(f"Markdown file saved as {output_md}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a markdown file with images and transcript text, "
                    "optionally filtering frames by classification and using a unique image file for de-duplication."
    )
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file with transcript mapping (keys as 'processed_frame_XXXXXX.jpg').")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing image files.")
    parser.add_argument("--img_class_file", type=str, default=None,
                        help="Optional path to the image classification file (e.g., images_classified.txt).")
    parser.add_argument("--unique_img_file", type=str, default=None,
                        help="Optional path to the unique image file (e.g., unique_img.txt).")
    parser.add_argument("--output_md", type=str, default="output.md",
                        help="Path for the output markdown file (default: output.md).")
    parser.add_argument("--start_cut", type=int, default=0,
                        help="Number of frames to ignore from the beginning (default: 0).")
    parser.add_argument("--end_cut", type=int, default=0,
                        help="Number of frames to ignore from the end (default: 0).")

    args = parser.parse_args()

    create_markdown(
        json_file=args.json_file,
        images_folder=args.images_folder,
        img_class_file=args.img_class_file,
        unique_img_file=args.unique_img_file,
        output_md=args.output_md,
        start_cut=args.start_cut,
        end_cut=args.end_cut
    )

if __name__ == "__main__":
    main()
