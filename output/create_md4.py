import os
import re
import json
import glob
import argparse
from PIL import Image

IMG_SCALE_FACTOR = 0.75
INV_IMG_SCALE_FACTOR = 1.0 / IMG_SCALE_FACTOR

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

def create_markdown(json_file, images_folder, img_class_file, output_md="output.md",
                    group_duplicates=False, unique_img_file=None,
                    start_cut=0, end_cut=0, transcript_json_file=None):
    """
    Read transcript mapping from json_file, image files from images_folder, and image classifications.
    If group_duplicates is False, only frames classified as "good" are processed individually.
    If group_duplicates is True, unique_img.txt is read and for each group only the representative image
    (from the lowest frame number) is used. The transcript text is grouped by occurrence,
    each prefixed with "PART_1", "PART_2", etc.

    The parameters start_cut and end_cut specify the number of slides to ignore from
    the beginning and the end of the sorted set of slides.

    In addition, a JSON file is saved mapping each representative image filename (for the slides that are processed)
    to the transcript text (as printed in the markdown file).
    """
    # Load transcript mapping and ensure keys are ints.
    with open(json_file, "r", encoding="utf-8") as f:
        transcript_mapping = json.load(f)
    transcript_mapping = {int(k): v for k, v in transcript_mapping.items()}

    # Get images dict mapping frame number to image file path.
    images_dict = get_images_dict(images_folder)
    
    transcript_mapping_json = {}  # New mapping to be saved as JSON

    if group_duplicates:
        if unique_img_file is None:
            print("Error: When group_duplicates is True, you must supply --unique_img_file")
            return
        # Read unique_img.txt, expected to be a JSON dict: { "processed_frame_xxxxxx.jpg": group_index, ... }
        with open(unique_img_file, "r", encoding="utf-8") as f:
            unique_img_dict = json.load(f)
        # Group transcript frames by group index.
        group_to_frames = {}
        for frame, text in transcript_mapping.items():
            fname = f"processed_frame_{frame:06d}.jpg"
            if fname in unique_img_dict:
                group = unique_img_dict[fname]
                if group not in group_to_frames:
                    group_to_frames[group] = []
                group_to_frames[group].append((frame, text))
        # Get sorted groups and then apply start_cut and end_cut.
        sorted_groups = sorted(group_to_frames.keys())
        if end_cut > 0:
            effective_groups = sorted_groups[start_cut:-end_cut] if len(sorted_groups) > (start_cut + end_cut) else []
        else:
            effective_groups = sorted_groups[start_cut:]
        # Write markdown with one image per effective group.
        with open(output_md, "w", encoding="utf-8") as md:
            for group in effective_groups:
                occurrences = sorted(group_to_frames[group], key=lambda x: x[0])
                rep_frame = occurrences[0][0]
                if rep_frame not in images_dict:
                    print(f"Warning: No image found for frame {rep_frame} in group {group}")
                    continue
                img_path = images_dict[rep_frame]
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                    continue
                new_width = width // INV_IMG_SCALE_FACTOR
                new_height = height // INV_IMG_SCALE_FACTOR
                md.write(f'<img src="{img_path}" width="{new_width}" height="{new_height}" alt="Frame {rep_frame}" />\n\n')
                transcript_text = ""
                # Write transcript parts for each occurrence.
                for idx, (frame, text) in enumerate(occurrences, start=1):
                    part_line = f"PART_{idx}: {text.strip()}\n\n"
                    md.write(part_line)
                    transcript_text += part_line
                md.write("---\n\n")
                rep_filename = f"processed_frame_{rep_frame:06d}.jpg"
                transcript_mapping_json[rep_filename] = transcript_text
        print(f"Markdown file saved as {output_md}")
    else:
        # Normal processing: if classification file is provided, filter for "good" frames; otherwise, use all frames.
        if img_class_file:
            good_frames = get_good_frames(img_class_file, images_folder)
        else:
            good_frames = set(images_dict.keys())
        sorted_frames = sorted([frame for frame in transcript_mapping.keys() if frame in good_frames])
        if end_cut > 0:
            effective_frames = sorted_frames[start_cut:-end_cut] if len(sorted_frames) > (start_cut + end_cut) else []
        else:
            effective_frames = sorted_frames[start_cut:]
        with open(output_md, "w", encoding="utf-8") as md:
            for frame in effective_frames:
                text = transcript_mapping[frame].strip()
                if frame not in images_dict:
                    print(f"Warning: No image found for frame {frame}")
                    continue
                img_path = images_dict[frame]
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as e:
                    print(f"Error opening {img_path}: {e}")
                    continue
                new_width = width // INV_IMG_SCALE_FACTOR
                new_height = height // INV_IMG_SCALE_FACTOR
                md.write(f'<img src="{img_path}" width="{new_width}" height="{new_height}" alt="Frame {frame}" />\n\n')
                md.write(text + "\n\n")
                md.write("---\n\n")
                rep_filename = f"processed_frame_{frame:06d}.jpg"
                transcript_mapping_json[rep_filename] = text
        print(f"Markdown file saved as {output_md}")

    # Save the transcript mapping JSON if an output path was provided.
    if transcript_json_file:
        with open(transcript_json_file, "w", encoding="utf-8") as jf:
            json.dump(transcript_mapping_json, jf, indent=4, ensure_ascii=False)
        print(f"Transcript JSON mapping saved as {transcript_json_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a markdown file with images and corresponding transcript text. "
                    "Optionally group duplicate images using a unique image file. "
                    "Use --start_cut and --end_cut to ignore a number of slides from the beginning and end. "
                    "Also saves a transcript JSON mapping."
    )
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file (e.g., output.json) with transcript mapping.")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing image files.")
    parser.add_argument("--img_class_file", type=str, required=False, default=None,
                        help="Optional path to the image classification file (e.g., images_classified.txt). "
                             "If not provided, all images in images_folder are used.")
    parser.add_argument("--output_md", type=str, default="output.md",
                        help="Path for the output markdown file (default: output.md).")
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
    
    create_markdown(args.json_file, args.images_folder, args.img_class_file,
                    args.output_md, args.group_duplicates, args.unique_img_file,
                    start_cut=args.start_cut, end_cut=args.end_cut,
                    transcript_json_file=args.output_transcript_json)

if __name__ == "__main__":
    main()
