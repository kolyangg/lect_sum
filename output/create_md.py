
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
    # Use glob to find matching files
    files = glob.glob(os.path.join(images_folder, "processed_frame_*.jpg"))
    for file in files:
        basename = os.path.basename(file)
        m = pattern.search(basename)
        if m:
            frame_num = int(m.group(1))
            images_dict[frame_num] = file
    return images_dict

def create_markdown(json_file, images_folder, output_md="output.md"):
    """
    Read transcript mapping from json_file and image files from images_folder.
    For each frame (key in the json) find the corresponding image,
    resize it to 50% of its original width and height (by embedding an HTML tag with explicit pixel values),
    and write a markdown file with the image followed by its transcript text.
    """
    # Load the transcript mapping
    with open(json_file, "r", encoding="utf-8") as f:
        transcript_mapping = json.load(f)
    
    # Convert keys to int for proper sorting (they may be strings)
    transcript_mapping = {int(k): v for k, v in transcript_mapping.items()}
    
    # Get images dict mapping frame number to image file path
    images_dict = get_images_dict(images_folder)
    
    # Open the markdown file for writing
    with open(output_md, "w", encoding="utf-8") as md:
        # Process frames in sorted order
        for frame in sorted(transcript_mapping.keys()):
            text = transcript_mapping[frame].strip()
            if frame not in images_dict:
                print(f"Warning: No image found for frame {frame}")
                continue
            img_path = images_dict[frame]
            # Open image to get its size
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
            new_width = width // 2
            new_height = height // 2

            # Write image with HTML tag using explicit width/height (in pixels)
            # The src path is relative; adjust if necessary.
            md.write(f'<img src="{img_path}" width="{new_width}" height="{new_height}" alt="Frame {frame}"/>\n\n')
            # Write transcript text below the image
            md.write(text + "\n\n")
            # Optional separator
            md.write("---\n\n")
    print(f"Markdown file saved as {output_md}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a markdown file with images and corresponding transcript text."
    )
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file (e.g., output.json) with transcript mapping.")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing image files.")
    parser.add_argument("--output_md", type=str, default="output.md",
                        help="Path for the output markdown file (default: output.md).")
    args = parser.parse_args()
    create_markdown(args.json_file, args.images_folder, args.output_md)

if __name__ == "__main__":
    main()
