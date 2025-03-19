#!/usr/bin/env python3
import os
import re
import json
import argparse
import sys
from PIL import Image

def get_grouped_images(img_class_file):
    """
    Reads the image classification file (expected to be a JSON dictionary mapping
    image filenames to group indices). Returns a dictionary mapping each group index to the representative image filename,
    chosen as the image with the smallest frame number.
    """
    with open(img_class_file, "r", encoding="utf-8") as f:
        img_class_dict = json.load(f)
    
    groups = {}  # group -> (frame_number, image_filename)
    pattern = re.compile(r'processed_frame_(\d+)\.jpg')
    
    for img_name, group in img_class_dict.items():
        m = pattern.search(img_name)
        if not m:
            continue
        try:
            frame_num = int(m.group(1))
        except Exception:
            continue
        
        if group not in groups:
            groups[group] = (frame_num, img_name)
        else:
            if frame_num < groups[group][0]:
                groups[group] = (frame_num, img_name)
    
    # Return a dict mapping group -> representative image filename
    rep_images = {group: img_name for group, (frame_num, img_name) in groups.items()}
    return rep_images

def create_pdf_from_images(image_paths, output_file):
    """
    Opens all images from image_paths, converts them to RGB (if necessary), and saves them as a multi-page PDF.
    """
    if not image_paths:
        print("No images to process.")
        sys.exit(1)
    
    image_list = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_list.append(img.copy())
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
    
    if not image_list:
        print("No images could be opened.")
        sys.exit(1)
    
    first_image = image_list[0]
    if len(image_list) == 1:
        first_image.save(output_file, "PDF", resolution=100.0)
    else:
        first_image.save(output_file, "PDF", resolution=100.0, save_all=True, append_images=image_list[1:])
    print(f"PDF saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a PDF from images in a folder. "
                    "If an image classification file is provided, include one representative image per class (ordered by class name)."
    )
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing image files.")
    parser.add_argument("--img_class_file", type=str, required=False,
                        help="Optional: Path to the image classification file (e.g., unique_img.txt).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path for the output PDF file.")
    args = parser.parse_args()

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    image_paths = []
    
    if args.img_class_file:
        rep_images = get_grouped_images(args.img_class_file)
        if not rep_images:
            print("No grouped images found in the classification file.")
            sys.exit(1)
        # Process one image per group, sorting groups by their names
        for group in sorted(rep_images.keys(), key=lambda x: str(x)):
            img_name = rep_images[group]
            img_path = os.path.join(args.images_folder, img_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                print(f"Warning: Image {img_path} not found.")
    else:
        # Process all valid image files in the folder, sorted by filename.
        for filename in sorted(os.listdir(args.images_folder)):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(args.images_folder, filename)
                image_paths.append(img_path)
    
    create_pdf_from_images(image_paths, args.output_file)

if __name__ == "__main__":
    main()
