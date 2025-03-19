import os
import sys
import argparse
import re
import json
import imagehash
from PIL import Image

def get_good_images(img_class_file, images_folder):
    """
    If img_class_file is provided, read the classification file (e.g., images_classified.txt)
    and return a list of full image paths for those images classified as "good".
    Expected file format: "processed_frame_XXXXXX.jpg: classification"
    
    If img_class_file is not provided, returns all image paths from images_folder
    (only files ending with .jpg, .jpeg, or .png).
    """
    good_images = []
    
    if img_class_file:
        pattern = re.compile(r'processed_frame_(\d+)\.jpg')
        with open(img_class_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                image_name, classification = line.split(":", 1)
                if classification.strip().lower() == "good":
                    image_name = image_name.strip()
                    full_path = os.path.join(images_folder, image_name)
                    if os.path.exists(full_path):
                        good_images.append(full_path)
                    else:
                        print(f"Warning: {full_path} not found.")
    else:
        # If no classification file is provided, use all images in the folder.
        for file in os.listdir(images_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                good_images.append(os.path.join(images_folder, file))
    
    return good_images

def group_images_by_similarity(good_images, threshold=5):
    """
    Group images by similarity based on perceptual hash.
    The images are first sorted alphabetically by filename so that the first file becomes group 0.
    Then each image is compared (using hamming distance) to already grouped images.
    If the distance is <= threshold, it is assigned the same group; otherwise, a new group number is created.
    Returns a dictionary mapping image filename (basename) to group index (starting from 0).
    """
    # Sort the list of good images alphabetically by basename.
    good_images_sorted = sorted(good_images, key=lambda x: os.path.basename(x))
    
    groups = []  # list of tuples: (group_index, hash)
    image_to_group = {}  # mapping from image filename (basename) to group index
    next_index = 0

    for image_path in good_images_sorted:
        try:
            with Image.open(image_path) as img:
                hash_val = imagehash.phash(img)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

        assigned = False
        # Compare against already grouped images
        for group_index, group_hash in groups:
            if hash_val - group_hash <= threshold:
                image_to_group[os.path.basename(image_path)] = group_index
                assigned = True
                break
        if not assigned:
            # Create a new group for this image
            image_to_group[os.path.basename(image_path)] = next_index
            groups.append((next_index, hash_val))
            next_index += 1

    return image_to_group

def main():
    parser = argparse.ArgumentParser(
        description="Merge duplicate images based on similarity. "
                    "Only images classified as 'good' are processed if an image classification file is provided. "
                    "Otherwise, all images in the folder are used."
    )
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing images.")
    parser.add_argument("--img_class_file", type=str, required=False, default=None,
                        help="Optional path to the image classification file (e.g., images_classified.txt). "
                             "If not provided, all images in images_folder are used.")
    parser.add_argument("--output_file", type=str, default="unique_img.txt",
                        help="Output file for the unique images dictionary (default: unique_img.txt).")
    args = parser.parse_args()

    good_images = get_good_images(args.img_class_file, args.images_folder)
    if not good_images:
        print("No good images found.")
        sys.exit(1)

    image_to_group = group_images_by_similarity(good_images, threshold=10)
    # Save the output file with keys sorted in increasing order.
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(image_to_group, f, indent=4, sort_keys=True)
    print(f"Unique images dictionary saved to {args.output_file}")

if __name__ == "__main__":
    main()
