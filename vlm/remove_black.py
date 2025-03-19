#!/usr/bin/env python3
import os
import argparse
import ast  # or 'json' if your file is in JSON format
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Remove rows with >90% black pixels in top 20% of an image.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the folder containing original images.')
    parser.add_argument('--unique_img_file', type=str, required=True,
                        help='Path to the text file containing a dictionary with keys = image file names.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the folder where processed images will be saved.')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Read the dictionary of filenames from the text file
    # (Assumes the file content is something like: {"img1.png": ..., "img2.jpg": ..., ...})
    with open(args.unique_img_file, 'r') as f:
        file_content = f.read()
        # If your unique_img_file is Python dict-like, we can parse with 'ast.literal_eval'.
        # If it's JSON, use 'json.loads' instead.
        try:
            filenames_dict = ast.literal_eval(file_content)
        except Exception as e:
            raise ValueError(f"Could not parse the dictionary from {args.unique_img_file}: {e}")

    # Process each filename in the dictionary
    for filename in filenames_dict.keys():
        input_path = os.path.join(args.input_folder, filename)
        if not os.path.isfile(input_path):
            print(f"Warning: {input_path} does not exist, skipping.")
            continue

        # Open the image
        img = Image.open(input_path).convert('RGB')
        width, height = img.size

        # Determine the cutoff for top 20%
        top_cutoff = int(0.2 * height)

        # Load pixel data for row-by-row analysis
        pixels = img.load()

        # We'll store the rows that we decide to keep here
        kept_rows = []

        # 1) Process rows from the top 20% of the image
        for row in range(top_cutoff):
            row_pixels = [pixels[col, row] for col in range(width)]
            # Count how many are black
            black_count = sum(1 for px in row_pixels if px == (0, 0, 0))
            # Calculate fraction of black pixels
            fraction_black = black_count / float(width)

            # Keep this row only if black pixels <= 90%
            if fraction_black <= 0.90:
                kept_rows.append(row_pixels)

        # 2) Keep all rows from the remaining 80%
        for row in range(top_cutoff, height):
            row_pixels = [pixels[col, row] for col in range(width)]
            kept_rows.append(row_pixels)

        # Create a new image with the kept rows
        new_height = len(kept_rows)
        new_img = Image.new('RGB', (width, new_height))

        for r, row_pixels in enumerate(kept_rows):
            for c, px in enumerate(row_pixels):
                new_img.putpixel((c, r), px)

        # Save the resulting image to output_folder
        output_path = os.path.join(args.output_folder, filename)
        new_img.save(output_path)
        print(f"Processed and saved: {output_path}")

if __name__ == '__main__':
    main()
