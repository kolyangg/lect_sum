#!/usr/bin/env python3
import gdown
import argparse
import os
import re

def convert_gdrive_url(url: str) -> str:
    """
    Convert a Google Drive sharing link to the format required by gdown.
    
    Args:
        url: Google Drive sharing link.
        
    Returns:
        Converted URL suitable for gdown.
    """
    # Extract file ID using regex for /d/<FILE_ID>
    file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if file_id_match:
        file_id = file_id_match.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    # Fallback: try matching an id=<FILE_ID> parameter
    file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if file_id_match:
        file_id = file_id_match.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return url

def ensure_directory_exists(filepath: str) -> None:
    """Ensure that the directory for the given filepath exists."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def download_file(url: str, output: str) -> bool:
    """
    Download a file from Google Drive using gdown.
    
    Args:
        url: Google Drive file URL.
        output: Destination file path.
        
    Returns:
        True if download was successful, False otherwise.
    """
    try:
        ensure_directory_exists(output)
        # Convert the URL into a direct download link.
        download_url = convert_gdrive_url(url)
        # Use fuzzy=True to handle typical sharing links.
        gdown.download(download_url, output, quiet=False, fuzzy=True)
        return True
    except Exception as e:
        print(f"Error downloading {url} to {output}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download a file from Google Drive and save it as testing_video_manual.webm"
    )
    parser.add_argument("url", type=str, help="Google Drive file sharing URL")
    args = parser.parse_args()

    output_file = "testing_video_manual.webm"
    print(f"Downloading file to {output_file}...")
    if download_file(args.url, output_file):
        print("Download completed successfully.")
    else:
        print("Download failed.")

if __name__ == "__main__":
    main()
