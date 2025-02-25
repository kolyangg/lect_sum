import os
import cv2
import numpy as np
import torch
from PIL import Image
import random
from collections import defaultdict
import warnings
import argparse

def load_nanodet_model():
    """
    Loads and returns YOLOv5 model for person detection.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model

def detect_person(image, detection_model, conf_threshold=0.5):
    """
    Runs person detection and returns bounding boxes.
    """
    results = detection_model(image)
    detections = []
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0 and conf >= conf_threshold:  # class 0 is person
            bbox = (int(x1), int(y1), int(x2), int(y2))
            detections.append((bbox, float(conf), "person"))
    return detections

def determine_corner(image_shape, bbox, corner_threshold=0.25):
    """
    Determines if person is in a corner.
    """
    height, width = image_shape[:2]
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    is_right = center_x > width * (1 - corner_threshold)
    is_left = center_x < width * corner_threshold
    is_top = center_y < height * corner_threshold
    is_bottom = center_y > height * (1 - corner_threshold)
    
    if is_top and is_right:
        return 'top_right', True
    elif is_top and is_left:
        return 'top_left', True
    elif is_bottom and is_right:
        return 'bottom_right', True
    elif is_bottom and is_left:
        return 'bottom_left', True
    
    return None, False

def draw_detection(image, bbox, corner_type=None):
    """
    Draws detection bbox and info on image.
    """
    img_draw = image.copy()
    cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    if corner_type:
        text = f"{corner_type}"
        cv2.putText(img_draw, text, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img_draw

def analyze_folder(folder_path, output_folder="debug_output", min_det_conf=0.5):
    """
    Analyzes all images in a folder for person detection and corners.
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    print("Loading detection model...")
    detection_model = load_nanodet_model()
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    
    # Process each image
    corner_detections = defaultdict(list)  # Store detections by corner type
    
    print(f"\nProcessing {len(image_files)} images...")
    for idx, img_file in enumerate(image_files):
        print(f"\nProcessing image {idx + 1}/{len(image_files)}: {img_file}")
        
        # Load image
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_file}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect person
        detections = detect_person(image_rgb, detection_model, min_det_conf)
        
        if not detections:
            print("No person detected")
            continue
        
        # Get first person detection
        bbox, conf, _ = detections[0]
        
        # Check if person is in corner
        corner_type, is_corner = determine_corner(image.shape, bbox)
        
        if is_corner:
            print(f"Found person in {corner_type} corner")
            print(f"bbox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
            
            # Draw and save detection
            img_draw = draw_detection(image_rgb, bbox, corner_type)
            
            # Save annotated image
            output_path = os.path.join(output_folder, f"detected_{img_file}")
            Image.fromarray(img_draw).save(output_path)
            
            # Store detection
            corner_detections[corner_type].append({
                'file': img_file,
                'bbox': bbox
            })
        else:
            print("Person not in corner")
    
    # Print summary
    print("\nDetection Summary:")
    for corner_type, detections in corner_detections.items():
        print(f"\n{corner_type} corner:")
        print(f"Found {len(detections)} images")
        if detections:
            print("Bounding boxes:")
            for det in detections:
                print(f"File: {det['file']}")
                print(f"bbox: {det['bbox']}")
    
    return corner_detections


def find_max_bbox(detections):
    """
    Calculate maximum bounding box that covers all detections.
    """
    if not detections:
        return None
        
    # Initialize with first bbox
    first_bbox = detections[0]['bbox']
    max_bbox = list(first_bbox)  # Convert to list for modification
    
    # Find max bounds across all detections
    for det in detections[1:]:
        bbox = det['bbox']
        max_bbox[0] = min(max_bbox[0], bbox[0])  # min x1  # max top
        max_bbox[1] = min(max_bbox[1], bbox[1])   # min y1  # min bottom - 30 pixels
        max_bbox[2] = max(max_bbox[2], bbox[2])  # max x2  # max left + 30 pixels
        max_bbox[3] = max(max_bbox[3], bbox[3])  # max y2  # min right
        
    print(f"Max bbox pre_adjustments: {max_bbox}")
    
    # add safety margin
    margin = 30
    
    max_bbox[0] = max_bbox[0] - margin
    max_bbox[3] = max_bbox[3] + margin
    
    print(f"Max bbox post_adjustments: {max_bbox}")
    
        
    return tuple(max_bbox)



def find_black_border(image, side='top'):
    """
    Find where black border ends for a given side.
    Returns the coordinate where color changes significantly from black.
    """
    height, width = image.shape[:2]
    threshold = 30  # threshold for black color
    
    if side == 'top':
        for y in range(height):
            line = image[y, :]
            if np.mean(line) > threshold:
                return y
    elif side == 'bottom':
        for y in range(height-1, -1, -1):
            line = image[y, :]
            if np.mean(line) > threshold:
                return y
    elif side == 'left':
        for x in range(width):
            line = image[:, x]
            if np.mean(line) > threshold:
                return x
    elif side == 'right':
        for x in range(width-1, -1, -1):
            line = image[:, x]
            if np.mean(line) > threshold:
                return x
    
    return 0 if side in ['top', 'left'] else (height-1 if side == 'bottom' else width-1)

def find_borders_from_samples(images, n_samples=3):
    """
    Find borders from random sample images.
    Returns maximum border coordinates with added safety margin.
    """
    if len(images) > n_samples:
        sample_images = random.sample(images, n_samples)
    else:
        sample_images = images
    
    borders = []
    for image in sample_images:
        # Convert to grayscale for border detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        top = find_black_border(gray, 'top')
        bottom = find_black_border(gray, 'bottom')
        left = find_black_border(gray, 'left')
        right = find_black_border(gray, 'right')
        
        borders.append((top, bottom, left, right))
        print(f"Found borders: top={top}, bottom={bottom}, left={left}, right={right}")
    
    # Get maximum borders and add safety margin
    max_borders = (
        max(b[0] for b in borders),      # max top
        min(b[1] for b in borders), # min bottom 
        max(b[2] for b in borders), # max left
        min(b[3] for b in borders)       # min right
    )
    
    return max_borders

def process_images_with_bbox(folder_path, corner_detections, output_folder, debug_print = False):
    """
    Process images with border removal and bbox handling.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate max bbox for each corner type
    max_bboxes = {}
    for corner_type, detections in corner_detections.items():
        max_bbox = find_max_bbox(detections)
        if max_bbox:
            print(f"Max bbox for {corner_type}: {max_bbox}")
            max_bboxes[corner_type] = max_bbox
    
    # Step 1: First remove bbox areas (make them black)
    print("\nStep 1: Removing bbox areas...")
    images_without_bbox = []
    filenames = []
    
    for corner_type, detections in corner_detections.items():
        if corner_type not in max_bboxes:
            continue
            
        max_bbox = max_bboxes[corner_type]
        
        for det in detections:
            img_path = os.path.join(folder_path, det['file'])
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {det['file']}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make bbox area black
            x1, y1, x2, y2 = max_bbox
            image_rgb[y1:y2, x1:x2] = [0, 0, 0]
            
            images_without_bbox.append(image_rgb)
            filenames.append(det['file'])
    
    # Step 2: Find borders from sample images
    print("\nStep 2: Finding borders from samples...")
    max_borders = find_borders_from_samples(images_without_bbox)
    print(f"Maximum borders: top={max_borders[0]}, bottom={max_borders[1]}, left={max_borders[2]}, right={max_borders[3]}")
    
    
    # Step 3: Process all images
    print("\nStep 3: Processing all images...")
    for i, (image, filename) in enumerate(zip(images_without_bbox, filenames)):
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale for border detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find actual borders for this image
        top = find_black_border(gray, 'top')
        bottom = find_black_border(gray, 'bottom')
        left = find_black_border(gray, 'left')
        right = find_black_border(gray, 'right')
        
        # Cut out the non-black part of the image
        cropped = image[top:bottom, left:right]
        
        # Create output image of the cropped size
        processed = cropped.copy()
        
        # Get the max_bbox coordinates in original image space
        max_bbox = max_bboxes[corner_type]
        
        # Transform max_bbox coordinates to cropped image space
        # Ensure we get the full overlapping region
        bbox_left = max(0, max_bbox[0] - left)  # Shift left by border amount
        bbox_top = max(0, max_bbox[1] - top)    # Shift top by border amount
        bbox_right = min(right - left, max_bbox[2] - left)  # Adjust right to cropped width
        bbox_bottom = min(bottom - top, max_bbox[3] - top)  # Adjust bottom to cropped height
        
        # Fill the overlapping region with white
        # Only fill if we have a valid region
        if (bbox_right > bbox_left and bbox_bottom > bbox_top and
            bbox_left < processed.shape[1] and bbox_top < processed.shape[0]):
            processed[bbox_top:bbox_bottom, bbox_left:bbox_right] = [255, 255, 255]
        
        # Save processed image
        output_path = os.path.join(output_folder, f"processed_{filename}")
        Image.fromarray(processed).save(output_path)
        
        # Save debug version showing the transformations
        if debug_print:
            debug = image.copy()
            # Draw original borders in green
            cv2.rectangle(debug, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw max_bbox in original coordinates in red
            cv2.rectangle(debug, 
                         (max_bbox[0], max_bbox[1]), 
                         (max_bbox[2], max_bbox[3]), 
                         (0, 0, 255), 2)
            debug_path = os.path.join(output_folder, f"debug_{filename}")
            Image.fromarray(debug).save(debug_path)
        
    print(f"\nProcessing complete:")
    print(f"Processed {len(images_without_bbox)} images")
    print(f"Output saved to: {output_folder}")
    
    
    
import imagehash
import multiprocessing

def compare_images(image1, image2, min_diff=10, corner_threshold=0.25):
    """
    Compares two frames using the non-corner region.
    Returns True if frames are significantly different.
    """
    # Get the region excluding the top-right corner
    height, width = image1.shape[:2]
    region1 = image1[int(height * corner_threshold):, :int(width * (1 - corner_threshold))]
    region2 = image2[int(height * corner_threshold):, :int(width * (1 - corner_threshold))]
    
    # Convert regions to PIL Images for hash comparison
    pil_region1 = Image.fromarray(region1)
    pil_region2 = Image.fromarray(region2)
    
    # Compare using perceptual hash
    hash1 = imagehash.average_hash(pil_region1)
    hash2 = imagehash.average_hash(pil_region2)
    
    return (hash1 - hash2) > min_diff


def process_video_segment(args):
    """
    Process a segment of video frames.
    Args:
        args: tuple of (video_path, start_frame, end_frame, temp_folder, frame_step, min_diff, fps)
    Returns:
        List of (frame_count, frame_rgb, filename, time_sec) for unique frames in this segment
    """
    video_path, start_frame, end_frame, temp_folder, frame_step, min_diff, fps = args
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    segment_frames = []
    previous_frame = None
    frame_count = start_frame
    
    while frame_count < end_frame and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time_sec = frame_count / fps
            
            # Compare with previous frame
            if previous_frame is None or compare_images(frame_rgb, previous_frame, min_diff):
                filename = f"frame_{frame_count:06d}.jpg"
                segment_frames.append((frame_count, frame_rgb, filename, time_sec))
                previous_frame = frame_rgb
        
        frame_count += 1
    
    cap.release()
    return segment_frames

def extract_unique_frames(video_path, frame_interval=1, min_diff=10, num_workers=None):
    """
    Extract frames from video using multiple processes, filter unique ones, and track timing.
    Args:
        video_path: Path to video file
        frame_interval: Time interval between frames in seconds
        min_diff: Minimum difference threshold for frame comparison
        num_workers: Number of worker processes (default: CPU count - 1)
    Returns:
        temp_folder: Path to folder with extracted frames
        frame_info: Dictionary with frame timing {filename: (start_time, end_time)}
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create temporary folder
    temp_folder = os.path.join(
        os.path.dirname(video_path),
        f"temp_frames_{os.path.splitext(os.path.basename(video_path))[0]}"
    )
    os.makedirs(temp_folder, exist_ok=True)
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(fps * frame_interval)
    cap.release()
    
    # Split video into segments
    frames_per_worker = total_frames // num_workers
    segments = []
    
    for i in range(num_workers):
        start_frame = i * frames_per_worker
        end_frame = start_frame + frames_per_worker if i < num_workers - 1 else total_frames
        segment = (video_path, start_frame, end_frame, temp_folder, frame_step, min_diff, fps)
        segments.append(segment)
    
    print(f"Processing video with {num_workers} workers...")
    
    # Process segments in parallel
    with multiprocessing.Pool(num_workers) as pool:
        segment_results = pool.map(process_video_segment, segments)
    
    # Combine results and eliminate duplicates between segments
    previous_frame = None
    frame_info = {}  # Dictionary to store frame timing information
    current_frame_start = None
    current_frame_name = None
    
    # Flatten and sort by frame count
    all_segment_frames = []
    for segment in segment_results:
        all_segment_frames.extend(segment)
    all_segment_frames.sort(key=lambda x: x[0])  # Sort by frame count
    
    print("Combining results and eliminating duplicates...")
    
    # Process combined frames
    for frame_count, frame_rgb, filename, time_sec in all_segment_frames:
        if previous_frame is None or compare_images(frame_rgb, previous_frame, min_diff):
            # If we had a previous frame, set its end time
            if current_frame_name:
                frame_info[current_frame_name] = (current_frame_start, time_sec)
            
            # Save the new frame
            output_path = os.path.join(temp_folder, filename)
            Image.fromarray(frame_rgb).save(output_path)
            
            # Update tracking info
            current_frame_start = time_sec
            current_frame_name = filename
            previous_frame = frame_rgb
            
            if len(frame_info) % 10 == 0:
                print(f"Processed {len(frame_info)} unique frames...")
    
    # Handle the last frame's end time
    if current_frame_name:
        frame_info[current_frame_name] = (current_frame_start, total_frames / fps)
    
    
    return temp_folder, frame_info


def process_video(video_path, output_folder, frame_files_file, frame_interval=1, min_diff=10, debug_print=False, num_workers = 2):
    """
    Process video file using existing image processing pipeline.
    Args:
        video_path: Path to video file
        output_folder: Where to save final processed frames
        frame_interval: Time between frames in seconds
        min_diff: Threshold for frame difference
        debug_print: Whether to save debug visualizations
    """
    # Extract unique frames
    frames_folder, frame_files = extract_unique_frames(video_path, frame_interval, min_diff, num_workers=num_workers)
    
    try:
        # Analyze frames for person detection
        corner_detections = analyze_folder(frames_folder, output_folder, min_det_conf=0.5)
        
        # Process detected frames
        if corner_detections:
            process_images_with_bbox(frames_folder, corner_detections, output_folder, debug_print)
        else:
            print("No corner detections found in video frames")
    
    finally:
        # Clean up temporary files
        if os.path.exists(frames_folder):
            import shutil
            shutil.rmtree(frames_folder)
            print(f"Cleaned up temporary folder: {frames_folder}")

    # save frame_files as txt file
    with open(os.path.join(output_folder, frame_files_file), "w") as f:
        for item in frame_files:
            f.write("%s\n" % item)
            
    # remove detected frames from output folder
    for file in os.listdir(output_folder):
        if file.startswith("detected_"):
            os.remove(os.path.join(output_folder, file))

# Example usage:
# process_video("your_video.mp4", "output_folder", frame_interval=1, min_diff=10)


def main(video, output_folder, frame_files_file):
    
    warnings.filterwarnings("ignore", category=FutureWarning)  
    print(f"Processing video: {video}")
    print(f"Output folder: {output_folder}")
    print(f"Frame files file: {frame_files_file}")     

    process_video(video, output_folder, frame_files_file, frame_interval=1, min_diff=2.5, num_workers = 1, debug_print = False) # min_diff = 10 default
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract slides from a video.")
    parser.add_argument("-v", "--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="Directory to store output frames.")
    parser.add_argument("-f", "--frame_files_file", type=str, required=True,
                        help="Filename for the text file listing frame files.")
    
    args = parser.parse_args()
    
    main(args.video, args.output_folder, args.frame_files_file)