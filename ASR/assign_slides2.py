import os
import re
import json
import glob
import argparse
import math
import cv2

def parse_transcript(transcript_file):
    """
    Reads the transcript file and extracts segments.
    Each segment is expected to be of the form:
      [start-end] text
    Returns a list of tuples: (start_time, text) sorted by start_time.
    """
    pattern = re.compile(r'\[(\d+\.\d+)-([0-9\.NA]+)\]\s*([^[]+)')
    segments = []
    with open(transcript_file, "r", encoding="utf-8") as f:
        content = f.read()
        matches = pattern.findall(content)
        for match in matches:
            start_str, end_str, text = match
            text = text.strip()
            if not text:
                continue
            try:
                start_time = float(start_str)
            except ValueError:
                continue
            segments.append((start_time, text))
    # sort segments by their start times
    segments.sort(key=lambda x: x[0])
    return segments

def combine_into_sentences(segments):
    """
    Combine consecutive segments into full sentences.
    We assume that if a segment’s text does not end with a sentence terminator (. ! or ?),
    then the next segment is a continuation of the same sentence.
    The start time of the sentence is taken as the start time of the first segment.
    Returns a list of tuples: (sentence_start_time, sentence_text)
    """
    sentences = []
    current_sentence = ""
    current_start = None
    terminators = {'.', '!', '?'}
    for start, text in segments:
        if current_sentence == "":
            current_sentence = text
            current_start = start
        else:
            # add a space between segments
            current_sentence += " " + text
        # Check if the current sentence ends with a terminator
        if current_sentence.strip() and current_sentence.strip()[-1] in terminators:
            sentences.append((current_start, current_sentence.strip()))
            current_sentence = ""
            current_start = None
    if current_sentence:
        # add the remaining text as a sentence
        sentences.append((current_start, current_sentence.strip()))
    return sentences

def get_frame_times(images_folder, fps):
    """
    List image files (assumed to match pattern processed_frame_*.jpg),
    extract the numeric frame number, convert it to time (frame_number / fps),
    and return a list of tuples: (frame_number, frame_time)
    sorted by frame_time.
    """
    files = glob.glob(os.path.join(images_folder, "processed_frame_*.jpg"))
    frame_pattern = re.compile(r'processed_frame_(\d+)\.jpg')
    frames = []
    for file in files:
        basename = os.path.basename(file)
        m = frame_pattern.search(basename)
        if m:
            frame_num = int(m.group(1))
            frame_time = frame_num / fps
            frames.append((frame_num, frame_time))
    frames.sort(key=lambda x: x[1])
    return frames

def assign_transcript_to_frames(transcript_file, images_folder, fps=25, output_json="output.json"):
    """
    Reads the transcript file and image folder, assigns transcript text to each image frame
    based on the time interval from the frame's time until the next.
    
    The transcript text is first parsed into segments (with start times),
    then combined into sentences (ensuring sentences are complete).
    
    For each frame (using its time from frame_number/fps), the function
    collects all sentences that start before the next frame's time.
    If a sentence spans the boundary, it is included fully in the current frame's text.
    
    The resulting dictionary is saved as a JSON file.
    
    Args:
      transcript_file (str): Path to the transcript text file.
      images_folder (str): Path to the folder containing image files.
      fps (float): Frames per second (default: 25). Used to convert frame numbers to seconds.
      output_json (str): Path for the output JSON file.
    
    Returns:
      dict: Mapping from frame number (int) to transcript text (str).
    """
    segments = parse_transcript(transcript_file)
    sentences = combine_into_sentences(segments)
    
    frames = get_frame_times(images_folder, fps)
    if not frames:
        print("No frames found in folder:", images_folder)
        return {}
    
    mapping = {}
    num_frames = len(frames)
    sentence_index = 0
    num_sentences = len(sentences)
    
    for i, (frame_num, frame_time) in enumerate(frames):
        if i < num_frames - 1:
            next_frame_time = frames[i+1][1]
        else:
            next_frame_time = math.inf
        
        slide_sentences = []
        while sentence_index < num_sentences:
            sent_start, sent_text = sentences[sentence_index]
            if sent_start < frame_time:
                sentence_index += 1
                continue
            if sent_start < next_frame_time:
                slide_sentences.append(sent_text)
                sentence_index += 1
            else:
                break
        
        slide_text = " ".join(slide_sentences).strip()
        mapping[frame_num] = slide_text

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(mapping, fout, ensure_ascii=False, indent=2)
    print(f"Saved assigned transcript to {output_json}")
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign transcript text to image frames based on time intervals."
    )
    parser.add_argument("--transcript_file", type=str, required=True,
                        help="Path to the transcript text file.")
    parser.add_argument("--images_folder", type=str, required=True,
                        help="Path to the folder containing image files.")
    parser.add_argument("--fps", type=float, default=25,
                        help="Frames per second used to convert frame numbers to seconds (default: 25).")
    parser.add_argument("--output_json", type=str, default="output.json",
                        help="Path to save the resulting JSON file (default: output.json).")
    parser.add_argument("--video_file", type=str, required=False,
                        help="Path to a video file to extract FPS from. If provided, the FPS will be read from this file.")

    args = parser.parse_args()
    
    # If a video file is provided, extract fps from it; otherwise, use the provided --fps parameter.
    if args.video_file:
        cap = cv2.VideoCapture(args.video_file)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Using FPS from video file '{args.video_file}': {video_fps}")
        fps = video_fps
    else:
        fps = args.fps
    
    assign_transcript_to_frames(args.transcript_file, args.images_folder, fps, args.output_json)
