import streamlit as st
import subprocess
import os
import glob
import uuid
import base64
import re


st.set_page_config(page_title="RL Lecture Processing", layout="wide")

st.title("RL Lecture Processing Frontend")
st.write("This interface processes a YouTube video in multiple steps:")

# User inputs
youtube_url = st.text_input("Enter YouTube Video URL:", value="https://www.youtube.com/watch?v=MKd59yzfJKw")

# Common parameters for steps 2 and 4 (frame extraction and transcription)
ending_time = st.text_input("Enter ending time (for frame extraction & transcription)", value="1:31:00")

# Parameters for step 9
start_cut = st.number_input("Enter start_cut (for output creation)", value=1, step=1)
end_cut = st.number_input("Enter end_cut (for output creation)", value=2, step=1)

# A placeholder to display command logs
log_area = st.empty()

# Ensure session state variable for downloaded file
if "downloaded_file" not in st.session_state:
    st.session_state.downloaded_file = None


def run_command(cmd):
    """Run a shell command and update a single output field showing all printouts."""
    st.write(f"**Running:** `{cmd}`")
    try:
        # Use line buffering for immediate output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        output = ""
        # Create a single placeholder for the log output.
        log_placeholder = st.empty()
        # Read and update the output continuously
        for line in process.stdout:
            output += line
            log_placeholder.text_area("Execution Log", output, height=300)
        process.stdout.close()
        process.wait()
        return output
    except Exception as e:
        st.error(f"Error running command: {e}")
        return ""
    

def embed_local_images(md_content):
    # This function finds image tags in the Markdown and replaces the src
    # with a base64-encoded version if the file exists.
    def replace_img(match):
        src = match.group(1)
        if os.path.exists(src):
            with open(src, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
            # Determine mime type based on extension
            ext = os.path.splitext(src)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
            elif ext == ".png":
                mime = "image/png"
            else:
                mime = "application/octet-stream"
            # Construct the data URL
            new_src = f"data:{mime};base64,{encoded}"
            return f'<img src="{new_src}"'
        else:
            # If the file doesn't exist, return the original match.
            return match.group(0)

    # Look for <img src="..."> patterns and process them
    pattern = r'<img\s+src="([^"]+)"'
    return re.sub(pattern, replace_img, md_content)


# Step 1: Download video and save as temp with proper extension
if st.button("Step 1: Download Video"):
    if not youtube_url:
        st.error("Please enter a YouTube video URL.")
    else:
        cmd_download = f'yt-dlp -o "temp.%(ext)s" {youtube_url} -k'
        st.info("Downloading video and saving as temp with the proper extension.")
        run_command(cmd_download)
        # After download, find the downloaded file (e.g., temp.mkv, temp.webm, etc.)
        downloaded_files = glob.glob("temp.*")
        if downloaded_files:
            st.session_state.downloaded_file = downloaded_files[0]
            st.success(f"Step 1 complete. Downloaded file: {st.session_state.downloaded_file}")
        else:
            st.error("Downloaded file not found.")

# Step 2: Extract frames from the video
if st.button("Step 2: Extract Frames"):
    if not st.session_state.downloaded_file:
        st.error("Please complete Step 1: Download Video first.")
    else:
        cmd_extract_frames = (
            f'python3 slides/extract_slides2.py -v {st.session_state.downloaded_file} -o temp/temp_slides '
            f'-f frame_files_RL2.txt -t {ending_time}'
        )
        st.info("Extracting frames...")
        run_command(cmd_extract_frames)
        st.success("Step 2 complete.")

# Step 3: Extract audio track
if st.button("Step 3: Extract Audio"):
    if not st.session_state.downloaded_file:
        st.error("Please complete Step 1: Download Video first.")
    else:
        cmd_extract_audio = (
            f'ffmpeg -y -i {st.session_state.downloaded_file} -vn -acodec pcm_s16le -ar 16000 -ac 1 temp.wav'
        )
        st.info("Extracting audio track...")
        run_command(cmd_extract_audio)
        st.success("Step 3 complete.")

# Step 4: Transcribe audio
if st.button("Step 4: Transcribe Audio"):
    cmd_transcribe = (
        f'python3 -u ASR/simple_asr3.py --input_file temp.wav --return_timestamps 1 '
        f'--output_file temp/transcription_RL.txt --batch_size 1 --cpu_cores 1 --max_time {ending_time}'
    )
    st.info("Transcribing audio...")
    run_command(cmd_transcribe)
    st.success("Step 4 complete.")

# Step 5: Align transcript with frames
if st.button("Step 5: Assign Transcript to Frames"):
    if not st.session_state.downloaded_file:
        st.error("Please complete Step 1: Download Video first.")
    else:
        cmd_assign = (
            f'python3 ASR/assign_slides2.py --transcript_file temp/transcription_RL.txt '
            f'--images_folder temp/temp_slides --output_json temp/img_text_RL.json --video_file {st.session_state.downloaded_file}'
        )
        st.info("Assigning transcript to frames...")
        run_command(cmd_assign)
        st.success("Step 5 complete.")

# Step 6: Clean up transcript
if st.button("Step 6: Clean Up Transcript"):
    cmd_cleanup_transcript = (
        'python3 ASR/asr_cleanup.py --input_json temp/img_text_RL.json --output_json temp/img_text_RL_adj.json'
    )
    st.info("Cleaning up transcript...")
    run_command(cmd_cleanup_transcript)
    st.success("Step 6 complete.")

# Step 7: Classify slides (clean up frames)
if st.button("Step 7: Classify Slides"):
    cmd_classify = (
        'python3 vlm/classify_slides.py --input_folder temp/temp_slides'
        '--output_classification_file temp/RL_images_check.txt --output_captions_file temp/RL_images_captions.txt'
    )
    st.info("Classifying slides...")
    run_command(cmd_classify)
    st.success("Step 7 complete.")

# Step 8: Merge duplicate frames
if st.button("Step 8: Merge Duplicate Slides"):
    cmd_merge_dupl = (
        'python3 vlm/merge_dupl2.py --images_folder temp/temp_slides --output_file temp/unique_img_RL.txt'
    )
    st.info("Merging duplicate slides...")
    run_command(cmd_merge_dupl)
    st.success("Step 8 complete.")

# Step 9: Clean up images (remove black pixels on top)
if st.button("Step 9: Final image clean-up"):
    cmd_cleanup = (
        'python3 vlm/remove_black.py --input_folder temp/temp_slides --unique_img_file temp/unique_img_RL.txt --output_folder temp/temp_slides2'
    )
    st.info("Doing final clean-up...")
    run_command(cmd_cleanup)
    st.success("Step 9 complete.")

# Step 10: Create a final raw json file with notes
if st.button("Step 10: Group notes for same slides"):
    cmd_group_json = (
        f'python3 output/raw_json_final.py --json_file temp/img_text_RL_adj.json --images_folder temp/temp_slides2 '
        f'--unique_img_file temp/unique_img_RL.txt --output_transcript_json temp/RL_json1.json '
        f'--group_duplicates --start_cut {start_cut} --end_cut {end_cut}'
    )
    st.info("Grouping notes...")
    run_command(cmd_group_json)
    st.success("Step 10 complete.")



# Step 11: Summarize the lecture
if st.button("Step 11: Summarize lecture notes"):
    cmd_cleanup = (
        'python3 rag/json_summ.py --api --final_json temp/RL_json1.json --system_prompt rag/summ_prompt.txt --json_structured temp/RL_json1_str2.json'
    )
    st.info("Summarizing notes...")
    run_command(cmd_cleanup)
    st.success("Step 11 complete.")


# Step 12: Fix LATEX in formulas
if st.button("Step 12: Fix LATEX in formulas"):
    cmd_latex = (
        'python3 output/fix_latex.py --input_json temp/RL_json1_str2.json --output_json temp/RL_json1_str3.json'
    )
    st.info("Cleaning LATEX...")
    run_command(cmd_latex)
    st.success("Step 12 complete.")


# Step 13: Create a list of terms for RAG
if st.button("Step 13: Create a list of terms for RAG"):
    cmd_notes = (
        'python3 notes_txt.py --json_structured temp/RL_json1_str3.json --output temp/notes.txt'
    )
    st.info("Saving notes as text...")
    run_command(cmd_notes)

    cmd_terms = (
        'python3 rag/extract_terms.py --notes temp/notes.txt --output temp/terms.txt --retry 3 --api'
    )
    st.info("Extracting list of terms...")
    run_command(cmd_terms)


    st.success("Step 13 complete.")



# Step 14: Create Markdown output and display it
if st.button("Step 14: Create Markdown Output"):
    cmd_create_md = (
        f'python3 output/create_md_only.py --json_file temp/RL_json1_str3.json --images_folder temp/temp_slides2 --output_md temp/RL_md_str.md '
        f'--unique_img_file temp/unique_img_RL.txt'
    )

    st.info("Creating Markdown output...")
    run_command(cmd_create_md)
    st.success("Step 14 complete.")

    if os.path.exists("temp/RL_md_str.md"):
        with open("temp/RL_md_str.md", "r", encoding="utf-8") as f:
            md_content = f.read()
        # Process the markdown to embed local images
        md_content = embed_local_images(md_content)
        st.markdown("### Generated Markdown Output")
        st.markdown(md_content, unsafe_allow_html=True)
