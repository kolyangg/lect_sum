import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline, WhisperTimeStampLogitsProcessor
import soundfile as sf
import argparse
import time
import numpy as np
from datasets import Dataset

def transcribe_audio(
    input_file, 
    output_file, 
    batch_size=16, 
    cpu_cores=1, 
    language="russian", 
    return_timestamps=False,
    chunk_length=30,       # chunk length in seconds
    overlap_seconds=5,     # overlap in seconds between consecutive chunks
    max_time=None          # new argument: maximum processing time (HH:MM:SS)
):
    print("Transcribing audio with the following settings:")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Batch size: {batch_size}")
    print(f"  CPU cores: {cpu_cores}")
    print(f"  Language: {language}")
    print(f"  Return timestamps: {return_timestamps}")
    print(f"  Chunk length: {chunk_length} sec")
    print(f"  Overlap seconds: {overlap_seconds}")
    if max_time:
        print(f"  Max time: {max_time}")

    torch_dtype = torch.bfloat16

    # --- Load audio and split into overlapping chunks ---
    wav, sr = sf.read(input_file)
    print("Sample rate:", sr)
    duration = len(wav) / sr
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Limit the audio processing duration if max_time is provided
    if max_time is not None:
        try:
            h, m, s = map(int, max_time.split(":"))
            max_time_seconds = h * 3600 + m * 60 + s
            duration = min(duration, max_time_seconds)
            print(f"Processing audio up to {max_time} ({duration:.2f} seconds)")
        except Exception as e:
            print(f"Invalid max_time format: {max_time}. Should be HH:MM:SS. Error: {e}")

    # Compute step size for sliding window
    step = chunk_length - overlap_seconds
    chunks = []
    start_times = []
    current_start = 0.0
    while current_start < duration:
        current_end = current_start + chunk_length
        start_idx = int(current_start * sr)
        end_idx = int(min(current_end * sr, len(wav)))
        chunks.append(wav[start_idx:end_idx])
        start_times.append(current_start)
        current_start += step

    data = {"audio": chunks, "start_time": start_times}
    dataset = Dataset.from_dict(data)

    # --- Helper function to process a batch of audio with silence filtering ---
    def transcribe_batch(batch, asr_pipeline_local, language):
        silence_threshold = 0.01  # Adjust this threshold if needed
        audio_list = batch["audio"]
        # Prepare list indices for non-silent chunks
        non_silent_indices = []
        non_silent_audio = []
        for i, a in enumerate(audio_list):
            a = np.array(a, dtype=np.float32).squeeze()
            if np.max(np.abs(a)) < silence_threshold:
                pass
            else:
                non_silent_indices.append(i)
                non_silent_audio.append(a)
        
        # For non-silent chunks, pad to the same length
        if non_silent_audio:
            max_len = max(len(a) for a in non_silent_audio)
            padded_list = []
            for a in non_silent_audio:
                a = np.array(a, dtype=np.float32).squeeze()
                if a.ndim != 1:
                    raise ValueError("Expected a single channel (1D) audio array.")
                if len(a) < max_len:
                    a = np.pad(a, (0, max_len - len(a)), mode="constant")
                padded_list.append(a)
            results = asr_pipeline_local(
                padded_list,
                generate_kwargs={
                    "language": language, 
                    "max_new_tokens": 256,
                    "logits_processor": [WhisperTimeStampLogitsProcessor(asr_pipeline_local.model.generation_config)]
                }
            )
        else:
            results = []
        
        # Assemble transcriptions in original order; silent chunks yield empty strings
        transcriptions = ["" for _ in range(len(audio_list))]
        j = 0
        for i, a in enumerate(audio_list):
            if np.max(np.abs(a)) < silence_threshold:
                transcriptions[i] = ""
            else:
                r = results[j]
                chunk_offset = batch["start_time"][i]
                if return_timestamps and "chunks" in r and r["chunks"]:
                    chunk_strs = []
                    for ch in r["chunks"]:
                        ts = ch.get("timestamp", [None, None])
                        start_time_chunk = ts[0] if ts[0] is not None else 0.0
                        end_time_chunk = ts[1]
                        abs_start = chunk_offset + start_time_chunk
                        if end_time_chunk is not None:
                            abs_end = chunk_offset + end_time_chunk
                            end_time_str = f"{abs_end:.2f}"
                        else:
                            end_time_str = "NA"
                        chunk_strs.append(f"[{abs_start:.2f}-{end_time_str}] {ch['text'].strip()}")
                    transcriptions[i] = " ".join(chunk_strs)
                else:
                    transcriptions[i] = r["text"].strip()
                j += 1
        batch["transcription"] = transcriptions
        return batch

    # --- Worker function for multiprocessing on CPU ---
    def transcribe_worker(batch, language, batch_size):
        local_device = torch.device("cpu")
        model = WhisperForConditionalGeneration.from_pretrained(
            "antony66/whisper-large-v3-russian",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")
        asr_pipeline_local = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=chunk_length,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            torch_dtype=torch_dtype,
            device=local_device,
        )
        return transcribe_batch(batch, asr_pipeline_local, language)

    # --- Decide processing mode (GPU or CPU multiprocess) ---
    if torch.cuda.is_available() or cpu_cores == 1:
        device_main = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WhisperForConditionalGeneration.from_pretrained(
            "antony66/whisper-large-v3-russian",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")
        asr_pipeline_global = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=chunk_length,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            torch_dtype=torch_dtype,
            device=device_main,
        )

        def transcribe_batch_main(batch):
            return transcribe_batch(batch, asr_pipeline_global, language)

        dataset = dataset.map(transcribe_batch_main, batched=True, batch_size=batch_size)
    else:
        dataset = dataset.map(
            transcribe_worker,
            batched=True,
            batch_size=batch_size,
            num_proc=cpu_cores,
            fn_kwargs={"language": language, "batch_size": batch_size}
        )

    # --- Combine transcriptions and write to output file ---
    lines = []
    for text in dataset["transcription"]:
        if text.strip():
            lines.append(text)
    full_transcription = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_transcription)

    print("Transcription saved to", output_file)
    return full_transcription

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using specified parameters."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="testing_video_manual.wav",
        help="Path to the input audio file (default: testing_video_manual.wav)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="transcription_timest.txt",
        help="Path to the output transcription file (default: transcription_timest.txt)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)"
    )
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=16,
        help="Number of CPU cores to use (default: 16)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="russian",
        help="Language of the audio (default: russian)"
    )
    parser.add_argument(
        "--return_timestamps",
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="Whether to return timestamps (True/False) (default: True)"
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=30,
        help="Chunk length in seconds (default: 30)"
    )
    parser.add_argument(
        "--overlap_seconds",
        type=int,
        default=5,
        help="Overlap in seconds between chunks (default: 5)"
    )
    parser.add_argument(
        "--max_time",
        type=str,
        required=False,
        help="Maximum time to process the audio in HH:MM:SS format (optional)"
    )
    
    args = parser.parse_args()
    
    transcribe_audio(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        cpu_cores=args.cpu_cores,
        language=args.language,
        return_timestamps=args.return_timestamps,
        chunk_length=args.chunk_length,
        overlap_seconds=args.overlap_seconds,
        max_time=args.max_time
    )

if __name__ == "__main__":
    main()
