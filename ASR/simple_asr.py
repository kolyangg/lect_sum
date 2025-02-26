# HF code to use Whisper

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import soundfile as sf
import argparse

import time
import numpy as np
import torch
import soundfile as sf
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    pipeline,
    WhisperTimeStampLogitsProcessor
)


# torch_dtype = torch.bfloat16 # set your preferred type here 

# device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'
#     setattr(torch.distributed, "is_initialized", lambda : False) # monkey patching
# device = torch.device(device)

# whisper = WhisperForConditionalGeneration.from_pretrained(
#     "antony66/whisper-large-v3-russian", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
#     # add attn_implementation="flash_attention_2" if your GPU supports it
# )

# processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")

# asr_pipeline = pipeline(
#     "automatic-speech-recognition",
#     model=whisper,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=256,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )


# # Read the WAV file into a NumPy array (and get its sample rate)

# wav, sr = sf.read('../LLM_valid_ru.wav')

# # Optionally, check that your sample rate is what you expect (e.g. 16000 Hz)
# print("Sample rate:", sr)

# # Now pass the numpy array to the ASR pipeline
# asr = asr_pipeline(wav, generate_kwargs={"language": "russian", "max_new_tokens": 256}, return_timestamps=False)
# print(asr['text'])


def transcribe_audio(
    input_file, 
    output_file, 
    batch_size=16, 
    cpu_cores=1, 
    language="russian", 
    return_timestamps=False,
    chunk_length=30,       # chunk length in seconds
    overlap_seconds=5      # overlap in seconds between consecutive chunks
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

    torch_dtype = torch.bfloat16

    # --- Load audio and split into overlapping chunks ---
    wav, sr = sf.read(input_file)
    print("Sample rate:", sr)
    duration = len(wav) / sr
    print(f"Audio duration: {duration:.2f} seconds")
    
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

    # --- Helper function to process a batch of audio ---
    def transcribe_batch(batch, asr_pipeline_local, language):
        audio_list = batch["audio"]
        # Ensure each sample is a 1D float32 numpy array; pad them to the same length in the batch.
        max_len = max(len(a) for a in audio_list)
        padded_list = []
        for a in audio_list:
            a = np.array(a, dtype=np.float32).squeeze()
            if a.ndim != 1:
                raise ValueError("Expected a single channel (1D) audio array.")
            if len(a) < max_len:
                a = np.pad(a, (0, max_len - len(a)), mode="constant")
            padded_list.append(a)
        # Include the WhisperTimeStampLogitsProcessor in generate_kwargs.
        results = asr_pipeline_local(
            padded_list,
            generate_kwargs={
                "language": language, 
                "max_new_tokens": 256,
                "logits_processor": [WhisperTimeStampLogitsProcessor(asr_pipeline_local.model.generation_config)]
            }
        )
        if return_timestamps:
            transcriptions = []
            for r in results:
                if "chunks" in r and r["chunks"]:
                    chunk_strs = []
                    for ch in r["chunks"]:
                        # Get the timestamp list; if the ending timestamp is missing, output "NA"
                        ts = ch.get("timestamp", [None, None])
                        start_time_chunk = ts[0] if ts[0] is not None else 0.0
                        end_time_chunk = ts[1]
                        if end_time_chunk is None:
                            end_time_str = "NA"
                        else:
                            end_time_str = f"{end_time_chunk:.2f}"
                        chunk_strs.append(f"[{start_time_chunk:.2f}-{end_time_str}] {ch['text'].strip()}")
                    transcriptions.append(" ".join(chunk_strs))
                else:
                    transcriptions.append(r["text"].strip())
            batch["transcription"] = transcriptions
        else:
            batch["transcription"] = [r["text"].strip() for r in results]
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
    # Write each chunk's transcription on a new line, along with its start time.
    lines = []
    for start, text in zip(dataset["start_time"], dataset["transcription"]):
        lines.append(f"[Chunk start {start:.2f}s]: {text}")
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
        help="Path to the input audio file (default: ../LLM_valid_ru.wav)"
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
    
    args = parser.parse_args()
    
    transcribe_audio(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        cpu_cores=args.cpu_cores,
        language=args.language,
        return_timestamps=args.return_timestamps,
        chunk_length=args.chunk_length,
        overlap_seconds=args.overlap_seconds
    )

if __name__ == "__main__":
    main()