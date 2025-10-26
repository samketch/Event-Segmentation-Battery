"""
===============================================================
 Video Transcription and Alignment Script (WhisperX + PyTorch)
===============================================================
This script automatically processes all .mp4 videos in a folder,
transcribes the audio using a WhisperX speech-to-text model, 
aligns the words precisely to the audio track, and saves the 
results as CSV files with start/end timestamps and text.

Inputs:
  - VIDEO_DIR: folder containing input .mp4 video files
  - MODEL_SIZE: size of WhisperX model to use (tiny, base, small, medium, large)

Outputs:
  - CSV transcripts saved in OUTPUT_DIR
      Columns: start (s), end (s), text
  - Skips files already processed (no overwriting)

Requirements:
  - Python packages: whisperx, torch, pandas
  - GPU with CUDA recommended for faster processing (falls back to CPU)

Author: Sam Ketcheson
"""

import os
import whisperx
import torch
import pandas as pd

# =========================
# Configuration Section
# =========================

# Folder containing all the video files to process (change this to your video path)
VIDEO_DIR  = r"C:\Users\Smallwood Lab\friends-event-segmentation\Tasks\taskScripts\resources\Movie_Task\videos"

# Folder where output transcript CSVs will be saved (change this to your output directory)
OUTPUT_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Transcripts"

# Whisper model size to use. Options (in increasing accuracy + size): 
#   "tiny", "base", "small", "medium", "large"
# Larger models are more accurate but slower and require more VRAM/CPU.
MODEL_SIZE = "large"
# =========================

# Ensure the output directory exists; if not, create it
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detect whether to use GPU (CUDA) or fallback to CPU
#   - CUDA (GPU) will be much faster if available
#   - CPU will work but may be significantly slower
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the chosen WhisperX model into memory
# compute_type="int8" is a quantization setting to save memory and run faster
print(f"Loading WhisperX model '{MODEL_SIZE}' on {device} …")
model = whisperx.load_model(MODEL_SIZE, device, compute_type="int8")

# =========================
# Main Processing Loop
# =========================

# Loop through every file in the video folder
for file in os.listdir(VIDEO_DIR):
    
    # Only process files ending in .mp4 (skip other file types)
    if not file.lower().endswith(".mp4"):
        continue

    # Build full paths for the input video and output transcript
    video_path = os.path.join(VIDEO_DIR, file)
    base = os.path.splitext(file)[0]   # strip off .mp4 to get base name
    out_path = os.path.join(OUTPUT_DIR, f"{base}_aligned.csv")

    # Skip if this video has already been processed (prevents duplicate work)
    if os.path.exists(out_path):
        print(f"Skipping {file} (already processed).")
        continue

    print(f"\n🎬 Processing {file} …")

    # --- Step 1: Transcribe audio from the video ---
    # The model generates text segments with approximate timestamps
    result = model.transcribe(video_path)

    # --- Step 2: Load alignment model ---
    # Improves timing accuracy by aligning words precisely with the audio waveform
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    # --- Step 3: Align words to audio ---
    # Refines the initial transcription by syncing word boundaries with audio
    aligned = whisperx.align(result["segments"], align_model, metadata, video_path, device)

    # --- Step 4: Save aligned transcript to CSV ---
    # Collect start time, end time, and text for each segment
    rows = []
    for seg in aligned["segments"]:
        rows.append({
            "start": seg["start"],   # start time in seconds
            "end": seg["end"],       # end time in seconds
            "text": seg["text"]      # spoken content
        })

    # Convert list of dicts → DataFrame → CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"✅ Saved transcript: {out_path}")

# =========================
# Wrap-up
# =========================

print("\n🎉 All videos processed. Transcripts with timestamps are in:")
print(OUTPUT_DIR)
