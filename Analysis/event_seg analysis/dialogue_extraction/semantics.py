"""
=====================================================================
 Semantic Similarity Timecourse Generator (Sentence Transformers)
=====================================================================
This script takes transcript CSV files (with timestamps + text), 
computes semantic embeddings in sliding windows, and generates 
a timecourse of semantic similarity between consecutive windows.

Inputs:
  - TRANSCRIPT_DIR: folder containing transcript CSV files
        Each CSV must have at least two columns:
          • "start" = start time (seconds)
          • "text"  = spoken content
  - WINDOW_SIZE: size of the sliding window in seconds (default = 30)
  - STEP_SIZE: how much the window advances each step (default = 5)
  - MODEL_NAME: SentenceTransformers model for embeddings
        (default = "all-MiniLM-L6-v2")

Outputs:
  - For each transcript:
        1) A CSV with window center times, embeddings, and similarity values
        2) A PNG plot of semantic similarity vs. time
  - All outputs are saved in OUTPUT_DIR

Requirements:
  - Python packages: pandas, numpy, matplotlib, sentence-transformers, scikit-learn
  - Transcript CSVs with "start" and "text" columns

Author: Sam Ketcheson
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Config
# =========================

# Folder containing transcript CSVs (change to path of the output directory used in extract_diologue.py)
TRANSCRIPT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Transcripts"

# Folder to save output semantic results (CSV + plots) (change to your desired output folder)
OUTPUT_DIR     = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Semantics"

# Windowing parameters
WINDOW_SIZE    = 30    # Size of each time window in seconds
STEP_SIZE      = 5     # Step size (advance per window) in seconds

# SentenceTransformers embedding model
#   Options: "all-MiniLM-L6-v2" (fast, small), "all-mpnet-base-v2" (slower, higher quality), etc.
MODEL_NAME     = "all-MiniLM-L6-v2"

# =========================

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the embedding model once (can take a few seconds)
print(f"Loading model {MODEL_NAME} …")
model = SentenceTransformer(MODEL_NAME)


def create_windows(df, window_size, step_size):
    """
    Slide a time window across the transcript and combine text inside each window.

    Parameters:
        df (DataFrame): Transcript with columns 'start' and 'text'
        window_size (int): Duration of each window (seconds)
        step_size (int): Amount of shift between windows (seconds)

    Returns:
        DataFrame: Windows with columns 'time' (center of window) and 'text'
    """
    max_time = df['start'].max()
    windows = []
    t = 0
    while t < max_time:
        t_end = t + window_size
        # Select transcript rows within this time window
        chunk = df[(df['start'] >= t) & (df['start'] < t_end)]
        if not chunk.empty:
            text = " ".join(chunk['text'].tolist())
            # Save the midpoint of the window + concatenated text
            windows.append({"time": t + window_size/2, "text": text})
        t += step_size
    return pd.DataFrame(windows)


def process_file(file):
    """
    Process a single transcript file:
      1) Create sliding windows
      2) Generate embeddings
      3) Compute similarity between consecutive windows
      4) Save CSV + plot
    """
    print(f"Processing {file} …")
    if file == "practice_clip_aligned.csv":
        return
    df = pd.read_csv(os.path.join(TRANSCRIPT_DIR, file))
    
    # Check that required columns exist
    if not {"start", "text"}.issubset(df.columns):
        raise ValueError(f"{file} must contain 'start' and 'text' columns")
    
    # --- Step 1: Create sliding windows ---
    win_df = create_windows(df, WINDOW_SIZE, STEP_SIZE)

    # --- Step 2: Compute embeddings for each window ---
    embeddings = model.encode(win_df["text"].tolist(), show_progress_bar=True)
    win_df["embedding"] = list(embeddings)

    # --- Step 3: Compute cosine similarity with previous window ---
    sims = [np.nan]  # first window has no previous
    for i in range(1, len(embeddings)):
        sims.append(cosine_similarity([embeddings[i-1]], [embeddings[i]])[0,0])
    win_df["similarity_prev"] = sims

    # --- Step 4: Save results to CSV ---
    out_csv = os.path.join(OUTPUT_DIR, file.replace(".csv", "_semantic.csv"))
    win_df.to_csv(out_csv, index=False)
    print(f"Saved semantic timecourse: {out_csv}")

    # --- Step 5: Plot similarity over time ---
    plt.figure(figsize=(12,4))
    plt.plot(win_df["time"], win_df["similarity_prev"], marker="o", alpha=0.7)
    plt.title(f"Semantic similarity timecourse – {file}")
    plt.xlabel("Time (s)")
    plt.ylabel("Cosine similarity (vs. previous window)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_fig = out_csv.replace(".csv", ".png")
    plt.savefig(out_fig)
    plt.close()
    print(f"Saved plot: {out_fig}")


# =========================
# Run the pipeline
# =========================

# Loop through all transcript files in the input directory
for file in os.listdir(TRANSCRIPT_DIR):
    if file.endswith(".csv"):
        process_file(file)

print("Done – semantic timecourses + plots created!")
