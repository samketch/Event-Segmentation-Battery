"""
=====================================================================
 Semantic Representational Similarity Analysis (RSA)
=====================================================================
This script computes an all-to-all semantic similarity matrix for each
video based on sliding-window text embeddings of the transcript.

Fixes included:
 - Forces 'start' and 'end' columns to numeric (prevents truncation bug)
 - Adds diagnostic prints showing max time and number of windows
 - Ensures RSA windows span full video duration (e.g., ~1300 s)

Outputs (per video):
 - *_RSA_matrix.npy  : full cosine similarity matrix (time × time)
 - *_RSA_matrix.png  : visual heatmap of the matrix
 - *_RSA_windows.csv : time and text per window

Author: Sam Ketcheson
=====================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Config
# =========================
TRANSCRIPT_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Transcripts"
OUTPUT_DIR     = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\RSA"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sliding window parameters
WINDOW_SIZE = 45   # seconds per window
STEP_SIZE   = 5    # step size (seconds)
MODEL_NAME  = "all-MiniLM-L6-v2"  # small + fast model

print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# =========================
# Helper function
# =========================
def create_windows(df, window_size, step_size):
    """Slide a time window across transcript and combine text."""
    max_time = df["start"].max()
    windows = []
    t = 0
    while t < max_time:
        t_end = t + window_size
        chunk = df[(df["start"] >= t) & (df["start"] < t_end)]
        if not chunk.empty:
            text = " ".join(chunk["text"].tolist())
            windows.append({"time": t + window_size / 2, "text": text})
        t += step_size
    return pd.DataFrame(windows)

# =========================
# Main loop
# =========================
for file in os.listdir(TRANSCRIPT_DIR):
    if not file.endswith("_aligned.csv"):
        continue

    base = file.replace("_aligned.csv", "")
    print(f"\n Processing {base} …")

    # Load transcript
    df = pd.read_csv(os.path.join(TRANSCRIPT_DIR, file))

    # --- Force numeric conversion (important) ---
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")

    # Drop any rows with NaN start times
    df = df.dropna(subset=["start", "text"])

    # Diagnostic: check duration
    max_time = df["start"].max()
    print(f"  ↳ Transcript duration: {max_time:.1f} seconds")
    print(f"  ↳ Total transcript rows: {len(df)}")

    if max_time < 60:
        print(f"   Duration seems too short ({max_time:.1f}s). Check transcript file!")
        continue

    # Step 1: create sliding windows
    win_df = create_windows(df, WINDOW_SIZE, STEP_SIZE)
    print(f"  ↳ Created {len(win_df)} windows (size={WINDOW_SIZE}s, step={STEP_SIZE}s)")

    # Step 2: compute embeddings
    embeddings = model.encode(win_df["text"].tolist(), show_progress_bar=True)
    print("  ↳ Embeddings computed.")

    # Step 3: compute RSA matrix
    sim_matrix = cosine_similarity(embeddings)
    np.save(os.path.join(OUTPUT_DIR, f"{base}_RSA_matrix.npy"), sim_matrix)

    # Step 4: save window info
    win_df.to_csv(os.path.join(OUTPUT_DIR, f"{base}_RSA_windows.csv"), index=False)

    # Step 5: plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Semantic RSA matrix – {base}")
    plt.xlabel("Time window index")
    plt.ylabel("Time window index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{base}_RSA_matrix.png"))
    plt.close()

    print(f" Saved RSA matrix, window CSV, and plot for {base}")

print("\n All RSA matrices generated successfully.")
