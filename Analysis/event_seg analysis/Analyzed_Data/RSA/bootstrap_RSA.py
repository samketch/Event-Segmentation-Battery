# ============================================================================
# Script will run 
    #Semantic Embedding/Video Transcript (timestamped) RSA
    #Three seperate within-across event analyses (bootstrat, mid-point, fake boundary midpoint)
# Author: Sam Ketcheson
# ============================================================================
"""
Part 1: RSA

Computes an all-to-all semantic similarity matrix for each
video based on sliding-window text embeddings of the transcript.

Fixes included:
 - Forces 'start' and 'end' columns to numeric (prevents truncation bug)
 - Adds diagnostic prints showing max time and number of windows
 - Ensures RSA windows span full video duration (e.g., ~1300 s)

Outputs (per video):
 - *_RSA_matrix.npy  : full cosine similarity matrix (time × time)
 - *_RSA_matrix.png  : visual heatmap of the matrix
 - *_RSA_windows.csv : time and text per window
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os                   # For directory navigation and file management
import numpy as np           # Numerical arrays and mathematical operations
import pandas as pd          # CSV reading/writing and data wrangling
import matplotlib.pyplot as plt  # Plotting for RSA heatmaps and histograms
from sentence_transformers import SentenceTransformer  # Pretrained embedding model
from sklearn.metrics.pairwise import cosine_similarity # Compute pairwise cosine similarity
from scipy.stats import percentileofscore               # Used for computing empirical p-values


# ----------------------------------------------------------------------------
# CONFIGURATION SECTION
# ----------------------------------------------------------------------------
# Paths for input transcripts and output analysis folders
TRANSCRIPT_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Transcripts"
RSA_DIR     = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\RSA"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR      = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\RSA\Within_Across_Bootstrap" 
BOOTSTRAP_DIR = os.path.join(OUTPUT_DIR, "Bootstrap")
MIDDLEPOINT_DIR = os.path.join(OUTPUT_DIR, "Middlepoint")
FAKEBOUNDARY_DIR = os.path.join(OUTPUT_DIR, "Fake_Boundary")

# Ensure output directories exist before attempting to save files
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BOOTSTRAP_DIR, exist_ok=True)
os.makedirs(MIDDLEPOINT_DIR, exist_ok=True)
os.makedirs(FAKEBOUNDARY_DIR, exist_ok=True)



# Core parameters for the sliding-window RSA procedure
WINDOW_SIZE = 15 #Determine how many seconds of dialogue each window should cover
STEP_SIZE   = 1    # Move the window by X second each step
BOOTSTRAPS = 1000  # Number of resampling iterations for across-event bootstrap
COMBINE_LEFT_RIGHT = True  # Whether to pool across left/right sides of event
SHIFT_RADIUS = 0   # How many indices to shift segment comparisons (for stability)
MIDPOINT_FRAC = 0.25  # Fraction determining segment spacing in middle-point analysis
MODEL_NAME  = "all-MiniLM-L6-v2"  # Compact and efficient sentence-transformer model

# Load model into memory (only once)
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ----------------------------------------------------------------------------
# FUNCTION: create_windows
# ----------------------------------------------------------------------------
def load_kde_boundaries(video_name, kde_csv_path):
    df = pd.read_csv(kde_csv_path)
    # Find column that likely stores boundary times
    time_col = [c for c in df.columns if any(k in c.lower() for k in ["bound", "time", "second"])][0]
    # Filter for this video
    vid_col = [c for c in df.columns if "video" in c.lower()][0]
    vals = df.loc[df[vid_col].str.contains(video_name, case=False, na=False), time_col].dropna().to_numpy()
    return np.sort(vals) if len(vals) > 0 else None

def create_windows(df, window_size, step_size):
    """Slide a fixed-size window across the transcript and group text into segments.

    Args:
        df (pd.DataFrame): Transcript dataframe containing start times and text.
        window_size (float): Duration of each time window in seconds.
        step_size (float): Step size between successive windows in seconds.

    Returns:
        pd.DataFrame: A dataframe with two columns – 'time' (midpoint of window)
                      and 'text' (concatenated dialogue lines within window).
    """
    # Find the last timestamp in transcript to know how far to slide
    max_time = df["start"].max()
    windows = []  # Store window metadata
    t = 0         # Initialize window start position

    # Iterate over entire transcript timeline
    while t < max_time:
        t_end = t + window_size  # Define end of window

        # Extract transcript lines that begin within this window
        chunk = df[(df["start"] >= t) & (df["start"] < t_end)]

        if not chunk.empty:
            # Concatenate all dialogue text in this window
            text = " ".join(chunk["text"].tolist())
            # Record midpoint of window and its text content
            windows.append({"time": t + window_size / 2, "text": text})

        # Advance to the next window start
        t += step_size 

    # Return full list as DataFrame for later embedding
    return pd.DataFrame(windows)

# ----------------------------------------------------------------------------
# MAIN LOOP – PART 1: Build RSA Matrices for Each Transcript
# ----------------------------------------------------------------------------
for file in os.listdir(TRANSCRIPT_DIR):
    # Only process transcript CSVs ending with '_aligned.csv'
    if not file.endswith("_aligned.csv"):
        continue

    base = file.replace("_aligned.csv", "")  # Base name without suffix
    print(f"\nProcessing transcript: {base}")

    # Load the transcript file
    df = pd.read_csv(os.path.join(TRANSCRIPT_DIR, file))

    # Convert 'start' and 'end' columns to numeric, coercing invalid entries
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")

    # Drop rows missing start times or text entries
    df = df.dropna(subset=["start", "text"])

    # Compute video duration for diagnostics
    max_time = df["start"].max()
    print(f"  ↳ Duration: {max_time:.1f}s | Rows: {len(df)}")

    # Sanity check for very short transcripts
    if max_time < WINDOW_SIZE:
        print(f" Skipping {base}: too short ({max_time:.1f}s)")
        continue

    # STEP 1: Create sliding windows
    win_df = create_windows(df, WINDOW_SIZE, STEP_SIZE)
    print(f"  ↳ Created {len(win_df)} windows (size={WINDOW_SIZE}s, step={STEP_SIZE}s)")

    # STEP 2: Compute text embeddings for each window
    embeddings = model.encode(win_df["text"].tolist(), show_progress_bar=True)
    print("  ↳ Text embeddings computed.")

    # STEP 3: Compute all-to-all cosine similarities (RSA matrix)
    sim_matrix = cosine_similarity(embeddings)

    # STEP 4: Save outputs
    np.save(os.path.join(RSA_DIR, f"{base}_RSA_matrix.npy"), sim_matrix)
    win_df.to_csv(os.path.join(RSA_DIR, f"{base}_RSA_windows.csv"), index=False)

    # STEP 5: Plot and save a heatmap for visual inspection
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(label="Cosine similarity")
    plt.title(f"Semantic RSA matrix – {base}")
    plt.xlabel("Time window index")
    plt.ylabel("Time window index")
    # --- Overlay event boundary lines --

    KDE_CSV = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
    boundaries = load_kde_boundaries(base, KDE_CSV)


    if boundaries is not None:
        # Convert seconds → matrix indices
        total_duration = sim_matrix.shape[0] * STEP_SIZE
        scale_factor = sim_matrix.shape[0] / total_duration
        indices = np.ceil(boundaries * scale_factor).astype(int)
        indices = np.unique(indices[indices <= sim_matrix.shape[0]])

        # Add vertical and horizontal lines
        for idx in indices:
            plt.axvline(x=idx, color="white", linestyle="--", linewidth=0.6, alpha=0.7)
            plt.axhline(y=idx, color="white", linestyle="--", linewidth=0.6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(RSA_DIR, f"{base}_RSA_matrix.png"))
    plt.close()

    print(f"Saved RSA matrix, window CSV, and heatmap for {base}")

print("\nAll RSA matrices generated successfully.")




# ============================================================================
# PART 2 – Within- vs Across-Event RSA Bootstrap + Middle Point Analyses
# ============================================================================
# Purpose: Quantify differences in semantic similarity within and across event
# boundaries, test via bootstrapping, and generate diagnostic plots.
# ----------------------------------------------------------------------------
# 1. Bootstrap: compare within-event vs random across-event similarities.
# 2. Middle-point: compare short segments inside vs across events.
# 3. Fake-boundary: Check using artificial event splits.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# FUNCTION: load_boundaries
# ----------------------------------------------------------------------------
def load_boundaries(base):
    """Load event boundary times for a given video.

    Searches multiple possible file name variants for flexibility and returns
    an array of boundary times (in seconds).
    """
    candidates = [
        f"{base}_consensus_boundaries.csv",
        f"{base}_consensus_bouundaries.csv",
        f"{base}_boundaries.csv",
        f"{base}_consensus.csv",
    ]
    for c in candidates:
        path = os.path.join(BOUNDARY_DIR, c)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Identify correct time column by name keywords
            col = [x for x in df.columns if any(k in x.lower() for k in ["bound", "time", "second"])][0]
            return df[col].dropna().to_numpy()
    return None

# ----------------------------------------------------------------------------
# FUNCTION: bootstrap_across
# ----------------------------------------------------------------------------
def bootstrap_across(vals, n_samples, bootstraps=1000, random_state=0):
    """Resample across-event similarities to form a null distribution.

    Args:
        vals (array): Array of across-event similarity values.
        n_samples (int): Number of samples to draw each bootstrap.
        bootstraps (int): Number of bootstrap iterations.
    Returns:
        np.array: Bootstrapped mean similarity distribution.
    """
    rng = np.random.default_rng(random_state)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]  # Remove NaNs
    if len(vals) == 0:
        return np.array([])
    # Randomly sample with replacement, compute mean each time
    boot_means = [np.mean(rng.choice(vals, size=n_samples, replace=True)) for _ in range(bootstraps)]
    return np.array(boot_means)

# ----------------------------------------------------------------------------
# FUNCTION: upper_triangle_mean
# ----------------------------------------------------------------------------
def upper_triangle_mean(block, min_lag = 0):
    """Compute mean of upper-triangle values of a square matrix.

    Args:
        block (np.array): A square within-event similarity submatrix.
    Returns:
        tuple: (mean value, number of unique pairs)
    """
    if block.size == 0:
        return np.nan, 0
    # k = min_lag lifts the diagonal by that many indices
    iu = np.triu_indices_from(block, k=min_lag)
    if iu[0].size == 0:
        return np.nan, 0
    return np.mean(block[iu]), iu[0].size

# ----------------------------------------------------------------------------
# FUNCTION: compute_middle_point_values
# ----------------------------------------------------------------------------
def compute_middle_point_values(M, start, end, n_time, shift_radius=0, frac=0.25, seg_len=5):
    """Compute within- vs across-event similarity for middle segments.

    Selects two small segments inside the event (within) and two segments
    straddling the real boundary (across) to compare mean similarities.
    """
    length = end - start
    if length < (seg_len * 2 + 1):
    # If the event is too short to fit two segments of size seg_len
    # (plus at least one frame of separation), skip it entirely.
    # This prevents trying to sample overlapping or invalid segments.
        return None  # Skip very short events

    # Compute the total event length and key positions inside it
    half_len = length // 2                     # Integer midpoint of the event
    d = max(1, int(np.floor(half_len * frac))) # Distance offset between segments (fraction of half-length)
    center = start + half_len                  # Absolute index of the event midpoint

    # Ensure segments used for within comparison are at least one window apart

    min_sep = int(np.ceil(WINDOW_SIZE / STEP_SIZE))
    d = max(d, min_sep)

    # Define helper function to constrain any computed index
    # within the valid matrix boundaries [lo, hi]
    def clamp(a, lo, hi):
        return max(lo, min(hi, a))

    # Initialize containers for similarity values
    within_vals, across_vals = [], []

    # Optionally shift comparison windows slightly forward/backward
    # by ±shift_radius to average over small offsets (robustness)
    for s in range(-shift_radius, shift_radius + 1):

        # --------------------------
        # WITHIN-EVENT SEGMENTS
        # --------------------------
        # Define first within-event segment (seg1)
        seg1_start = clamp(center - d - seg_len + s, start, end - seg_len)
        seg1_end = seg1_start + seg_len

        # Define second within-event segment (seg2)
        seg2_start = clamp(center + d + s, start, end - seg_len)
        seg2_end = seg2_start + seg_len

        # Extract similarity block between seg1 and seg2
        within_block = M[seg1_start:seg1_end, seg2_start:seg2_end]
        if within_block.size == 0:
            # Skip if segment indices fall out of range
            continue

        # Average all pairwise similarities inside that within-event block
        within_vals.append(np.mean(within_block))

        # --------------------------
        # ACROSS-BOUNDARY SEGMENTS
        # --------------------------
        b = end  # Define the event boundary (end of current event)

        # Segment 3: before boundary (just inside current event)
        seg3_start = clamp(b - d - seg_len + s, 0, n_time - seg_len)
        seg3_end = seg3_start + seg_len

        # Segment 4: after boundary (start of next event)
        seg4_start = clamp(b + d + s, 0, n_time - seg_len)
        seg4_end = seg4_start + seg_len

        # Guard: skip if segment 4 would exceed matrix dimensions
        if seg4_end > n_time:
            continue

        # Extract across-boundary similarity block
        across_block = M[seg3_start:seg3_end, seg4_start:seg4_end]
        if across_block.size > 0:
            # Average and record mean similarity across boundary
            across_vals.append(np.mean(across_block))

    # If no valid within-event values were collected, skip this event
    if not within_vals:
        return None

    # Compute overall means for within and across segments
    return {
        "within_value": float(np.mean(within_vals)),                    # Mean within-event similarity
        "across_value": float(np.mean(across_vals)) if across_vals else np.nan,  # Mean across-event similarity
        "lag_used": int(d),                                             # Distance offset used
        "center_idx": int(center),                                      # Event midpoint index
        "segment_length": seg_len,                                      # Segment duration (in time windows)
        "type": "segment_middle"                                        # Tag for analysis type
    }

def compute_fake_boundary_midpoint_values(M, start, end, next_start, n_time):
    """Compute fake-boundary similarity comparisons.

    Splits an event into two halves (creating a "fake boundary") and compares:
      1. Similarity between the midpoints of the two halves (within-event)
      2. Similarity between the midpoint of the second half and
         the midpoint of the *next* event (across-event).

         ## divide window size by 2
    """

    length = end - start
    if length < 4:
        return None

    # Define midpoints
    mid = start + length // 2
    mid1 = start + (mid - start) // 2
    mid2 = mid + (end - mid) // 2

    # Average a small patch around each midpoint
    patch = 5
    half = patch // 2
    r1 = slice(max(0, mid1 - half), min(n_time, mid1 + half + 1))
    r2 = slice(max(0, mid2 - half), min(n_time, mid2 + half + 1))
    within_value = float(np.mean(M[r1, r2]))

    # Across: midpoint of next event
    if next_start is not None and next_start < n_time:
        next_mid1 = next_start + (next_start + (end - start)//2 - next_start) // 2
        r_next = slice(max(0, next_mid1 - half), min(n_time, next_mid1 + half + 1))
        across_value = float(np.mean(M[r2, r_next])) if (mid2 < n_time and next_mid1 < n_time) else np.nan
    else:
        across_value = np.nan

    return {
        "within_value": within_value,
        "across_value": across_value,
        "difference": within_value - across_value,
        "mid1": mid1,
        "mid2": mid2,
        "type": "fake_boundary"
    }
# ----------------------------------------------------------------------------
# MAIN ANALYSIS LOOP 
# ----------------------------------------------------------------------------
# Purpose: For each video’s RSA matrix, extract per-event within/across
# similarity values, run bootstrap significance tests, and compute
# middle-point and fake-boundary controls.

# summary          → holds per-video within/across bootstrap results (DataFrames)
# summary_middle   → holds middle-point + fake-boundary results for all events
summary, summary_middle = [], []


# Iterate over all RSA matrices created in Part 1
for file in os.listdir(RSA_DIR):
    if not file.endswith("_RSA_matrix.npy"):
        # Skip non-RSA files (only process semantic similarity matrices)
        continue

    base = file.replace("_RSA_matrix.npy", "")  # Strip suffix to get video name
    print(f"\nAnalyzing {base}…")

    # ------------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------------
    # Load the full RSA (similarity) matrix for this video
    M = np.load(os.path.join(RSA_DIR, file))
    n_time = M.shape[0]  # The number of sliding windows (timepoints)

    # Load consensus event boundaries (from KDE outputs, in seconds)
    boundaries = load_boundaries(base)
    if boundaries is None:
        print(f"No boundaries found for {base}.")
        continue  # Skip this video if no boundary file is available

    # ------------------------------------------------------------------------
    # 2. MAP BOUNDARY TIMES → MATRIX INDICES
    # ------------------------------------------------------------------------
    # RSA indices are 1 per second (since STEP_SIZE = 1),
    # so scale_factor converts seconds to matrix indices.
    total_duration = n_time * STEP_SIZE
    scale_factor = n_time / total_duration

    # Convert boundary times (sec) to integer indices in the RSA matrix
    indices = np.ceil(boundaries * scale_factor).astype(int)

    # Remove duplicates, keep only valid indices (within matrix size)
    indices = np.unique(indices[indices <= n_time])

    # Ensure last matrix index (video end) is included as final boundary
    if indices[-1] < n_time:
        indices = np.append(indices, n_time)

    # Construct arrays of event start and end indices
    # Example: if indices = [100, 250, 400], then
    # event_starts = [0, 100, 250]; event_ends = [100, 250, 400]
    event_starts = np.concatenate(([0], indices[:-1]))
    event_ends = indices
    event_results = []  # Store per-event stats for this video

    # ------------------------------------------------------------------------
    # 3. EVENT-BY-EVENT ANALYSIS
    # ------------------------------------------------------------------------
    for e, (start, end) in enumerate(zip(event_starts, event_ends)):
        if end - start < 1:
            # Skip degenerate events (zero or one timepoint)
            continue

        # ------------------------------------------------------------
        # WITHIN-EVENT SIMILARITY
        # ------------------------------------------------------------
        # Extract the block of the RSA matrix corresponding to all pairs of
        # time windows within this event: M[start:end, start:end]
        within_block = M[start:end, start:end]

        # Compute average similarity across all unique pairs inside this block
        within_mean, n_pairs = upper_triangle_mean(within_block)

        # Skip event if too small or contains NaN values
        if n_pairs == 0 or np.isnan(within_mean):
            continue

        # ------------------------------------------------------------
        # ACROSS-EVENT SIMILARITY
        # ------------------------------------------------------------
        # "Across" comparisons look at pairs that span event boundaries.
        # Two possible directions:
        #   - right_block: current event → later timepoints (after end)
        #   - left_block:  earlier timepoints → current event (before start)
        right_block = M[start:end, end:n_time]
        left_block = M[0:start, start:end]

        # Combine left/right sides if COMBINE_LEFT_RIGHT = True
        across_vals = (
            np.concatenate([left_block.flatten(), right_block.flatten()])
            if COMBINE_LEFT_RIGHT
            else right_block.flatten()
        )

        # ------------------------------------------------------------
        # BOOTSTRAP TEST (Within vs Across)
        # ------------------------------------------------------------
        if across_vals.size == 0:
            # No valid across-event values (edge cases near start/end)
            z, p, across_mean = np.nan, np.nan, np.nan
        else:
            # Draw 1000 bootstrap samples of across-event means
            boot = bootstrap_across(across_vals, n_pairs, BOOTSTRAPS)

            # Compute overall across-event mean from bootstraps
            across_mean = np.mean(boot)

            # Compute z-score comparing within-event mean to bootstrap distribution
            z = (within_mean - np.mean(boot)) / np.std(boot)

            # Compute one-tailed p-value: proportion of boot means ≥ within_mean
            p = 1 - percentileofscore(boot, within_mean) / 100.0

        # Store per-event results for this video
        event_results.append({
            "video": base,           # Video identifier
            "event_index": e + 1,    # Event number (1-indexed)
            "start_idx": start,      # Start index in RSA matrix
            "end_idx": end,          # End index in RSA matrix
            "within_mean": within_mean,  # Mean similarity inside event
            "across_mean": across_mean,  # Mean similarity outside event (bootstrap)
            "z": z,                      # Standardized difference score
            "p": p,                      # Empirical p-value
            "n_pairs": n_pairs           # Number of pairs used in within calculation
        })

        # ------------------------------------------------------------
        # MIDDLE-POINT ANALYSIS
        # ------------------------------------------------------------
        # Compare short segments inside event vs segments across the boundary.
        mp = compute_middle_point_values(
            M, start, end, n_time,
            shift_radius=SHIFT_RADIUS,
            frac=MIDPOINT_FRAC
        )
        if mp is not None:
            summary_middle.append({
                "video": base,
                "event_index": e + 1,
                **mp
            })

        # ------------------------------------------------------------
        # FAKE-BOUNDARY ANALYSIS
        # ------------------------------------------------------------
        # Control analysis: split current event into halves,
        # compare internal vs true boundaries.
        lag_excl = int(np.ceil(WINDOW_SIZE / STEP_SIZE))  
        within_mean, n_pairs = upper_triangle_mean(within_block, min_lag=lag_excl)

        # Exclude windows too close to the boundary to avoid overlap leakage
        #gap = lag_excl
        gap = 0

        # Right side: start comparing to indices at least 'gap' after 'end'
        right_start = min(end + gap, n_time)
        right_block = M[start:end, right_start:n_time]

        # Left side: only up to indices at least 'gap' before 'start'
        left_end = max(0, start - gap)
        left_block = M[0:left_end, start:end]

        if COMBINE_LEFT_RIGHT:
            across_vals = np.concatenate([left_block.flatten(), right_block.flatten()])
        else:
            across_vals = right_block.flatten()

        
        next_start = event_starts[e + 1] if e + 1 < len(event_starts) else None
        fb = compute_fake_boundary_midpoint_values(M, start, end, next_start, n_time)
        if fb is not None:
            summary_middle.append({
                "video": base,
                "event_index": e + 1,
                **fb
            })

    # ------------------------------------------------------------------------
    # 4. SAVE PER-VIDEO RESULTS
    # ------------------------------------------------------------------------
    if event_results:
        # Convert event-level dictionary list → DataFrame
        df = pd.DataFrame(event_results)

        # Save detailed per-event bootstrap results to CSV
        # Save within/across bootstrap results to its own subfolder
        df.to_csv(
            os.path.join(BOOTSTRAP_DIR, f"{base}_within_across_bootstrap.csv"),
            index=False
            )

        # Store DataFrame for overall summary (combined later)
        summary.append(df)
        print(f"{len(df)} events analyzed and saved for {base}.")


# =========================
# Save combined summaries (Part 2)
# =========================
# After looping all videos, concatenate results and output combined CSVs.
if summary:
    summary_all = pd.concat(summary, ignore_index=True)
    summary_all.to_csv(os.path.join(BOOTSTRAP_DIR, "within_across_summary.csv"), index=False)
    print("Combined bootstrap summary saved.")

if summary_middle:
    df_middle = pd.DataFrame(summary_middle)
    df_middle.to_csv(os.path.join(MIDDLEPOINT_DIR, "middlepoint_summary.csv"), index=False)
    print(f"Middle point summary saved to {MIDDLEPOINT_DIR}")

    # --- Save fake-boundary analysis separately (cleaned) ---
    # Extract only rows produced by the fake-boundary function and keep a
    # minimal set of columns for downstream visualization/inspection.
    df_fake = df_middle[df_middle["type"] == "fake_boundary"].copy()
    if not df_fake.empty:
        keep_cols = ["video", "event_index", "within_value", "across_value", "difference", "mid1", "mid2", "type"]
        df_fake = df_fake[[c for c in keep_cols if c in df_fake.columns]]
        fake_csv = os.path.join(FAKEBOUNDARY_DIR, "fakeboundary_summary.csv")
        df_fake.to_csv(fake_csv, index=False)
        print(f"Fake-boundary summary saved to {FAKEBOUNDARY_DIR}")

    # --- Per-video histograms ---
    # For each video and analysis type, plot histogram of (within - across)
    # differences to visualize effect distributions.
    for vid in df_middle["video"].unique():
        subset = df_middle[df_middle["video"] == vid]
        for t in subset["type"].unique():
            sub = subset[subset["type"] == t]
            plt.hist(sub["within_value"] - sub["across_value"], bins=30, edgecolor="k")
            plt.xlabel("Within minus across similarity")
            plt.ylabel("Count")
            plt.title(f"{vid}: {t} analysis")
            plt.tight_layout()
            # Save plots to different folders for fake-boundary vs others
            # Route histograms to their respective analysis folders
            save_dir = (
                FAKEBOUNDARY_DIR if "fake" in t.lower() else
                MIDDLEPOINT_DIR if "middle" in t.lower() else
                BOOTSTRAP_DIR
            )
            plt.savefig(os.path.join(save_dir, f"{vid}_{t}_difference_distribution.png"))
            plt.close()
            print(f"Saved histogram for {vid} ({t})")
    # --- Combined histogram for all fake-boundary results ---
    if not df_fake.empty:
        plt.hist(df_fake["difference"].dropna(), bins=30, edgecolor="k")
        plt.xlabel("Within minus across similarity")
        plt.ylabel("Count")
        plt.title("Combined Fake-Boundary Within vs Across Differences")
        plt.tight_layout()
        combined_path = os.path.join(FAKEBOUNDARY_DIR, "combined_fakeboundary_difference_distribution.png")
        plt.savefig(FAKEBOUNDARY_DIR)
        plt.close()
        print(f"Combined fake-boundary histogram saved to {FAKEBOUNDARY_DIR}")
    
        # --- Combined histogram for all MIDDLE-POINT results ---
    df_middlepoint = df_middle[df_middle["type"] == "segment_middle"].copy()
    if not df_middlepoint.empty:
        plt.hist(df_middlepoint["within_value"] - df_middlepoint["across_value"], bins=30, edgecolor="k")
        plt.xlabel("Within minus across similarity")
        plt.ylabel("Count")
        plt.title("Combined Middle-Point Within vs Across Differences")
        plt.tight_layout()
        combined_mid_path = os.path.join(MIDDLEPOINT_DIR, "combined_middlepoint_difference_distribution.png")
        plt.savefig(combined_mid_path)
        plt.close()
        print(f"Combined middle-point histogram saved to {combined_mid_path}")

    # --- Combined histogram for BOOTSTRAP results ---
    bootstrap_summary_path = os.path.join(BOOTSTRAP_DIR, "within_across_summary.csv")
    if os.path.exists(bootstrap_summary_path):
        df_bootstrap = pd.read_csv(bootstrap_summary_path)
        if not df_bootstrap.empty:
            plt.hist(df_bootstrap["within_mean"] - df_bootstrap["across_mean"], bins=30, edgecolor="k")
            plt.xlabel("Within minus across similarity")
            plt.ylabel("Count")
            plt.title("Combined Bootstrap Within vs Across Differences")
            plt.tight_layout()
            combined_boot_path = os.path.join(BOOTSTRAP_DIR, "combined_bootstrap_difference_distribution.png")
            plt.savefig(combined_boot_path)
            plt.close()
            print(f"Combined bootstrap histogram saved to {combined_boot_path}")


    # Quick overall summary print: average (within - across) across all events
    df = pd.read_csv(os.path.join(BOOTSTRAP_DIR,("within_across_summary.csv")))
    mean_diff = df['within_mean'] - df['across_mean']
    print(mean_diff.mean(), mean_diff.std())
