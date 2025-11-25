# =====================================================================
# Label_PCA_Timepoints_WithinAcross.py
# =====================================================================
# For each video, labels each PCA timepoint as within-event (1) or across-event (0)
# based on consensus boundary times.
#
# Inputs:
#   1. Master PCA timecourse file:
#         C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All_trimmed.csv
#   2. Consensus boundary file:
#         C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv
#
# Logic:
#   - For each video, load all consensus boundary times.
#   - Each Run_time between two boundaries = WITHIN (1)
#   - Each Run_time within ±CROSS_WINDOW seconds of a boundary = ACROSS (0)
#   - All other timepoints are WITHIN (1)
#
# Output:
#   CSV with columns:
#     VideoName, Run_time, Mean_PCA_1 … Mean_PCA_n, Within_or_Across
# =====================================================================

import os
import re
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
PCA_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All.csv"
BOUNDARY_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
OUTPUT_CSV = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"

CROSS_WINDOW = 5  # seconds around each boundary to count as "across"
IGNORE_FIRST_SEC = 60  # optional trim at start if needed

# ----------------------------------------
def read_boundaries(boundary_csv):
    """Return tidy dataframe: Video, boundary_sec."""
    b = pd.read_csv(boundary_csv)
    # Detect likely columns
    name_col = [c for c in b.columns if any(k in c.lower() for k in ["video", "task", "movie", "file"])][0]
    time_col = [c for c in b.columns if any(k in c.lower() for k in ["time", "sec", "boundary"])][0]

    out = b[[name_col, time_col]].copy()
    out.columns = ["Video", "boundary_sec"]
    out["Video"] = out["Video"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    out["boundary_sec"] = pd.to_numeric(out["boundary_sec"], errors="coerce")
    out = out.dropna(subset=["boundary_sec"])
    return out.sort_values(["Video", "boundary_sec"])

# ----------------------------------------
def label_within_across(pca_df, boundaries, cross_window=5):
    """
    Given a single video's PCA dataframe and its boundaries,
    label each Run_time as within-event (1) or across-event (0).
    """
    times = pca_df["Run_time"].to_numpy()
    label = np.ones_like(times, dtype=int)  # default = within (1)

    # mark across (0) for any time within ±cross_window of a boundary
    for b in boundaries:
        mask = np.abs(times - b) <= cross_window
        label[mask] = 0

    pca_df["Within_or_Across"] = label
    return pca_df

# ----------------------------------------
def main():
    print("Loading PCA timecourses...")
    df = pd.read_csv(PCA_FILE)
    if "Task_name" in df.columns:
        df.rename(columns={"Task_name": "VideoName"}, inplace=True)

    # identify PCA columns
    pca_cols = [c for c in df.columns if c.startswith("Mean_PCA_")]
    if not pca_cols:
        raise RuntimeError("No Mean_PCA_ columns found in PCA file.")

    print("Loading boundaries...")
    boundaries = read_boundaries(BOUNDARY_FILE)
    videos = df["VideoName"].unique()

    all_labeled = []

    for vid in videos:
        sub = df[df["VideoName"].str.lower() == vid.lower()].copy()
        print(sub)
        if vid == "Movie Task-lms":
            IGNORE_FIRST_SEC = 75
        if vid == "Movie Task-c4":
            IGNORE_FIRST_SEC = 75
        if vid == "Movie Task-summer":
            IGNORE_FIRST_SEC = 75
        else:
            IGNORE_FIRST_SEC = 60

        sub = sub[sub["Run_time"] >= IGNORE_FIRST_SEC].reset_index(drop=True)

        import os

        vid_clean = os.path.splitext(vid.lower().strip())[0]
        boundaries["Video_clean"] = boundaries["Video"].astype(str).str.lower().str.replace(".mp4", "", regex=False).str.strip()

        bsecs = boundaries[boundaries["Video_clean"] == vid_clean]["boundary_sec"].to_numpy(dtype=float)
        print("Video:", repr(vid))
        print("Boundary videos available:", [repr(v) for v in boundaries["Video"].unique() if "prestige" in v.lower()])


        if len(bsecs) == 0:
            # try relaxed match
            candidates = boundaries[boundaries["Video"].str.lower().str.contains(re.escape(vid.lower()))]
            bsecs = candidates["boundary_sec"].to_numpy(dtype=float)

        if len(bsecs) == 0:
            print(f"⚠️ No boundaries found for {vid}, skipping.")
            continue

        labeled = label_within_across(sub, bsecs, CROSS_WINDOW)
        all_labeled.append(labeled)

    if not all_labeled:
        raise RuntimeError("No videos labeled successfully.")

    out_df = pd.concat(all_labeled, ignore_index=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved labeled PCA data to:\n{OUTPUT_CSV}")
    print(f"Total rows: {len(out_df):,}")
    print(f"Columns: {', '.join(out_df.columns)}")

if __name__ == "__main__":
    main()
