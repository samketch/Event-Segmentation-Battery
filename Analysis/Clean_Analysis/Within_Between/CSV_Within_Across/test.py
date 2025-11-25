# =====================================================================
# Label_PCA_Timepoints_WithinAcross.py
# =====================================================================
# For each video, labels each PCA timepoint as within-event (1) or across-event (0)
# based on consensus boundary times.
#
# Supports both:
#   - Fixed CROSS_WINDOW (in seconds)
#   - Fractional window based on local event length (FRACTIONAL = True)
#
# Outputs labeled CSV:
#   VideoName, Run_time, Mean_PCA_*, Within_or_Across
# =====================================================================

import os
import re
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
PCA_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All_trimmed.csv"
BOUNDARY_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
OUTPUT_CSV = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"

# labeling parameters
CROSS_WINDOW = 5       # seconds around each boundary for "across"
FRACTIONAL = False     # True = adaptive window based on event length
FRAC_F = 0.15          # fraction of local event length if FRACTIONAL=True
W_MIN, W_MAX = 5, 20   # min/max allowed seconds
IGNORE_FIRST_SEC = 60  # optional trim at start if needed

# ----------------------------------------
def read_boundaries(boundary_csv):
    """Return tidy dataframe: Video, boundary_sec."""
    b = pd.read_csv(boundary_csv)
    b.columns = [c.strip() for c in b.columns]
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
def compute_local_windows(boundary_secs, f=0.15, w_min=5, w_max=20):
    """Compute per-boundary adaptive windows based on local event length."""
    b = np.array(sorted(boundary_secs), dtype=float)
    if len(b) == 0:
        return np.array([])
    prev = np.r_[b[0], b[:-1]]
    nxt  = np.r_[b[1:], b[-1]]
    local = np.minimum(b - prev, nxt - b)
    w = np.clip(f * local, w_min, w_max)
    return w

# ----------------------------------------
def label_within_across_fixed(pca_df, boundaries, cross_window=5):
    times = pca_df["Run_time"].to_numpy()
    label = np.ones_like(times, dtype=int)
    for b in boundaries:
        mask = np.abs(times - b) <= cross_window
        label[mask] = 0
    pca_df["Within_or_Across"] = label
    return pca_df

# ----------------------------------------
def label_within_across_fractional(pca_df, boundaries, f=0.15, w_min=5, w_max=20):
    times = pca_df["Run_time"].to_numpy()
    label = np.ones_like(times, dtype=int)
    b = np.array(sorted(boundaries), dtype=float)
    wloc = compute_local_windows(b, f=f, w_min=w_min, w_max=w_max)
    for bb, ww in zip(b, wloc):
        mask = np.abs(times - bb) <= ww
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

        # video-specific trimming
        if vid in ["Movie Task-lms", "Movie Task-c4", "Movie Task-summer"]:
            ignore = 75
        else:
            ignore = IGNORE_FIRST_SEC
        sub = sub[sub["Run_time"] >= ignore].reset_index(drop=True)

        # normalize video name for matching
        vid_clean = os.path.splitext(vid.lower().strip())[0]
        boundaries["Video_clean"] = boundaries["Video"].astype(str).str.lower().str.replace(".mp4", "", regex=False).str.strip()

        bsecs = boundaries[boundaries["Video_clean"] == vid_clean]["boundary_sec"].to_numpy(dtype=float)
        if len(bsecs) == 0:
            candidates = boundaries[boundaries["Video"].str.lower().str.contains(re.escape(vid.lower()))]
            bsecs = candidates["boundary_sec"].to_numpy(dtype=float)

        if len(bsecs) == 0:
            print(f"⚠️ No boundaries found for {vid}, skipping.")
            continue

        if FRACTIONAL:
            labeled = label_within_across_fractional(sub, bsecs, f=FRAC_F, w_min=W_MIN, w_max=W_MAX)
        else:
            labeled = label_within_across_fixed(sub, bsecs, cross_window=CROSS_WINDOW)

        all_labeled.append(labeled)

    if not all_labeled:
        raise RuntimeError("No videos labeled successfully.")

    out_df = pd.concat(all_labeled, ignore_index=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # quick summary counts
    vc = out_df["Within_or_Across"].value_counts().to_dict()
    within_ct = vc.get(1, 0)
    across_ct = vc.get(0, 0)

    print(f"\n✅ Saved labeled PCA data to:\n{OUTPUT_CSV}")
    print(f"Total rows: {len(out_df):,}")
    print(f"Within count: {within_ct}")
    print(f"Across count: {across_ct}")
    print(f"Columns: {', '.join(out_df.columns)}")

if __name__ == "__main__":
    main()
