# =====================================================================
# PCA_RSA_perVideo_perPCA_realIntervals.py
# =====================================================================
# For each video and each PCA component:
#   1) Skip first 60 s (or 75 s for long videos; first probe excluded)
#   2) Treat each remaining PCA sample as representing one real interval:
#        10 s (regular videos) or 15 s (long videos)
#   3) Compute RSA directly across those PCA interval scores
#   4) Save RSA as CSV
#   5) Make two plots:
#        - Interval-level RSA (1 cell per probe)
#        - Block-expanded RSA (each cell expanded to 10×10 or 15×15 pixels)
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
INPUT_DIR   = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
OUTPUT_ROOT = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\PCA_RSA"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

SAVE_PLOTS     = True
LONG_VIDEOS    = ["lms", "c4", "summer"]  # case-insensitive name check

# -------------------- HELPERS --------------------
def detect_time_col(df):
    for c in ["Run_time", "Time_rounded", "Time", "t", "sec"]:
        if c in df.columns:
            return c
    return None

def is_long_video(name):
    name = name.lower()
    return any(tag in name for tag in LONG_VIDEOS)

def infer_dt_seconds(times):
    d = np.diff(times)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 10.0
    return float(np.median(d))

def compute_rsa(values):
    """Compute RSA as normalized pairwise similarity between PCA intervals."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return np.empty((0, 0))
    if np.all(np.isnan(vals)) or np.nanstd(vals) == 0:
        return np.full((len(vals), len(vals)), np.nan)

    # Z-score across the entire series
    z = (vals - np.nanmean(vals)) / np.nanstd(vals)
    # Outer product gives pairwise similarity
    S = np.outer(z, z)
    # Clip to valid range
    S = np.clip(S, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S



def block_expand_RSA(S, block_len):
    """Expand each RSA cell into a block of size block_len × block_len for visualization."""
    if S.size == 0:
        return np.empty((0, 0))
    block = np.ones((block_len, block_len))
    return np.kron(S, block)

# -------------------- MAIN --------------------
def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    for file in files:
        video_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file)

        time_col = detect_time_col(df)
        if time_col is None:
            print(f"Skipping {video_name}: no time column found")
            continue

        times = pd.to_numeric(df[time_col], errors="coerce").values
        pca_cols = [c for c in df.columns if c.lower().startswith("pca") or c.lower().startswith("mean_pca")]
        if not pca_cols:
            print(f"Skipping {video_name}: no PCA columns found")
            continue

        # Determine cutoff and dt (sampling interval)
        long_flag = is_long_video(video_name)
        start_cut_s = 75 if long_flag else 60
        dt = infer_dt_seconds(times)
        block_len = int(round(dt))  # for visualization expansion

        for pca in pca_cols:
            out_dir = os.path.join(OUTPUT_ROOT, pca)
            os.makedirs(out_dir, exist_ok=True)

            values = pd.to_numeric(df[pca], errors="coerce").values
            # Drop first probe (before cutoff)
            mask = times > start_cut_s
            values = values[mask]
            times_used = times[mask]
            if values.size == 0:
                print(f"Warning: {video_name} {pca} has no data after cutoff")
                continue

            # Compute RSA
            S = compute_rsa(values)
            out_csv = os.path.join(out_dir, f"{video_name}_RSA.csv")
            pd.DataFrame(S).to_csv(out_csv, index=False)
            print(f"Saved RSA: {pca} → {out_csv} shape={S.shape}")

            if not SAVE_PLOTS:
                continue

            # ---------- (1) Interval-level RSA ----------
            plt.figure(figsize=(6, 5))
            plt.imshow(S, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
            plt.title(f"{pca} RSA (per interval): {video_name}")
            plt.xlabel("Interval (probe index)")
            plt.ylabel("Interval (probe index)")
            cbar = plt.colorbar()
            cbar.set_label("r")
            plt.tight_layout()
            out_png_interval = os.path.join(out_dir, f"{video_name}_RSA_interval.png")
            plt.savefig(out_png_interval, dpi=150)
            plt.close()

            
if __name__ == "__main__":
    main()
