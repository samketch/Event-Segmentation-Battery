# =====================================================================
# Validate_RSA.py
# =====================================================================
# Validates RSA matrices saved from PCA_RSA_perVideo_perPCA_realIntervals.py
#
# For each RSA CSV:
#   1) Reload the original PCA timecourse
#   2) Recompute the outer-product-of-z RSA manually
#   3) Compare against the saved RSA file
#   4) Print PASS/FAIL summary per file
#
# Output: console summary of match quality
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------
RSA_ROOT   = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\PCA_RSA"
INPUT_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
TOL        = 1e-5
LONG_VIDEOS = ["lms", "c4", "summer"]

# -------------------- HELPERS --------------------
def is_long_video(name):
    name = name.lower()
    return any(tag in name for tag in LONG_VIDEOS)

def detect_time_col(df):
    for c in ["Run_time", "Time_rounded", "Time", "t", "sec"]:
        if c in df.columns:
            return c
    return None

def recompute_rsa_from_series(values):
    """Recompute the outer-product-of-z RSA (the same formula used in PCA_RSA.py)."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return np.empty((0, 0))
    if np.all(np.isnan(vals)) or np.nanstd(vals) == 0:
        return np.full((len(vals), len(vals)), np.nan)
    z = (vals - np.nanmean(vals)) / np.nanstd(vals)
    S = np.outer(z, z)
    S = np.clip(S, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S

# -------------------- MAIN --------------------
def main():
    rsa_files = glob.glob(os.path.join(RSA_ROOT, "Mean_PCA_*", "*_RSA.csv"))
    if not rsa_files:
        print(f"No RSA CSVs found in {RSA_ROOT}")
        return

    total, passed = 0, 0

    for rsa_path in sorted(rsa_files):
        total += 1
        rsa = pd.read_csv(rsa_path).to_numpy(dtype=float)
        pca_name = rsa_path.split(os.sep)[-2]
        video_name = os.path.basename(rsa_path).replace("_RSA.csv", "")

        # Find original CSV
        base_name = video_name.replace("_PCA_timecourse", "")
        orig_candidates = glob.glob(os.path.join(INPUT_DIR, f"{base_name}*.csv"))
        if not orig_candidates:
            print(f"⚠️ No source file found for {video_name}")
            continue
        df = pd.read_csv(orig_candidates[0])
        time_col = detect_time_col(df)
        if time_col is None:
            print(f"⚠️ Skipping {video_name}: no time column found")
            continue

        times = pd.to_numeric(df[time_col], errors="coerce").values
        vals = pd.to_numeric(df[pca_name], errors="coerce").values

        # Apply cutoff (same logic as generator)
        start_cut_s = 75 if is_long_video(video_name) else 60
        mask = times > start_cut_s
        vals = vals[mask]

        if vals.size != rsa.shape[0]:
            print(f"⚠️ Size mismatch for {video_name} | expected {vals.size}, got {rsa.shape[0]}")
            continue

        rsa_recomputed = recompute_rsa_from_series(vals)

        # --- compare matrices ---
        diff = np.nanmax(np.abs(rsa - rsa_recomputed))
        same_shape = rsa.shape == rsa_recomputed.shape
        diag_ok = np.allclose(np.diag(rsa), np.ones(rsa.shape[0]), atol=1e-5)
        symmetric_ok = np.allclose(rsa, rsa.T, atol=1e-5)
        within_tol = diff < TOL

        if same_shape and diag_ok and symmetric_ok and within_tol:
            print(f"✅ PASS: {video_name} | {pca_name} | n={rsa.shape[0]}")
            passed += 1
        else:
            print(f"⚠️ FAIL: {video_name} | {pca_name} | n={rsa.shape[0]}")
            if not same_shape:
                print("   ❌ shape mismatch")
            if not diag_ok:
                print("   ❌ diagonal not all 1s")
            if not symmetric_ok:
                print("   ❌ matrix not symmetric")
            if not within_tol:
                print(f"   ❌ values differ (max diff={diff:.4g})")

    print(f"\nValidation complete: {passed}/{total} RSAs passed all checks.")



# =====================================================================
# Inspect_PCA_Timepoint_Correlation.py
# =====================================================================
# Pick a video, PCA component, and one timepoint (interval index),
# then correlate that interval's z-score with all others.
# Produces a line plot of correlation vs. interval index
# and highlights which parts of the RSA correspond to "white" zones.
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
VIDEO_NAME = "Movie Task-lms_PCA_timecourse"
PCA_NAME   = "Mean_PCA_1"
INTERVAL_INDEX = 39  # the timepoint you suspect is near-zero correlation

RSA_ROOT   = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\PCA_RSA"
INPUT_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
LONG_VIDEOS = ["lms", "c4", "summer"]

# -------------------- HELPERS --------------------
def is_long_video(name):
    name = name.lower()
    return any(tag in name for tag in LONG_VIDEOS)

def detect_time_col(df):
    for c in ["Run_time", "Time_rounded", "Time", "t", "sec"]:
        if c in df.columns:
            return c
    return None

# -------------------- MAIN --------------------
def main():
    # Load original PCA timecourse
    src_file = os.path.join(INPUT_DIR, VIDEO_NAME.replace("PCA_timecourse","PCA_timecourse.csv"))
    df = pd.read_csv(src_file)
    time_col = detect_time_col(df)
    times = pd.to_numeric(df[time_col], errors="coerce").values
    values = pd.to_numeric(df[PCA_NAME], errors="coerce").values

    # Apply cutoff (same as RSA script)
    start_cut_s = 75 if is_long_video(VIDEO_NAME) else 60
    mask = times > start_cut_s
    times = times[mask]
    values = values[mask]

    # Z-score the PCA values
    z = (values - np.nanmean(values)) / np.nanstd(values)

    # Compute outer-product RSA
    rsa = np.outer(z, z)
    np.fill_diagonal(rsa, 1.0)

    n = len(z)
    if INTERVAL_INDEX >= n:
        print(f"❌ Index {INTERVAL_INDEX} out of range (0–{n-1})")
        return

    # Extract correlations for the selected timepoint
    corr_vector = rsa[INTERVAL_INDEX, :]

    # Plot correlation profile
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n), corr_vector, marker='o', color='steelblue')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(INTERVAL_INDEX, color='r', linestyle=':', label=f"Interval {INTERVAL_INDEX}")
    plt.title(f"{PCA_NAME} correlations for interval {INTERVAL_INDEX}\n{VIDEO_NAME}")
    plt.xlabel("Other intervals (probe index)")
    plt.ylabel("Similarity (r)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: print a few stats
    print(f"\nSelected interval index: {INTERVAL_INDEX}")
    print(f"Mean correlation to others: {np.mean(corr_vector):.3f}")
    print(f"Min: {np.min(corr_vector):.3f}, Max: {np.max(corr_vector):.3f}")

if __name__ == "__main__":
    main()

