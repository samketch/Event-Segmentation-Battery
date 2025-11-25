"""
=====================================================================
 Visual Change × Event Segmentation Permutation Correlation
=====================================================================
Computes correlation between frame-to-frame visual change
and KDE boundary density, with a circular-shift permutation test.

Input:
 - *_visualchange.csv : Time(s), VisualChange
 - *_kde_timeseries.csv : Time(s), BoundaryDensity

Output:
 - /perm_hists/*.png : Null distribution plots
 - /perm_correlation/*.png : Overlapping timecourse plots
 - permutation_summary.csv : observed r + p_perm per video

Author: Sam Ketcheson
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import glob

# =========================
# Config
# =========================
VISCHANGE_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Neural_model\Analyzed_Data\Frame_Embeddings"
KDE_DIR       = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Neural_model\perm"
VIDEO_DURATION = 22 * 60  # 22 minutes
N_PERM = 1000             # number of permutations

BASE_OUT = os.path.join(OUTPUT_DIR, "permutation")
HIST_DIR = os.path.join(BASE_OUT, "perm_hists")
CORR_DIR = os.path.join(BASE_OUT, "perm_correlation")

os.makedirs(BASE_OUT, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)
# =========================

results = []

visual_files = glob.glob(os.path.join(VISCHANGE_DIR, "*_visualchange.csv"))

for vis_file in visual_files:
    base = os.path.basename(vis_file)
    video_id = base.split("_")[0]  # e.g., "friends1"
    if video_id == "12":
        video_id = "12 years_6m"
    elif video_id == "500Days":
        video_id = "Movie Task-summer"
    elif video_id == "backToFuture":
        video_id = "back to the future_6m"
    elif video_id == "c4":
        video_id = "Movie Task-c4"
    elif video_id == "lms":
        video_id = "Movie Task-lms"
    elif video_id == "prestige":
        video_id = "the_prestige_6m"
    elif video_id == "pulpFiction":
        video_id = "pulp_fiction_6m"
    elif video_id == "shawshank":
        video_id = "shawshank clip_6m"
    print(video_id)

    kde_file = os.path.join(KDE_DIR, f"{video_id}_kde_timeseries.csv")
    kde_file = os.path.join(KDE_DIR, f"{video_id}_kde_timeseries.csv")

    if not os.path.exists(kde_file):
        print(f"Skipping {video_id}: KDE file not found at {kde_file}")
        continue

    print(f"\n=== Processing {video_id} ===")

    # --- Load visual change data ---
    vis = pd.read_csv(vis_file)
    vis_time = pd.to_numeric(vis["Time(s)"], errors="coerce").values
    vis_change = pd.to_numeric(vis["VisualChange"], errors="coerce").fillna(0).values

    # --- Load KDE data ---
    kde = pd.read_csv(kde_file)
    kde_time = pd.to_numeric(kde["Time(s)"], errors="coerce").values
    kde_density = pd.to_numeric(kde["BoundaryDensity"], errors="coerce").values

    # --- Common 1 Hz time axis ---
    #t = np.arange(0, VIDEO_DURATION + 1, 1)
    #vis_interp = np.interp(t, vis_time, vis_change)
    #kde_interp = np.interp(t, kde_time, kde_density)
    
    # =========================
    # Temporal resampling
    # =========================
    BIN_SIZE = 5  # seconds per bin for visual change only

    # 1-Hz grid for KDE (leave untouched)
    t_kde = np.arange(0, VIDEO_DURATION + 1, 1)
    kde_interp = np.interp(t_kde, kde_time, kde_density)

    # Interpolate visual-change at full 1-Hz resolution first
    vis_interp_full = np.interp(t_kde, vis_time, vis_change)

    # Now average visual change within BIN_SIZE-second bins
    n_bins = int(np.ceil(len(vis_interp_full) / BIN_SIZE))
    vis_binned = np.array([
        np.nanmean(vis_interp_full[i*BIN_SIZE:(i+1)*BIN_SIZE])
        for i in range(n_bins)
    ])

    # Define corresponding bin centers in seconds
    t_vis = np.arange(BIN_SIZE/2, VIDEO_DURATION + BIN_SIZE/2, BIN_SIZE)
    t_vis = t_vis[:len(vis_binned)]  # truncate if slightly long

    # Interpolate KDE onto the visual bin centers
    kde_for_bins = np.interp(t_vis, t_kde, kde_interp)

    # Z-score both
    vis_z = (vis_binned - np.nanmean(vis_binned)) / np.nanstd(vis_binned)
    kde_z = (kde_for_bins - np.nanmean(kde_for_bins)) / np.nanstd(kde_for_bins)

    # --- Align array lengths (safely handles rounding mismatches) ---
    min_len = min(len(vis_z), len(kde_z))
    vis_z = vis_z[:min_len]
    kde_z = kde_z[:min_len]


    # --- Observed correlation ---
    r_obs, _ = pearsonr(vis_z, kde_z)
    print(f"Observed correlation: r = {r_obs:.3f}")

    # --- Permutation test (circular shift) ---
    rng = np.random.default_rng()
    r_null = []
    for i in range(N_PERM):
        shift = rng.integers(low=1, high=len(vis_z))
        vis_shifted = np.roll(vis_z, shift)
        r, _ = pearsonr(vis_shifted, kde_z)
        r_null.append(r)

    r_null = np.array(r_null)

    # --- Permutation p-value ---
    p_perm = (np.sum(r_null >= r_obs) + 1) / (N_PERM + 1)
    print(f"Permutation p-value = {p_perm:.4f} (N={N_PERM})")

    # --- Save summary row ---
    results.append({
        "video": video_id,
        "r_obs": r_obs,
        "p_perm": p_perm
    })

    # --- Plot 1: null distribution ---
    plt.figure(figsize=(8, 5))
    plt.hist(r_null, bins=40, alpha=0.7, color="gray", edgecolor="black")
    plt.axvline(r_obs, color="red", linestyle="--", linewidth=2, label=f"Observed r={r_obs:.3f}")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Frequency")
    plt.title(f"{video_id} Permutation Test (N={N_PERM})\nObserved r={r_obs:.3f}, p={p_perm:.4f}")
    plt.legend()
    plt.tight_layout()
    out_fig_perm = os.path.join(HIST_DIR, f"{video_id}_permutation_hist.png")
    plt.savefig(out_fig_perm, dpi=300)
    plt.close()
    print(f"Saved null distribution plot: {out_fig_perm}")

    # --- Plot 2: overlapping z-scored timecourses ---
    plt.figure(figsize=(12, 5))
    plt.plot(t_vis, vis_z, label="Visual change (cosine distance)", alpha=0.8, color="steelblue")
    plt.plot(t_vis, kde_z, label="KDE boundary density", alpha=0.8, color="darkorange")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-scored value")
    plt.title(f"{video_id}: Visual Change vs Boundary Density\nObserved r={r_obs:.2f}, p_perm={p_perm:.3g}")
    plt.legend()
    plt.tight_layout()
    out_fig_line = os.path.join(CORR_DIR, f"{video_id}_timecourse.png")
    plt.savefig(out_fig_line, dpi=300)
    plt.close()
    print(f"Saved timecourse plot: {out_fig_line}")

# --- Save summary table ---
out_csv = os.path.join(BASE_OUT, "permutation_summary.csv")
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n Saved summary results to {out_csv}")
