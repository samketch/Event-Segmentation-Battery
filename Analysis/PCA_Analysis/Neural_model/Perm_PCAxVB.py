"""
=====================================================================
 Visual Change × mDES PCA Permutation Correlation
=====================================================================
Computes correlation between frame-to-frame visual change
and mDES PCA timecourses, with a circular-shift permutation test.

Input:
 - *_visualchange.csv : Time(s), VisualChange
 - *_PCA_timecourse.csv : Time(s), PCA1, PCA2, PCA3, ...

Output:
 - /perm_hists/<video>/*.png : Null distribution plots (per PCA)
 - /perm_correlation/<video>/*.png : Overlapping timecourse plots
 - permutation_summary.csv : observed r + p_perm per video & PCA
 - global_permutation_summary.csv : global correlation & p_perm per PCA

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
VISCHANGE_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Neural_model\Frame_Embeddings"
MDES_DIR      = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
OUTPUT_DIR    = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\PCA_Analysis\Neural_model"

VIDEO_DURATION = 22 * 60  # 22 minutes
N_PERM = 1000
IGNORE_FIRST_SEC = 60
BIN_SIZE = 5  # seconds per visual bin

BASE_OUT = os.path.join(OUTPUT_DIR, "permutation")
HIST_DIR = os.path.join(BASE_OUT, "perm_hists")
CORR_DIR = os.path.join(BASE_OUT, "perm_correlation")
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)

# =========================
results = []
global_data = {}  # store z-scored arrays by PCA component
# =========================

visual_files = glob.glob(os.path.join(VISCHANGE_DIR, "*_visualchange.csv"))

for vis_file in visual_files:
    base = os.path.basename(vis_file)
    video_id = base.split("_")[0]

    # --- map naming ---
    if video_id == "12":
        video_id = "12 years_6m.mp4"
    elif video_id == "500Days":
        video_id = "Movie Task-summer"
    elif video_id == "backToFuture":
        video_id = "back to the future_6m.mp4"
    elif video_id == "c4":
        video_id = "Movie Task-c4"
    elif video_id == "lms":
        video_id = "Movie Task-lms"
    elif video_id == "prestige":
        video_id = "the_prestige_6m.mp4"
    elif video_id == "pulpFiction":
        video_id = "pulp_fiction_6m.mp4"
    elif video_id == "shawshank":
        video_id = "shawshank clip_6m.mp4"

    mdes_file = os.path.join(MDES_DIR, f"{video_id}_PCA_timecourse.csv")
    if not os.path.exists(mdes_file):
        print(f"⚠️ Skipping {video_id}: mDES file not found")
        continue

    print(f"\n=== Processing {video_id} ===")

    # --- Make subfolders per video ---
    vid_hist_dir = os.path.join(HIST_DIR, video_id.replace(".mp4", ""))
    vid_corr_dir = os.path.join(CORR_DIR, video_id.replace(".mp4", ""))
    os.makedirs(vid_hist_dir, exist_ok=True)
    os.makedirs(vid_corr_dir, exist_ok=True)

    # --- Load visual change data ---
    vis = pd.read_csv(vis_file)
    vis = vis[vis["Time(s)"] >= IGNORE_FIRST_SEC]
    vis_time = pd.to_numeric(vis["Time(s)"], errors="coerce").values
    vis_change = pd.to_numeric(vis["VisualChange"], errors="coerce").values

    # ensure strictly increasing time for interpolation
    if len(vis_time):
        order = np.argsort(vis_time)
        vis_time = vis_time[order]
        vis_change = vis_change[order]

    # --- Load mDES data ---
    mdes = pd.read_csv(mdes_file)
    mdes = mdes[mdes["Run_time"] >= IGNORE_FIRST_SEC]
    mdes_time = pd.to_numeric(mdes["Run_time"], errors="coerce").values
    # don't pick PCA cols yet; we only need times to set the overlap
    if len(mdes_time):
        order_m = np.argsort(mdes_time)
        mdes_time = mdes_time[order_m]

    mdes_time = pd.to_numeric(mdes["Run_time"], errors="coerce").values
    pca_cols = [c for c in mdes.columns if c.lower().startswith("mean")]
    if not pca_cols:
        print(f"No PCA columns found in {mdes_file}")
        continue

    # --- Interpolate to 1 Hz only over the overlap between VIS and MDES ---
    # (We’ll use mdes_time below; we have it already)
    t_start = int(max(IGNORE_FIRST_SEC,
                    np.nanmin(vis_time) if len(vis_time) else IGNORE_FIRST_SEC,
                    np.nanmin(mdes_time) if len(mdes_time) else IGNORE_FIRST_SEC))

    t_end = int(min(VIDEO_DURATION,
                    np.nanmax(vis_time) if len(vis_time) else IGNORE_FIRST_SEC,
                    np.nanmax(mdes_time) if len(mdes_time) else IGNORE_FIRST_SEC))

    # if no overlap, skip
    if t_end <= t_start + 1:
        print(f"⚠️ Skipping {video_id}: no overlapping time after trimming.")
        continue

    t_full = np.arange(t_start, t_end + 1, 1)

    # interpolate both signals only within the overlap (no extrapolation/plateaus)
    vis_interp_full  = np.interp(t_full,  vis_time,  vis_change)
    # mdes per-component will be interpolated later to the same t_full/t_vis grid

    # --- Bin visual change on the overlap ---
    if BIN_SIZE <= 1:
        vis_binned = vis_interp_full.astype(float)
        t_vis = t_full.astype(float)
    else:
        n_bins = int(np.floor(len(t_full) / BIN_SIZE))
        # truncate to full bins to avoid ragged tail
        upto = n_bins * BIN_SIZE
        vis_binned = vis_interp_full[:upto].reshape(n_bins, BIN_SIZE).mean(axis=1)
        # center time of each bin
        t_vis = (t_full[:upto].reshape(n_bins, BIN_SIZE).mean(axis=1)).astype(float)


    for comp in pca_cols:
        mdes_vals = pd.to_numeric(mdes[comp], errors="coerce").values
        if len(mdes_time):
            mdes_vals = mdes_vals[order_m]  # keep aligned with sorted mdes_time

        # interpolate MDES to the same (binned) grid
        # --- Interpolate MDES to the same (binned) grid ---
        mdes_interp = np.interp(t_vis, mdes_time, mdes_vals)

        # --- Ensure equal length for vis and mdes after interpolation ---
        min_len = min(len(vis_binned), len(mdes_interp))
        vis_binned = vis_binned[:min_len]
        mdes_interp = mdes_interp[:min_len]
        t_vis = t_vis[:min_len]

        # --- Drop any bins that became non-finite on either side ---
        mask_valid = np.isfinite(vis_binned) & np.isfinite(mdes_interp)
        vis_b = vis_binned[mask_valid]
        mdes_b = mdes_interp[mask_valid]
        t_vis = t_vis[mask_valid]

        if len(vis_b) < 3:
            print(f"⚠️ Skipping {video_id} {comp}: too few valid samples.")
            continue



        # epsilon-safe z-scores (avoid inf if std==0)
        def zscore_safe(x, eps=1e-8):
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            return (x - mu) / (sd if sd > eps else 1.0)

        vis_z  = zscore_safe(vis_b)
        mdes_z = zscore_safe(mdes_b)

        # final non-finite guard (paranoia)
        mask = np.isfinite(vis_z) & np.isfinite(mdes_z)
        vis_z, mdes_z = vis_z[mask], mdes_z[mask]
        if len(vis_z) < 3:
            print(f"⚠️ Skipping {video_id} {comp}: not enough finite points after zscore.")
            continue
                
        # --- Store globally for later (used for global correlation) ---
        global_data.setdefault(comp, {"vis": [], "mdes": []})
        global_data[comp]["vis"].append(vis_z)
        global_data[comp]["mdes"].append(mdes_z)

        # --- observed correlation
        r_obs, _ = pearsonr(vis_z, mdes_z)
        print(f"→ {comp} | r = {r_obs:.3f}")

        # --- permutation (two-tailed)
        rng = np.random.default_rng()
        r_null = []
        for _ in range(N_PERM):
            shift = rng.integers(1, len(vis_z))
            r, _ = pearsonr(np.roll(vis_z, shift), mdes_z)
            r_null.append(r)
        r_null = np.array(r_null)
        p_perm = (np.sum(np.abs(r_null) >= abs(r_obs)) + 1) / (N_PERM + 1)
        


        results.append({
            "video": video_id,
            "component": comp,
            "r_obs": r_obs,
            "p_perm": p_perm
        })

        # --- Null histogram ---
        plt.figure(figsize=(8,5))
        plt.hist(r_null, bins=40, color="gray", edgecolor="black")
        plt.axvline(r_obs, color="red", linestyle="--", label=f"r={r_obs:.3f}")
        plt.title(f"{video_id} – {comp}\nPermutation p={p_perm:.4f}")
        plt.xlabel("Correlation (r)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vid_hist_dir, f"{comp}_perm_hist.png"), dpi=300)
        plt.close()

        # --- Timecourse plot ---
        plt.figure(figsize=(12,5))
        plt.plot(t_vis, vis_z, label="Visual change", color="steelblue", alpha=0.8)
        plt.plot(t_vis, mdes_z, label=f"{comp}", color="darkorange", alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Z-scored value")
        plt.title(f"{video_id}: Visual Change × {comp}\n r={r_obs:.2f}, p_perm={p_perm:.3g}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vid_corr_dir, f"{comp}_timecourse.png"), dpi=300)
        plt.close()




# ==========================================================
# 🔁 GLOBAL CORRELATION ACROSS ALL VIDEOS (gap-free concatenation)
# ==========================================================
global_results = []
global_corr_dir = os.path.join(CORR_DIR, "GLOBAL")
global_hist_dir = os.path.join(HIST_DIR, "GLOBAL")
os.makedirs(global_corr_dir, exist_ok=True)
os.makedirs(global_hist_dir, exist_ok=True)

for comp, data in global_data.items():
    # concatenate real segments back-to-back, no NaNs or padding
    vis_segments = [v[np.isfinite(v)] for v in data["vis"] if len(v)]
    mdes_segments = [m[np.isfinite(m)] for m in data["mdes"] if len(m)]
    vis_all = np.concatenate(vis_segments)
    mdes_all = np.concatenate(mdes_segments)
    t_all = np.arange(len(vis_all))  # purely sequential (no blank time)

    # z-score across the full concatenated arrays
    vis_z = (vis_all - np.mean(vis_all)) / np.std(vis_all)
    mdes_z = (mdes_all - np.mean(mdes_all)) / np.std(mdes_all)

    # --- Observed correlation
    r_obs, _ = pearsonr(vis_z, mdes_z)
    print(f"\n🌎 Global {comp}: Observed correlation r = {r_obs:.3f}")

    # --- Permutation test (circular shift, two-tailed)
    rng = np.random.default_rng()
    r_null = []
    for _ in range(N_PERM):
        shift = rng.integers(1, len(vis_z))
        r, _ = pearsonr(np.roll(vis_z, shift), mdes_z)
        r_null.append(r)
    r_null = np.array(r_null)
    p_perm = (np.sum(np.abs(r_null) >= abs(r_obs)) + 1) / (N_PERM + 1)
    print(f"Permutation p-value = {p_perm:.4f} (N={N_PERM})")

    # --- Plot 1: null distribution
    plt.figure(figsize=(8,5))
    plt.hist(r_null, bins=40, alpha=0.7, color="gray", edgecolor="black")
    plt.axvline(r_obs, color="red", linestyle="--", linewidth=2,
                label=f"Observed r={r_obs:.3f}")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Frequency")
    plt.title(f"Global {comp} Permutation Test\nObserved r={r_obs:.3f}, p={p_perm:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(global_hist_dir, f"GLOBAL_{comp}_perm_hist.png"), dpi=300)
    plt.close()

    # --- Plot 2: overlapping timecourses (gap-free)
    plt.figure(figsize=(12,5))
    plt.plot(t_all, vis_z, label="Visual change", alpha=0.8)
    plt.plot(t_all, mdes_z, label=f"{comp}", alpha=0.8)
    plt.xlabel("Concatenated sample index (no gaps)")
    plt.ylabel("Z-scored value")
    plt.title(f"Global Visual Change × {comp}\nObserved r={r_obs:.2f}, p_perm={p_perm:.3g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(global_corr_dir, f"GLOBAL_{comp}_timecourse.png"), dpi=300)
    plt.close()

    global_results.append({
        "component": comp,
        "r_obs": r_obs,
        "p_perm": p_perm,
        "N_perm": N_PERM
    })


# =========================
# Save results
# =========================
pd.DataFrame(results).to_csv(os.path.join(BASE_OUT, "permutation_summary.csv"), index=False)
pd.DataFrame(global_results).to_csv(os.path.join(BASE_OUT, "global_permutation_summary.csv"), index=False)
print(f"\n✅ Saved per-video and global permutation results to {BASE_OUT}")
