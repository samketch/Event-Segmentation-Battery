# ==========================================================
# GLOBAL Gradient-space movement analysis (across all videos)
# ==========================================================
# Author: Sam Ketcheson (global-only version)
# ----------------------------------------------------------
# This script:
#   • Loads all gradient data (combined file)
#   • Resamples gradients to 1 Hz
#   • Uses consensus boundaries for each video
#   • Pools all cross vs within boundary windows across videos
#   • Runs global t-test and saves one CSV + one plot
# ==========================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
GRADIENT_FILE = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs.csv"
BOUNDARY_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
OUTPUT_DIR    = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\State Space"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_FIRST_SEC = 5
WINDOW = 20  # ± seconds around each boundary

NAME_MAP = {
    "pulp_fiction_6m":      "pulpFiction",
    "the_prestige_6m":      "prestige",
    "12 years_6m":          "12_years",
    "back to the future_6m":"backToFuture",
    "shawshank clip_6m":    "shawshank",
    "Movie Task-lms":       "lms",
    "Movie Task-c4":        "c4",
    "Movie Task-summer":    "500Days",
}

# ------------------ LOAD DATA ------------------
df = pd.read_csv(GRADIENT_FILE)
df = df.rename(columns=lambda x: x.strip())
df["Task_name"] = df["Task_name"].str.replace(".mp4", "", regex=False)

grad_cols = [c for c in df.columns if c.startswith("gradient_")]
cons = pd.read_csv(BOUNDARY_FILE)
cons = cons.rename(columns=lambda x: x.strip())
cons["VideoName"] = cons["VideoName"].str.replace(".mp4", "", regex=False)

videos = df["Task_name"].unique()
print(f"✅ Found {len(videos)} videos in dataset")

# ------------------ HELPERS ------------------
def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def compute_speed(G):
    diffs = np.diff(G, axis=0)
    spd = np.sqrt((diffs ** 2).sum(axis=1))
    spd = np.concatenate([[spd[0]], spd])
    return zscore(spd)

def resample_to_1hz(times_s, mat):
    times_s = np.asarray(times_s, float)
    order = np.argsort(times_s)
    times_s, mat = times_s[order], mat[order]
    t_end = int(np.floor(times_s.max()))
    if t_end < 2:
        return None, None
    t1 = np.arange(0, t_end + 1, 1.0)
    M1 = np.empty((t1.size, mat.shape[1]))
    for j in range(mat.shape[1]):
        M1[:, j] = np.interp(t1, times_s, mat[:, j])
    return t1, M1

# ------------------ MAIN ------------------
all_cross, all_within = [], []

for vid_key in videos:
    vid_df = df[df["Task_name"] == vid_key].copy()
    if vid_df.empty:
        continue

    t_raw = vid_df["Run_time"].astype(float).values
    G_raw = vid_df[grad_cols].to_numpy(float)
    t1, G1 = resample_to_1hz(t_raw, G_raw)
    if t1 is None: 
        continue
    speed_z = compute_speed(G1)

    cons_name = NAME_MAP.get(vid_key, vid_key)
    b_times = cons.loc[cons["VideoName"] == cons_name, "ConsensusTime(s)"].values.astype(float)
    b_times = b_times[(b_times > IGNORE_FIRST_SEC) & (b_times < (t1.max() - WINDOW))]
    if len(b_times) == 0:
        continue

    mids = [(b_times[i] + b_times[i + 1]) / 2.0 for i in range(len(b_times) - 1)]

    for b in b_times:
        mask = (t1 >= b - WINDOW) & (t1 <= b + WINDOW)
        seg = speed_z[mask]
        if seg.size > 0:
            all_cross.append(seg)

    for m in mids:
        mask = (t1 >= m - WINDOW) & (t1 <= m + WINDOW)
        seg = speed_z[mask]
        if seg.size > 0:
            all_within.append(seg)

# ------------------ GLOBAL STATS ------------------
if len(all_cross) == 0 or len(all_within) == 0:
    print("⚠️ No valid segments accumulated across videos.")
else:
    cross_vals = np.concatenate(all_cross)
    within_vals = np.concatenate(all_within)
    tval, pval = ttest_ind(cross_vals, within_vals, equal_var=False)
    diff = np.mean(cross_vals) - np.mean(within_vals)

    global_df = pd.DataFrame({
        "Mean_Cross": [np.mean(cross_vals)],
        "Mean_Within": [np.mean(within_vals)],
        "Delta": [diff],
        "t": [tval],
        "p": [pval],
        "N_cross": [len(cross_vals)],
        "N_within": [len(within_vals)]
    })
    global_csv = os.path.join(OUTPUT_DIR, "Global_GradientSpace_summary.csv")
    global_df.to_csv(global_csv, index=False)

    print(f"\n🌎 GLOBAL Δcross−within = {diff:.4f}, t={tval:.3f}, p={pval:.3g}")
    print(f"✅ Global summary CSV saved to: {global_csv}")

    # ---------- PLOT ----------
    def avg_trace(segments, window_s=WINDOW):
        target_len = int(2 * window_s) + 1
        x_new = np.linspace(-window_s, window_s, target_len)
        resampled = []
        for s in segments:
            x_old = np.linspace(-1, 1, s.size)
            resampled.append(np.interp(np.linspace(-1, 1, target_len), x_old, s))
        arr = np.vstack(resampled)
        return x_new, arr.mean(axis=0), arr.std(axis=0) / np.sqrt(arr.shape[0])

    x, m1, s1 = avg_trace(all_cross, WINDOW)
    _, m2, s2 = avg_trace(all_within, WINDOW)

    plt.figure(figsize=(7,4))
    plt.fill_between(x, m1 - s1, m1 + s1, alpha=0.3, label="Across-boundary")
    plt.fill_between(x, m2 - s2, m2 + s2, alpha=0.3, label="Within-event")
    plt.plot(x, m1, label="Across-boundary")
    plt.plot(x, m2, label="Within-event")
    plt.axvline(0, color="k", ls="--")
    plt.xlabel("Time (s, centered)")
    plt.ylabel("Gradient-space movement (z)")
    plt.title(f"Global brain-state speed near boundaries\nΔ={diff:.3f}, p={pval:.3g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Global_GradientSpace_CrossVsWithin.png"), dpi=300)
    plt.close()
    print("✅ Global figure saved.\n")
