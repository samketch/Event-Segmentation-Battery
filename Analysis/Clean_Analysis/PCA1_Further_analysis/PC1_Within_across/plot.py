# =====================================================================
# Plot_tTest_PCA1_WithinVsAcross.py
# =====================================================================
# Plots Mean_PCA_1 for within- vs across-event timepoints and
# annotates the figure with Welch t-test results.
# =====================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

# ---------------- CONFIG ----------------
DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
OUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\PCA1_Further_analysis\PC1_Within_across"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COLUMN = "Mean_PCA_1"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# Split by Within vs Across
within = df.loc[df["Within_or_Across"] == 1, TARGET_COLUMN].dropna()
across = df.loc[df["Within_or_Across"] == 0, TARGET_COLUMN].dropna()

# Welch t-test
tval, pval = ttest_ind(within, across, equal_var=False)
diff = np.mean(across) - np.mean(within)

print(f"PCA 1: t={tval:.3f}, p={pval:.4f}, diff={diff:.3f}")
print(f"Within N={len(within)}, Across N={len(across)}")

# ---------------- PLOT ----------------
means = [np.mean(within), np.mean(across)]
sems = [np.std(within, ddof=1)/np.sqrt(len(within)),
        np.std(across, ddof=1)/np.sqrt(len(across))]

labels = ["Within-event", "Across-event"]
colors = ["tab:orange", "tab:blue"]

plt.figure(figsize=(5,5))
plt.bar(labels, means, yerr=sems, color=colors, capsize=5, alpha=0.8)
plt.ylabel("Mean PCA 1 (z)")
plt.title(f"PCA 1 within vs across events\n t={tval:.2f}, p={pval:.3g}")

# Add jittered scatter of all points for visibility
x1 = np.random.normal(0, 0.04, size=len(within))
x2 = np.random.normal(1, 0.04, size=len(across))
plt.scatter(x1, within, color="tab:orange", alpha=0.4, s=10)
plt.scatter(x2, across, color="tab:blue", alpha=0.4, s=10)

plt.tight_layout()

out_path = os.path.join(OUT_DIR, "PCA1_WithinVsAcross_tTest.png")
plt.savefig(out_path, dpi=300)
plt.show()

print(f"✅ Plot saved → {out_path}")
# =====================================================================
# Peri_ttest_PCA1_cross_vs_within.py
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind

# ---------------- CONFIG ----------------
DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
BOUNDARY_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COLUMN = "Mean_PCA_1"
WINDOW_SEC = 30
FS = 1.0  # interpolate to 1 Hz
MIN_COVERAGE = 0.75  # keep partial windows if at least 75 percent length

# ---------------- LOAD ----------------
df = pd.read_csv(DATA_PATH).sort_values(["VideoName","Run_time"]).reset_index(drop=True)
bdf = pd.read_csv(BOUNDARY_FILE)

# detect columns in boundary csv
name_col = [c for c in bdf.columns if any(k in c.lower() for k in ["video","task","movie","file"])][0]
time_col = [c for c in bdf.columns if any(k in c.lower() for k in ["time","sec","boundary"])][0]

# clean names for matching
bdf["Video_clean"] = (
    bdf[name_col].astype(str)
    .str.lower()
    .str.replace(".mp4","", regex=False)
    .str.strip()
)

def interp_to_fs(t, v, fs=1.0):
    t0 = np.ceil(np.nanmin(t))
    t1 = np.floor(np.nanmax(t))
    if t1 <= t0:
        return None, None
    t_new = np.arange(t0, t1 + 1/fs, 1/fs)
    v_new = np.interp(t_new, t, v)
    return t_new, v_new

def collect_windows(tu, vu, anchors, half_win, min_coverage=0.75):
    segs = []
    L = 2*half_win + 1
    for a in anchors:
        start, end = a - half_win, a + half_win
        m = (tu >= start) & (tu <= end)
        seg = vu[m]
        if len(seg) >= int(min_coverage * L):
            segs.append(seg)
    return segs

def resample_segment(seg, target_len):
    x_old = np.linspace(-1, 1, len(seg))
    x_new = np.linspace(-1, 1, target_len)
    return np.interp(x_new, x_old, seg)

cross_segments_all, within_segments_all = [], []

for vid, sub in df.groupby("VideoName"):
    stem = Path(vid).stem.lower().strip()

    # exact match first
    rows = bdf[bdf["Video_clean"] == stem]
    if rows.empty:
        # fallback to substring
        rows = bdf[bdf["Video_clean"].str.contains(stem, case=False)]
    if rows.empty:
        print(f"Skip, no boundaries for: {vid}")
        continue

    btimes = rows[time_col].dropna().astype(float).values
    if len(btimes) < 1:
        print(f"Skip, no times for: {vid}")
        continue

    midpoints = (btimes[:-1] + btimes[1:]) / 2.0
    if len(midpoints) < 1:
        continue

    t = sub["Run_time"].astype(float).values
    v = sub[TARGET_COLUMN].astype(float).values
    tu, vu = interp_to_fs(t, v, FS)
    if tu is None:
        continue

    half = int(WINDOW_SEC)
    cross_segments = collect_windows(tu, vu, btimes, half, MIN_COVERAGE)
    within_segments = collect_windows(tu, vu, midpoints, half, MIN_COVERAGE)

    if len(cross_segments) > 0:
        cross_segments_all.extend(cross_segments)
    if len(within_segments) > 0:
        within_segments_all.extend(within_segments)

if len(cross_segments_all) == 0 or len(within_segments_all) == 0:
    raise RuntimeError("No segments collected. Check coverage or WINDOW_SEC.")

# resample all segments to the same length
all_lengths = [len(s) for s in (cross_segments_all + within_segments_all)]
target_len = int(np.median(all_lengths))
cross_arr = np.vstack([resample_segment(s, target_len) for s in cross_segments_all])
within_arr = np.vstack([resample_segment(s, target_len) for s in within_segments_all])

# compute peri curves and test
cross_mean  = cross_arr.mean(axis=0)
within_mean = within_arr.mean(axis=0)
cross_sem   = cross_arr.std(axis=0, ddof=1) / np.sqrt(cross_arr.shape[0])
within_sem  = within_arr.std(axis=0, ddof=1) / np.sqrt(within_arr.shape[0])

tval, pval = ttest_ind(cross_arr.ravel(), within_arr.ravel(), equal_var=False)
delta = float(cross_arr.mean() - within_arr.mean())

time_axis = np.linspace(-WINDOW_SEC, WINDOW_SEC, target_len)

# ---------------- PLOT ----------------
plt.figure(figsize=(8,4.5))
plt.fill_between(time_axis, cross_mean - cross_sem, cross_mean + cross_sem,
                 color="tab:blue", alpha=0.25, label="Cross-boundary")
plt.fill_between(time_axis, within_mean - within_sem, within_mean + within_sem,
                 color="tab:orange", alpha=0.25, label="Within-event")
plt.plot(time_axis, cross_mean, color="tab:blue", lw=2)
plt.plot(time_axis, within_mean, color="tab:orange", lw=2)
plt.axvline(0, color="k", ls="--", lw=1)

plt.title("PCA 1 Within Vs Across Average")
plt.xlabel("Time relative to anchor (s)")
plt.ylabel("Mean_PCA_1")
plt.legend()
plt.tight_layout()

out_fig = os.path.join(OUT_DIR, f"Peri_PCA1_cross_vs_within_window{WINDOW_SEC}.png")
plt.savefig(out_fig, dpi=300)
plt.close()



print(f"Saved: {out_fig}")
print(f"Segments used  cross={cross_arr.shape[0]}  within={within_arr.shape[0]}")
