# =====================================================================
# PeriBoundary_PCA1_with_AsymmetryTests.py
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\PCA1_Further_analysis\Peri_boundary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SEC = 30
THRESHOLD = 0.001
TARGET_COLUMN = "Mean_PCA_1"

df = pd.read_csv(DATA_PATH).sort_values(["VideoName","Run_time"]).reset_index(drop=True)

all_peri = []        # rows: VideoName, BoundaryIdx, Time_from_Boundary, Value
by_video_means = []  # for plotting per video if needed

for video, sub in df.groupby("VideoName"):
    x = sub["Run_time"].values
    y = sub[TARGET_COLUMN].values
    sampling = float(np.median(np.diff(x)))
    half_pts = int(WINDOW_SEC / sampling)

    # use discrete boundary peaks if available in your data
    # use discrete consensus boundaries instead of continuous densities
    from pathlib import Path
    boundaries = pd.read_csv(
        r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
    )
    b_times = boundaries.loc[
        boundaries["VideoName"].str.contains(Path(video).stem, case=False), "ConsensusTime(s)"
    ].values

    if len(b_times) == 0:
        continue

    peri_mats = []
    kept = 0
    for bi, b in enumerate(b_times):
        center = np.argmin(np.abs(x - b))
        start = center - half_pts
        end   = center + half_pts + 1
        if start < 0 or end > len(x):
            continue
        seg = y[start:end]
        if len(seg) == 2*half_pts + 1:
            peri_mats.append(seg)
            kept += 1
            t_axis = (np.arange(-half_pts, half_pts+1) * sampling)
            all_peri.append(pd.DataFrame({
                "VideoName": video,
                "BoundaryIdx": bi,
                "Time_from_Boundary": t_axis,
                "Value": seg
            }))

    if kept == 0:
        continue

    peri_mats = np.asarray(peri_mats)
    mean_curve = np.nanmean(peri_mats, axis=0)
    by_video_means.append(pd.DataFrame({
        "VideoName": video,
        "Time_from_Boundary": t_axis,
        "Mean_Value": mean_curve
    }))

# Combine across videos and boundaries
all_peri_df = pd.concat(all_peri, ignore_index=True)
curve_mean = all_peri_df.groupby("Time_from_Boundary")["Value"].mean()
curve_sem  = all_peri_df.groupby("Time_from_Boundary")["Value"].sem()

# Save peri-boundary mean curve
out_curve_csv = os.path.join(OUTPUT_DIR, f"PeriBoundary_{TARGET_COLUMN}.csv")
curve_mean.to_csv(out_curve_csv, header=["Mean_Value"])

# Mirror-difference test
# Pair post +t with pre -t
times = curve_mean.index.values
pre_mask  = times < 0
post_mask = times > 0

# use only lags that exist on both sides
lags = np.intersect1d(np.abs(times[pre_mask]), times[post_mask])
diff_rows = []
for lag in lags:
    pre_val  = curve_mean.loc[-lag]
    post_val = curve_mean.loc[ lag]
    diff_rows.append({"Lag": lag, "PostMinusPre": post_val - pre_val})

diff_df = pd.DataFrame(diff_rows).sort_values("Lag")
out_diff_csv = os.path.join(OUTPUT_DIR, f"PeriBoundary_{TARGET_COLUMN}_MirrorDiff.csv")
diff_df.to_csv(out_diff_csv, index=False)

# Paired t-test on mirror differences
t_stat, p_val = stats.ttest_1samp(diff_df["PostMinusPre"].values, 0.0, nan_policy="omit")

# Pre vs post averages in fixed windows for a simple causal summary
pre_window  = (times >= -20) & (times < 0)
post_window = (times > 0) & (times <= 20)
pre_mean  = curve_mean.loc[times[pre_window]].mean()
post_mean = curve_mean.loc[times[post_window]].mean()

summary = pd.DataFrame({
    "Metric": ["Pre_mean_-20to0", "Post_mean_0to20", "Paired_t_on_mirrorDiff", "Paired_p_on_mirrorDiff"],
    "Value":  [pre_mean, post_mean, float(t_stat), float(p_val)]
})
out_summary_csv = os.path.join(OUTPUT_DIR, f"PeriBoundary_{TARGET_COLUMN}_AsymmetrySummary.csv")
summary.to_csv(out_summary_csv, index=False)

# Plot
plt.figure(figsize=(7,5))
plt.plot(curve_mean.index, curve_mean.values, lw=2)
plt.fill_between(curve_mean.index,
                 curve_mean - curve_sem,
                 curve_mean + curve_sem,
                 alpha=0.3)
plt.axvline(0, color="red", linestyle="--", lw=1)
plt.title(f"Peri-boundary trajectory of {TARGET_COLUMN}")
plt.xlabel("Time from boundary (s)")
plt.ylabel(f"{TARGET_COLUMN} (z)")
plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, f"PeriBoundary_{TARGET_COLUMN}.png")
plt.savefig(out_fig, dpi=300)
plt.close()

print(f"Saved curve CSV: {out_curve_csv}")
print(f"Saved mirror-difference CSV: {out_diff_csv}")
print(f"Saved asymmetry summary CSV: {out_summary_csv}")
print(f"Figure: {out_fig}")
