import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.signal import find_peaks
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ==========================================================
# CONFIG — match KDE script exactly
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\PCAvKDE_Overlay\top_10%\plot_average"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERCENTILE_CUTOFF = 90        # consensus threshold
SMOOTHING_WINDOW = 1          # keep as in your script
IGNORE_FIRST_SEC = 60         # ignore first minute for stats/peaks
WINDOW_MIN, WINDOW_MAX = 5, 30  # safety clamps (in seconds)
USE_MEDIAN_FOR_WINDOW = True     # set False to use mean

# --- same mm:ss formatter
def seconds_to_mmss(x, pos):
    m, s = divmod(int(x), 60)
    return f"{m:02d}:{s:02d}"
time_formatter = FuncFormatter(seconds_to_mmss)

def smooth(x, w):
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

# ==========================================================
# Helper: load PCA₁ + KDE for a video, aligned to same 1 Hz grid
# ==========================================================
def load_and_align(video):
    pca_file = os.path.join(MDES_DIR, f"{video}_PCA_timecourse.csv")
    kde_file = os.path.join(KDE_DIR, f"{video.replace('.mp4','')}_kde_timeseries.csv")
    if not (os.path.exists(pca_file) and os.path.exists(kde_file)):
        print(f"⚠️ Missing file for {video}")
        return None

    pca = pd.read_csv(pca_file)
    kde = pd.read_csv(kde_file)
    pca["Run_time"] = pd.to_numeric(pca["Run_time"], errors="coerce")
    kde = kde.rename(columns={kde.columns[0]: "Time", kde.columns[1]: "BoundaryStrength"})
    kde["Time"] = pd.to_numeric(kde["Time"], errors="coerce")

    # match per-video duration logic
    if "500Days" in video or "lms" in video or "c4" in video:
        dur = 11 * 60
    else:
        dur = 6 * 60
    t = np.arange(0, dur + 1)

    pca_interp = np.interp(t, pca["Run_time"], pca["Mean_PCA_1"])
    kde_interp = np.interp(t, kde["Time"], kde["BoundaryStrength"])

    # z-score + optional smoothing
    pca_z = (pca_interp - np.mean(pca_interp)) / (np.std(pca_interp) + 1e-12)
    kde_z = (kde_interp - np.mean(kde_interp)) / (np.std(kde_interp) + 1e-12)

    return t, smooth(pca_z, SMOOTHING_WINDOW), smooth(kde_z, SMOOTHING_WINDOW)

# ==========================================================
# MAIN LOOP
# ==========================================================
rows = []
videos = [f.replace("_PCA_timecourse.csv", "") for f in os.listdir(MDES_DIR)
          if f.endswith("_PCA_timecourse.csv")]

for vid in videos:
    loaded = load_and_align(vid)
    if loaded is None:
        continue
    t, pca_series, kde_series = loaded

    # restrict stats to after first minute
    valid_mask = t >= IGNORE_FIRST_SEC
    t_valid = t[valid_mask]
    pca_valid = pca_series[valid_mask]
    kde_valid = kde_series[valid_mask]

    if len(t_valid) < 3:
        print(f"⚠️ Too little valid data for {vid}")
        continue

    # --- 90th-percentile KDE threshold and consensus peaks
    cutoff = np.percentile(kde_valid, PERCENTILE_CUTOFF)
    peaks, _ = find_peaks(kde_valid, height=cutoff)
    peak_times = t_valid[peaks]

    # --- compute inter-peak intervals (event lengths)
    if len(peak_times) >= 2:
        intervals = np.diff(peak_times)
        median_len = float(np.median(intervals))
        mean_len = float(np.mean(intervals))
        typical_len = median_len if USE_MEDIAN_FOR_WINDOW else mean_len
        window = int(np.clip(round(typical_len / 2.0), WINDOW_MIN, WINDOW_MAX))
    else:
        intervals = np.array([])
        median_len = np.nan
        mean_len = np.nan
        window = 10  # fallback to 10s if not enough peaks

    # --- effect size using top10% high vs low (on trimmed region only)
    high_mask = kde_valid >= cutoff
    high_pca = pca_valid[high_mask]
    low_pca  = pca_valid[~high_mask]

    if len(high_pca) >= 3 and len(low_pca) >= 3:
        tval, pval = ttest_ind(high_pca, low_pca, equal_var=False)
        diff = float(np.mean(high_pca) - np.mean(low_pca))
    else:
        tval, pval, diff = np.nan, np.nan, np.nan

    # --- Event-triggered average using dynamic window
    segments = []
    for b in peak_times:
        start = b - window
        end   = b + window
        if start < t_valid[0] or end > t_valid[-1]:
            continue
        seg_mask = (t >= start) & (t <= end)
        seg = pca_series[seg_mask]
        # ensure exact length 2*window + 1 seconds
        if len(seg) == 2 * window + 1:
            segments.append(seg)
    segments = np.array(segments)

    # --- Plot per-video event-triggered average
    if segments.shape[0] >= 3:
        mean_trace = segments.mean(axis=0)
        sem_trace  = segments.std(axis=0) / np.sqrt(segments.shape[0])
        t_axis = np.arange(-window, window + 1)

        plt.figure(figsize=(6, 4))
        plt.fill_between(t_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                         color="skyblue", alpha=0.4)
        plt.plot(t_axis, mean_trace, color="tab:blue", lw=2)
        plt.axvline(0, color="tab:red", ls="--", lw=1.5, label="Consensus boundary")
        win_txt = f"window=±{window}s"
        lens_txt = f"typLen={median_len if USE_MEDIAN_FOR_WINDOW else mean_len:.1f}s " \
                   f"(med={median_len:.1f}, mean={mean_len:.1f})"
        plt.title(f"{vid}\nΔ={diff:.2f}, p={pval:.3g}, {lens_txt}")
        plt.xlabel("Time (s) from boundary")
        plt.ylabel("Z-scored PCA₁")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{vid}_PCA1_topBoundaries_dynamicWindow.png"), dpi=300)
        plt.close()

    # --- record summary row
    rows.append({
        "Video": vid,
        "Num_Peaks": int(len(peak_times)),
        "Median_EventLength_s": median_len,
        "Mean_EventLength_s": mean_len,
        "WindowUsed_s": int(window),
        "Num_Segments": int(segments.shape[0]) if segments.size else 0,
        "Delta_PCA_HighMinusLow": diff,
        "t": tval,
        "p": pval
    })

# ==========================================================
# Save summary table and bar plot
# ==========================================================
res = pd.DataFrame(rows)
res_path = os.path.join(OUTPUT_DIR, "PCA1_topBoundaries_dynamicWindow_summary.csv")
res.to_csv(res_path, index=False)

plt.figure(figsize=(7, 4))
plt.barh(res["Video"], res["Delta_PCA_HighMinusLow"], color="gray")
plt.axvline(0, color="k", ls="--")
plt.xlabel("Δ PCA₁ (High − Low Boundary Periods)")
plt.ylabel("Video")
plt.title("Change in PCA₁ during Top 10% Boundaries (dynamic window)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "PCA1_topBoundary_barplot_dynamicWindow.png"), dpi=300)
plt.close()

print("✅ Done. Outputs in:", OUTPUT_DIR)
print("   • Summary CSV:", res_path)
