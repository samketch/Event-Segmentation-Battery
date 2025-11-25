import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.signal import find_peaks

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\PCAvKDE_Overlay\cross_vs_within"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERCENTILE_CUTOFF = 90
SMOOTHING_WINDOW = 1
IGNORE_FIRST_SEC = 60
WINDOW_MIN, WINDOW_MAX = 5, 30
USE_MEDIAN_FOR_WINDOW = True

def smooth(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode="same")

# ==========================================================
# Helper: load all PCA components + KDE for one video
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

    max_time = min(pca["Run_time"].max(), kde["Time"].max())
    t = np.arange(0, max_time + 1)
    kde_interp = np.interp(t, kde["Time"], kde["BoundaryStrength"])
    kde_z = (kde_interp - np.mean(kde_interp)) / (np.std(kde_interp) + 1e-12)

    # Interpolate all PCA components
    pca_cols = [c for c in pca.columns if c.startswith("Mean_PCA_")]
    pca_interp = {}
    for c in pca_cols:
        vals = np.interp(t, pca["Run_time"], pca[c])
        vals_z = (vals - np.mean(vals)) / (np.std(vals) + 1e-12)
        pca_interp[c] = smooth(vals_z, SMOOTHING_WINDOW)

    df = pd.DataFrame({"Video": video, "Time": t, "KDE": smooth(kde_z, SMOOTHING_WINDOW)})
    for c in pca_cols:
        df[c] = pca_interp[c]
    return df


# ==========================================================
# MAIN LOOP OVER PCA COMPONENTS
# ==========================================================
videos = [f.replace("_PCA_timecourse.csv", "") for f in os.listdir(MDES_DIR)
          if f.endswith("_PCA_timecourse.csv")]

example = pd.read_csv(os.path.join(MDES_DIR, os.listdir(MDES_DIR)[0]))
pca_cols = [c for c in example.columns if c.startswith("Mean_PCA_")]
print(f"✅ Detected {len(pca_cols)} PCA components: {', '.join(pca_cols)}")

summary_rows_global = []

# ==========================================================
# MAIN LOOP
# ==========================================================
for pca_col in pca_cols:
    print(f"\n🎬 Running cross-vs-within boundary analysis for {pca_col}")

    cross_segments_all, within_segments_all, all_window_sizes = [], [], []
    per_video_results = []

    for vid in videos:
        df = load_and_align(vid)
        if df is None:
            continue
        df = df[df["Time"] >= IGNORE_FIRST_SEC].reset_index(drop=True)
        if len(df) < 3:
            continue

        kde_valid = df["KDE"].values
        pca_valid = df[pca_col].values
        t_valid   = df["Time"].values

        cutoff = np.percentile(kde_valid, PERCENTILE_CUTOFF)
        peaks, _ = find_peaks(kde_valid, height=cutoff)
        peak_times = t_valid[peaks]

        if len(peak_times) == 0:
            print(f"   ⚠️ No peaks detected in {vid}")
            continue

        # window size based on median boundary interval
        if len(peak_times) >= 2:
            intervals = np.diff(peak_times)
            median_len = float(np.median(intervals))
            mean_len   = float(np.mean(intervals))
            typical_len = median_len if USE_MEDIAN_FOR_WINDOW else mean_len
            window = int(np.clip(round(typical_len / 2.0), WINDOW_MIN, WINDOW_MAX))
        else:
            window = 10
        all_window_sizes.append(window)

        # ---------- build cross-boundary segments
        cross_segments = []
        for b in peak_times:
            start, end = b - window, b + window
            seg_mask = (df["Time"] >= start) & (df["Time"] <= end)
            seg = df.loc[seg_mask, pca_col].values
            if len(seg) >= 2:
                cross_segments.append(seg)

        # ---------- build within-event segments (midpoints between boundaries)
        within_segments = []
        midpoints = []
        if len(peak_times) >= 2:
            for i in range(len(peak_times) - 1):
                mid = (peak_times[i] + peak_times[i+1]) / 2
                midpoints.append(mid)
            for m in midpoints:
                start, end = m - window, m + window
                seg_mask = (df["Time"] >= start) & (df["Time"] <= end)
                seg = df.loc[seg_mask, pca_col].values
                if len(seg) >= 2:
                    within_segments.append(seg)

        # ---------- per-video stats and plot
        if len(cross_segments) >= 3 and len(within_segments) >= 3:
            cross_vals = np.concatenate(cross_segments)
            within_vals = np.concatenate(within_segments)
            tval, pval = ttest_ind(cross_vals, within_vals, equal_var=False)
            diff = float(np.mean(cross_vals) - np.mean(within_vals))
        else:
            tval, pval, diff = np.nan, np.nan, np.nan

        # ---------- plot per video
        video_outdir = os.path.join(OUTPUT_DIR, vid)
        os.makedirs(video_outdir, exist_ok=True)

        def avg_trace(segments, w):
            if len(segments) == 0:
                return None, None, None
            target_len = 2 * w + 1
            resampled = []
            for seg in segments:
                x_old = np.linspace(-1, 1, len(seg))
                x_new = np.linspace(-1, 1, target_len)
                resampled.append(np.interp(x_new, x_old, seg))
            arr = np.vstack(resampled)
            return arr.mean(axis=0), arr.std(axis=0)/np.sqrt(arr.shape[0]), np.linspace(-w, w, target_len)

        cross_mean, cross_sem, t_axis = avg_trace(cross_segments, window)
        within_mean, within_sem, _ = avg_trace(within_segments, window)

        if cross_mean is not None and within_mean is not None:
            plt.figure(figsize=(7,4))
            plt.fill_between(t_axis, cross_mean-cross_sem, cross_mean+cross_sem, color="tab:blue", alpha=0.3, label="Cross-boundary")
            plt.fill_between(t_axis, within_mean-within_sem, within_mean+within_sem, color="tab:orange", alpha=0.3, label="Within-event")
            plt.plot(t_axis, cross_mean, color="tab:blue")
            plt.plot(t_axis, within_mean, color="tab:orange")
            plt.axvline(0, color="k", ls="--", lw=1)
            plt.title(f"{vid} — {pca_col}\nΔ={diff:.2f}, p={pval:.3g}")
            plt.xlabel("Time (s)")
            plt.ylabel(f"Z-scored {pca_col}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(video_outdir, f"{pca_col}_CrossVsWithin.png"), dpi=300)
            plt.close()

        per_video_results.append({
            "Video": vid,
            "Component": pca_col,
            "Delta_CrossMinusWithin": diff,
            "t": tval,
            "p": pval,
            "Num_CrossSegs": len(cross_segments),
            "Num_WithinSegs": len(within_segments),
            "WindowUsed_s": window
        })

        # accumulate globally
        cross_segments_all.extend(cross_segments)
        within_segments_all.extend(within_segments)

    # ---------- GLOBAL ANALYSIS
    if len(cross_segments_all) < 3 or len(within_segments_all) < 3:
        print(f"⚠️ Not enough data for global {pca_col}")
        continue

    # resample all segments to same length
    median_window = int(np.median(all_window_sizes))
    target_len = 2 * median_window + 1

    def resample_all(segments):
        resampled = []
        for seg in segments:
            x_old = np.linspace(-1, 1, len(seg))
            x_new = np.linspace(-1, 1, target_len)
            resampled.append(np.interp(x_new, x_old, seg))
        return np.vstack(resampled)

    cross_arr = resample_all(cross_segments_all)
    within_arr = resample_all(within_segments_all)

    cross_vals = cross_arr.flatten()
    within_vals = within_arr.flatten()
    tval, pval = ttest_ind(cross_vals, within_vals, equal_var=False)
    diff = float(np.mean(cross_vals) - np.mean(within_vals))

    cross_mean = cross_arr.mean(axis=0)
    cross_sem  = cross_arr.std(axis=0) / np.sqrt(cross_arr.shape[0])
    within_mean = within_arr.mean(axis=0)
    within_sem  = within_arr.std(axis=0) / np.sqrt(within_arr.shape[0])
    t_axis = np.linspace(-median_window, median_window, target_len)

    plt.figure(figsize=(7,4))
    plt.fill_between(t_axis, cross_mean-cross_sem, cross_mean+cross_sem, color="tab:blue", alpha=0.3, label="Cross-boundary")
    plt.fill_between(t_axis, within_mean-within_sem, within_mean+within_sem, color="tab:orange", alpha=0.3, label="Within-event")
    plt.plot(t_axis, cross_mean, color="tab:blue")
    plt.plot(t_axis, within_mean, color="tab:orange")
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.title(f"{pca_col} (GLOBAL)\nΔ={diff:.2f}, p={pval:.3g}, window≈±{median_window}s")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Z-scored {pca_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{pca_col}_GLOBAL_CrossVsWithin.png"), dpi=300)
    plt.close()

    summary_rows_global.append({
        "Component": pca_col,
        "Delta_CrossMinusWithin": diff,
        "t": tval,
        "p": pval,
        "Num_CrossSegs": len(cross_arr),
        "Num_WithinSegs": len(within_arr),
        "WindowUsed_s": median_window,
        "Num_Videos": len(videos)
    })

# ==========================================================
# SAVE GLOBAL SUMMARY
# ==========================================================
summary_df = pd.DataFrame(summary_rows_global)
summary_path = os.path.join(OUTPUT_DIR, "GLOBAL_CrossVsWithin_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n✅ Finished all PCA components.")
print("   • Global summary:", summary_path)
print("   • Per-video results saved in each video folder.")
