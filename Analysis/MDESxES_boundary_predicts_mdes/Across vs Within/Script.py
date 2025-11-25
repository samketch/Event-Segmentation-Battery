import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\Across vs Within\LMM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERCENTILE_CUTOFF = 90
SMOOTHING_WINDOW = 1
IGNORE_FIRST_SEC = 60
WINDOW_MIN, WINDOW_MAX = 5, 30
USE_MEDIAN_FOR_WINDOW = True

# ==========================================================
# HELPERS
# ==========================================================
def smooth(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def load_and_align(video):
    pca_path = os.path.join(MDES_DIR, f"{video}_PCA_timecourse.csv")
    kde_path = os.path.join(KDE_DIR, f"{video}_kde_timeseries.csv")
    if not (os.path.exists(pca_path) and os.path.exists(kde_path)):
        print(f"⚠️ Missing file for {video}")
        return None
    pca = pd.read_csv(pca_path)
    kde = pd.read_csv(kde_path)
    pca["Run_time"] = pd.to_numeric(pca["Run_time"], errors="coerce")
    kde = kde.rename(columns={kde.columns[0]: "Time", kde.columns[1]: "BoundaryStrength"})
    kde["Time"] = pd.to_numeric(kde["Time"], errors="coerce")
    max_t = min(pca["Run_time"].max(), kde["Time"].max())
    t = np.arange(0, max_t + 1)
    kde_interp = np.interp(t, kde["Time"], kde["BoundaryStrength"])
    kde_z = (kde_interp - np.mean(kde_interp)) / (np.std(kde_interp) + 1e-12)
    pca_cols = [c for c in pca.columns if c.startswith("Mean_PCA_")]
    out = pd.DataFrame({"Video": video, "Time": t, "KDE": smooth(kde_z, SMOOTHING_WINDOW)})
    for c in pca_cols:
        vals = np.interp(t, pca["Run_time"], pca[c])
        vals_z = (vals - np.mean(vals)) / (np.std(vals) + 1e-12)
        out[c] = smooth(vals_z, SMOOTHING_WINDOW)
    return out[out["Time"] >= IGNORE_FIRST_SEC].reset_index(drop=True)

# ==========================================================
# LOAD ALL VIDEOS
# ==========================================================
videos = [f.replace("_PCA_timecourse.csv", "") for f in os.listdir(MDES_DIR)
          if f.endswith("_PCA_timecourse.csv")]
example = pd.read_csv(os.path.join(MDES_DIR, os.listdir(MDES_DIR)[0]))
pca_cols = [c for c in example.columns if c.startswith("Mean_PCA_")]
print(f"✅ Detected {len(pca_cols)} PCA components: {', '.join(pca_cols)}")

# ==========================================================
# LOOP OVER COMPONENTS
# ==========================================================
summary_rows = []
long_rows = []    # collect mean values for LMM

for pca_col in pca_cols:
    print(f"\n🎬 Running global cross-vs-within for {pca_col}")
    cross_segments_all, within_segments_all, all_window_sizes = [], [], []

    for vid in videos:
        df = load_and_align(vid)
        if df is None or len(df) < 3:
            continue
        kde_valid = df["KDE"].values
        t_valid   = df["Time"].values
        cutoff = np.percentile(kde_valid, PERCENTILE_CUTOFF)
        peaks, _ = find_peaks(kde_valid, height=cutoff)
        peak_times = t_valid[peaks]
        if len(peak_times) == 0:
            continue

        # window size based on median spacing
        if len(peak_times) >= 2:
            intervals = np.diff(peak_times)
            typical_len = np.median(intervals) if USE_MEDIAN_FOR_WINDOW else np.mean(intervals)
            window = int(np.clip(round(typical_len / 2.0), WINDOW_MIN, WINDOW_MAX))
        else:
            window = 10
        all_window_sizes.append(window)

        # cross segments
        for b in peak_times:
            seg = df.loc[(df["Time"] >= b - window) & (df["Time"] <= b + window), pca_col]
            if len(seg) > 2:
                cross_segments_all.append(seg.values)
                long_rows.append({
                    "Video": vid,
                    "BoundaryType": "Cross",
                    "Component": pca_col,
                    "MeanPCA": np.mean(seg)
                })

        # within segments
        if len(peak_times) >= 2:
            mids = [(peak_times[i] + peak_times[i+1]) / 2 for i in range(len(peak_times)-1)]
            for m in mids:
                seg = df.loc[(df["Time"] >= m - window) & (df["Time"] <= m + window), pca_col]
                if len(seg) > 2:
                    within_segments_all.append(seg.values)
                    long_rows.append({
                        "Video": vid,
                        "BoundaryType": "Within",
                        "Component": pca_col,
                        "MeanPCA": np.mean(seg)
                    })

    # skip if too few segments
    if len(cross_segments_all) < 3 or len(within_segments_all) < 3:
        print(f"⚠️ Not enough data for {pca_col}")
        continue

    # resample segments for plotting
    median_window = int(np.median(all_window_sizes))
    target_len = 2 * median_window + 1
    def resample(segments):
        out = []
        for s in segments:
            x_old = np.linspace(-1, 1, len(s))
            x_new = np.linspace(-1, 1, target_len)
            out.append(np.interp(x_new, x_old, s))
        return np.vstack(out)

    cross_arr = resample(cross_segments_all)
    within_arr = resample(within_segments_all)

    cross_vals = cross_arr.flatten()
    within_vals = within_arr.flatten()
    tval, pval = ttest_ind(cross_vals, within_vals, equal_var=False)
    diff = np.mean(cross_vals) - np.mean(within_vals)

    # plot mean traces
    cross_mean = cross_arr.mean(axis=0)
    cross_sem = cross_arr.std(axis=0) / np.sqrt(cross_arr.shape[0])
    within_mean = within_arr.mean(axis=0)
    within_sem = within_arr.std(axis=0) / np.sqrt(within_arr.shape[0])
    t_axis = np.linspace(-median_window, median_window, target_len)

    plt.figure(figsize=(7,4))
    plt.fill_between(t_axis, cross_mean-cross_sem, cross_mean+cross_sem,
                     color="tab:blue", alpha=0.3, label="Boundary-adjacent")
    plt.fill_between(t_axis, within_mean-within_sem, within_mean+within_sem,
                     color="tab:orange", alpha=0.3, label="Within-event")
    plt.plot(t_axis, cross_mean, color="tab:blue")
    plt.plot(t_axis, within_mean, color="tab:orange")
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.title(f"{pca_col} — Global Δ={diff:.2f}, p={pval:.3g}")
    plt.xlabel("Time (s, centered on boundary or midpoint)")
    plt.ylabel(f"Z-scored {pca_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{pca_col}_Global.png"), dpi=300)
    plt.close()

    summary_rows.append({
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
# SAVE GLOBAL SUMMARY + LMM
# ==========================================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "GLOBAL_ttest_summary.csv"), index=False)

# ---------- Prepare long-format data for LMM
df_long = pd.DataFrame(long_rows)
if df_long.empty:
    raise RuntimeError("No segment means found for LMM analysis.")

# Add participant identifier (modify this parsing logic as needed)
# Example: filenames like "sub-01_movie1_PCA_timecourse.csv" → Participant = "sub-01"
df_long["Participant"] = df_long["Video"].str.extract(r"(sub-\d+|P\d+|Participant\d+)", expand=False).fillna("Unknown")

df_long["BoundaryType"] = df_long["BoundaryType"].astype("category")
df_long["Video"] = df_long["Video"].astype("category")
df_long["Participant"] = df_long["Participant"].astype("category")

# ---------- Run LMM for each component
lmm_results = []
for comp in df_long["Component"].unique():
    sub = df_long[df_long["Component"] == comp]
    try:
        model = smf.mixedlm(
            "MeanPCA ~ BoundaryType",
            data=sub,
            groups=sub["Participant"],          # ✅ participant random intercept
            re_formula="~BoundaryType"          # optional random slope per participant
        ).fit()
        beta = model.params.get("BoundaryType[T.Within]", np.nan)
        pval = model.pvalues.get("BoundaryType[T.Within]", np.nan)
        lmm_results.append({
            "Component": comp,
            "Beta_Within_minus_Cross": beta,
            "p": pval
        })
    except Exception as e:
        print(f"⚠️ LMM failed for {comp}: {e}")

lmm_df = pd.DataFrame(lmm_results)
lmm_df.to_csv(os.path.join(OUTPUT_DIR, "LMM_summary_participant.csv"), index=False)


print("\n✅ Cross-vs-within t-tests and LMM complete.")
print(f"   • t-test summary: {os.path.join(OUTPUT_DIR, 'GLOBAL_ttest_summary.csv')}")
print(f"   • LMM summary:    {os.path.join(OUTPUT_DIR, 'LMM_summary.csv')}")
