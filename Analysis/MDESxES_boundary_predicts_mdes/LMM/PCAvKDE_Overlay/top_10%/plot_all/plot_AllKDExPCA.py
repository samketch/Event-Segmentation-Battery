import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import pearsonr
from scipy.signal import find_peaks

# ==========================================================
# CONFIG — consistent with consensus KDE setup
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\PCAvKDE_Overlay\top_10%\plot_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERCENTILE_CUTOFF = 90   # top 10% consensus boundaries
SMOOTHING_WINDOW = 3     # smoothing window (seconds)
IGNORE_FIRST_SEC = 60    # ignore first minute for stats

# ==========================================================
# Utility functions
# ==========================================================
def seconds_to_mmss(x, pos):
    """Convert seconds → mm:ss for axis ticks."""
    m, s = divmod(int(x), 60)
    return f"{m:02d}:{s:02d}"
time_formatter = FuncFormatter(seconds_to_mmss)

def smooth(y, w=3):
    """Simple moving-average smoother."""
    if w <= 1:
        return y
    return np.convolve(y, np.ones(w)/w, mode="same")

# ==========================================================
# Main Loop
# ==========================================================
results = []
videos = [f.replace("_PCA_timecourse.csv", "") for f in os.listdir(MDES_DIR) if f.endswith("_PCA_timecourse.csv")]

for vid in videos:
    pca_file = os.path.join(MDES_DIR, f"{vid}_PCA_timecourse.csv")
    kde_file = os.path.join(KDE_DIR, f"{vid.replace('.mp4','')}_kde_timeseries.csv")

    if not (os.path.exists(pca_file) and os.path.exists(kde_file)):
        print(f"⚠️ Missing file for {vid}")
        continue

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    pca = pd.read_csv(pca_file)
    kde = pd.read_csv(kde_file)
    pca["Run_time"] = pd.to_numeric(pca["Run_time"], errors="coerce")
    kde = kde.rename(columns={kde.columns[0]: "Time", kde.columns[1]: "BoundaryStrength"})
    kde["Time"] = pd.to_numeric(kde["Time"], errors="coerce")

    # Handle duration as in consensus KDE
    if any(x in vid for x in ["500Days", "lms", "c4"]):
        dur = 11 * 60
    else:
        dur = 6 * 60
    t = np.arange(0, dur + 1)

    # Interpolate & normalize
    pca_interp = np.interp(t, pca["Run_time"], pca["Mean_PCA_1"])
    kde_interp = np.interp(t, kde["Time"], kde["BoundaryStrength"])
    pca_z = (pca_interp - np.mean(pca_interp)) / np.std(pca_interp)
    kde_z = (kde_interp - np.mean(kde_interp)) / np.std(kde_interp)
    pca_smooth = smooth(pca_z, SMOOTHING_WINDOW)
    kde_smooth = smooth(kde_z, SMOOTHING_WINDOW)

    # ----------------------------------------------------------
    # Restrict stats to after first minute
    # ----------------------------------------------------------
    valid_mask = t >= IGNORE_FIRST_SEC
    t_valid = t[valid_mask]
    pca_valid = pca_smooth[valid_mask]
    kde_valid = kde_smooth[valid_mask]

    # ----------------------------------------------------------
    # Compute correlation on trimmed time window
    # ----------------------------------------------------------
    r, p = pearsonr(pca_valid, kde_valid)
    results.append({"Video": vid, "r": r, "p": p})

    # ----------------------------------------------------------
    # Identify top-10% consensus peaks (after 1 min)
    # ----------------------------------------------------------
    cutoff = np.percentile(kde_valid, PERCENTILE_CUTOFF)
    peaks, _ = find_peaks(kde_valid, height=cutoff)
    peak_times = t_valid[peaks]

    # ----------------------------------------------------------
    # Plot PCA₁ × KDE with consensus-style markers
    # ----------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    # KDE (steelblue)
    ax2.plot(t, kde_smooth, color="steelblue", label="KDE density")
    ax2.axhline(cutoff, color="black", linestyle="--", label=f"{PERCENTILE_CUTOFF}th percentile")
    ax2.scatter(peak_times, np.interp(peak_times, t, kde_smooth),
                color="darkorange", s=40, zorder=5, label="Consensus peaks")

    # PCA₁ (blue)
    ax1.plot(t, pca_smooth, color="tab:red", lw=2, alpha=0.85, label="PCA₁ (Sensory engagement)")

    # Labels / formatting
    ax1.set_xlabel("Time (mm:ss)")
    ax1.set_ylabel("Z-scored PCA₁", color="tab:blue")
    ax2.set_ylabel("Z-scored Boundary Strength", color="steelblue")
    ax1.set_xlim(0, dur)
    ax1.xaxis.set_major_formatter(time_formatter)
    ax1.xaxis.set_major_locator(MultipleLocator(60))
    ax1.grid(True, alpha=0.3)

    plt.title(f"{vid} — PCA₁ × KDE Boundary Strength (First 60 s excluded)\n"
              f"r = {r:.3f}, p = {p:.3g}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, f"{vid}_PCA1_KDE_fullTimeline_consensus_trimmed.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"✅ Saved: {out_fig}")

# ==========================================================
# Save summary correlations
# ==========================================================
summary_csv = os.path.join(OUTPUT_DIR, "PCA1_KDE_correlations_trimmed.csv")
pd.DataFrame(results).to_csv(summary_csv, index=False)
print("\n✅ Saved correlation summary (excluding first minute) to:")
print(summary_csv)
print("🎬 All figures and CSVs saved to:")
print(OUTPUT_DIR)
