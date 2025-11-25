import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\PCAvKDE_Overlay\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOOTHING_WINDOW = 3  # seconds for moving average smoothing

# ==========================================================
# Helper: smooth series
# ==========================================================
def smooth(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w)/w, mode="same")

# ==========================================================
# Helper: Load & align PCA and KDE
# ==========================================================
def load_and_align(video_name):
    pca_file = os.path.join(MDES_DIR, f"{video_name}_PCA_timecourse.csv")
    kde_file = os.path.join(KDE_DIR, f"{video_name}_kde_timeseries.csv")

    if not (os.path.exists(pca_file) and os.path.exists(kde_file)):
        print(f"⚠️ Missing file for {video_name}")
        return None

    # --- Load data
    pca = pd.read_csv(pca_file)
    kde = pd.read_csv(kde_file)

    pca["Run_time"] = pd.to_numeric(pca["Run_time"], errors="coerce")
    kde = kde.rename(columns={kde.columns[0]: "Run_time", kde.columns[1]: "BoundaryStrength"})
    kde["Run_time"] = pd.to_numeric(kde["Run_time"], errors="coerce")

    # --- Determine actual runtime from overlap
    max_time = min(pca["Run_time"].max(), kde["Run_time"].max())
    t = np.arange(0, max_time + 1)

    # --- Interpolate to common 1-Hz grid
    pca_interp = np.interp(t, pca["Run_time"], pca["Mean_PCA_1"])
    kde_interp = np.interp(t, kde["Run_time"], kde["BoundaryStrength"])

    # --- Z-score + smooth
    pca_z = (pca_interp - np.mean(pca_interp)) / np.std(pca_interp)
    kde_z = (kde_interp - np.mean(kde_interp)) / np.std(kde_interp)
    pca_s = smooth(pca_z, SMOOTHING_WINDOW)
    kde_s = smooth(kde_z, SMOOTHING_WINDOW)

    return pd.DataFrame({"Time": t, "PCA1": pca_s, "KDE": kde_s})

# ==========================================================
# MAIN LOOP
# ==========================================================
videos = [
    f.replace("_PCA_timecourse.csv", "")
    for f in os.listdir(MDES_DIR)
    if f.endswith("_PCA_timecourse.csv")
]

correlations = []

for vid in videos:
    df = load_and_align(vid)
    if df is None:
        continue

    # --- Correlation
    # --- Ignore the first 60 seconds
    df = df[df["Time"] >= 60].reset_index(drop=True)

    # --- Correlation (after exclusion)
    if len(df) > 2:
        r, p = pearsonr(df["PCA1"], df["KDE"])
    else:
        r, p = np.nan, np.nan

    correlations.append({"Video": vid, "r": r, "p": p})

    # --- Plot overlay with r annotation
    plt.figure(figsize=(10,4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df["Time"], df["PCA1"], color="tab:blue", lw=1.5, label="PCA₁ (Sensory Engagement)")
    ax2.plot(df["Time"], df["KDE"], color="tab:red", lw=1.5, alpha=0.6, label="KDE Boundary Strength")

    ax1.set_xlabel("Run Time (s)")
    ax1.set_ylabel("PCA₁ (z-scored)", color="tab:blue")
    ax2.set_ylabel("Boundary Strength (z-scored)", color="tab:red")

    plt.title(f"{vid}\nPCA₁ vs KDE — r={r:.2f}, p={p:.3g}")
    ax1.set_xlim(0, df["Time"].max())

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{vid}_PCA1_vs_KDE_corr.png"), dpi=300)
    plt.close()

    print(f"✅ Saved overlay with correlation: {vid} (r={r:.2f}, p={p:.3g})")

# ==========================================================
# Summary histogram across videos
# ==========================================================
corr_df = pd.DataFrame(correlations)
corr_df.to_csv(os.path.join(OUTPUT_DIR, "PCA1_KDE_correlations.csv"), index=False)

plt.figure(figsize=(6,4))
plt.hist(corr_df["r"], bins=np.linspace(-1,1,21), color="gray", edgecolor="black", alpha=0.8)
plt.axvline(0, color="k", ls="--", lw=1)
plt.xlabel("Pearson r (PCA₁ ↔ KDE)")
plt.ylabel("Number of Videos")
plt.title("Distribution of PCA₁–KDE Correlations Across Videos")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "PCA1_KDE_correlation_histogram.png"), dpi=300)
plt.close()

print("\n✅ Summary correlation histogram saved.")
print("✅ Per-video overlays saved to:", OUTPUT_DIR)
