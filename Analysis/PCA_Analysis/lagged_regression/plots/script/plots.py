import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy.signal import correlate

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_temporal_analysis\LMM\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Helper: merge mDES & KDE time series
# ==========================================================
def load_video_timeseries(video_name):
    """Find matching mDES + KDE timecourse files and merge on Run_time."""
    mfile = os.path.join(MDES_DIR, f"{video_name}_PCA_timecourse.csv")
    video_name = video_name.replace(".mp4", "")
    kde_file = os.path.join(BOUNDARY_DIR, f"{video_name}_kde_timeseries.csv")

    if not os.path.exists(mfile):
        print(f"⚠️ No PCA file for {video_name}")
        return None
    if not os.path.exists(kde_file):
        print(f"⚠️ No KDE file for {video_name}")
        return None

    # --- Load data
    df_mdes = pd.read_csv(mfile)
    df_mdes["Run_time"] = pd.to_numeric(df_mdes["Run_time"], errors="coerce").astype(float)

    kde = pd.read_csv(kde_file)
    kde = kde.replace(".mp4", "")
    kde = kde.rename(columns={kde.columns[0]: "Run_time", kde.columns[1]: "BoundaryStrength"})
    kde["Run_time"] = pd.to_numeric(kde["Run_time"], errors="coerce").astype(float)

    # --- Merge
    df = pd.merge_asof(
        df_mdes.sort_values("Run_time"),
        kde.sort_values("Run_time"),
        on="Run_time"
    ).dropna(subset=["BoundaryStrength", "Mean_PCA_1"])

    # --- Z-score both series for comparability ---
    for col in ["BoundaryStrength"] + [c for c in df.columns if c.startswith("Mean_PCA_")]:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)


    if df.empty:
        print(f"⚠️ No overlapping timepoints for {video_name}")
        return None

    df["Video"] = video_name
    return df


# ==========================================================
# 1️⃣ Overlay PCA₁ and BoundaryStrength
# ==========================================================
# ==========================================================
# AUTO-DETECT VIDEOS
# ==========================================================
videos = [
    f.replace("_PCA_timecourse.csv", "")
    for f in os.listdir(MDES_DIR)
    if f.endswith("_PCA_timecourse.csv")
]
print(f"Found {len(videos)} videos:", videos)


for vid in videos:
    df = load_video_timeseries(vid)
    if df is None:
        continue

    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df["Run_time"], df["Mean_PCA_1"], color="tab:blue", label="Mean_PCA_1 (Sensory)")
    ax2.plot(df["Run_time"], df["BoundaryStrength"], color="tab:red", alpha=0.6, label="Boundary Strength")

    ax1.set_xlabel("Run Time (s)")
    ax1.set_ylabel("PCA₁ (Sensory Engagement)", color="tab:blue")
    ax2.set_ylabel("Boundary Strength", color="tab:red")
    plt.title(f"{vid} — PCA₁ vs Boundary Strength")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{vid}_PCA1_vs_BoundaryOverlay.png"), dpi=300)
    plt.close()
    print(f"✅ Saved overlay for {vid}")


# ==========================================================
# 2️⃣ Lagged regressions (±2 timepoints)
# ==========================================================
from patsy import dmatrices, ModelDesc, EvalFactor

lag_results = []
for vid in videos:
    df = load_video_timeseries(vid)
    if df is None:
        continue

    for lag in range(-2, 3):
        colname = f"Boundary_lag{lag}"
        df[colname] = df["BoundaryStrength"].shift(lag)
        sub = df.dropna(subset=[colname, "Mean_PCA_1"])
        if len(sub) < 5:
            continue

        # Use Q() to safely reference the column name
        model = ols(f"Mean_PCA_1 ~ Q('{colname}')", data=sub).fit()

        lag_results.append({
            "Video": vid,
            "Lag": lag,
            "Coef": model.params.get(f"Q('{colname}')", np.nan),
            "pval": model.pvalues.get(f"Q('{colname}')", np.nan),
            "R2": model.rsquared
        })


lag_df = pd.DataFrame(lag_results)
if not lag_df.empty:
    lag_df.to_csv(os.path.join(OUTPUT_DIR, "Lagged_Regression_PCA1_vs_Boundary.csv"), index=False)
    print("✅ Saved lagged regression results")

# ==========================================================
# Plot slopes vs lag, marking significance
# ==========================================================
    plt.figure(figsize=(8, 5))

    for vid in lag_df["Video"].unique():
        sub = lag_df[lag_df["Video"] == vid].sort_values("Lag")

        # Base line for all lags
        plt.plot(sub["Lag"], sub["Coef"], marker="o", linestyle="-", alpha=0.4, label=f"{vid}")

        # Mark significant points (p < .05)
        sig = sub[sub["pval"] < 0.05]
        if not sig.empty:
            plt.scatter(sig["Lag"], sig["Coef"], color="red", s=70, edgecolor="k", zorder=5)
            # Optional: add star annotation
            for _, row in sig.iterrows():
                plt.text(row["Lag"], row["Coef"], "★", color="red",
                        fontsize=12, ha="center", va="bottom", fontweight="bold")

    mean_df = lag_df.groupby("Lag", as_index=False)["Coef"].mean()
    plt.plot(mean_df["Lag"], mean_df["Coef"], color="black", lw=2, label="Group mean")
    plt.axhline(0, color="k", linestyle="--")
    plt.title("Lagged Regression: Boundary → PCA₁\n★ = p < .05")
    plt.xlabel("Lag (timepoints; + = boundary leads PCA₁)")
    plt.ylabel("β (slope)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Lagged_Regression_Slopes_significant.png"), dpi=300)
    plt.close()
    print("✅ Saved significance-marked lag plot")

else:
    print("⚠️ No lag regression data found; skipping plot.")


# ==========================================================
# 3️⃣ Cross-Correlation (Boundary ↔ PCA₁)
# ==========================================================
ccf_results = []
lags = np.arange(-10, 11)

for vid in videos:
    df = load_video_timeseries(vid)
    if df is None:
        continue

    x = df["BoundaryStrength"] - df["BoundaryStrength"].mean()
    y = df["Mean_PCA_1"] - df["Mean_PCA_1"].mean()
    corr = correlate(y, x, mode="full")
    corr /= np.max(np.abs(corr))
    lags_full = np.arange(-len(x)+1, len(x))
    mask = (lags_full >= -10) & (lags_full <= 10)
    corr = corr[mask]
    lags_short = lags_full[mask]

    ccf_results.append(pd.DataFrame({
        "Video": vid,
        "Lag": lags_short,
        "Correlation": corr
    }))

    plt.figure(figsize=(6, 4))
    plt.plot(lags_short, corr, marker="o")
    plt.axvline(0, color="k", linestyle="--")
    plt.title(f"Cross-Correlation (Boundary ↔ PCA₁)\n{vid}")
    plt.xlabel("Lag (timepoints; +ve = boundary leads)")
    plt.ylabel("Correlation (r)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{vid}_CrossCorrelation.png"), dpi=300)
    plt.close()

if ccf_results:
    ccf_df = pd.concat(ccf_results, ignore_index=True)
    ccf_df.to_csv(os.path.join(OUTPUT_DIR, "CrossCorrelation_PCA1_vs_Boundary.csv"), index=False)
    print("✅ Saved cross-correlation results")
else:
    print("⚠️ No cross-correlation data to save.")

print("\n🎬 All analyses complete! Plots and CSVs saved to:")
print(OUTPUT_DIR)
