import os
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# MERGE ALL VIDEO DATA
# ==========================================================
all_dfs = []
for f in os.listdir(MDES_DIR):
    if not f.endswith("_PCA_timecourse.csv"):
        continue

    video = f.replace("_PCA_timecourse.csv", "").replace(".mp4", "")
    print(f"🎬 Loading {video}")

    # --- Load PCA timecourse ---
    df = pd.read_csv(os.path.join(MDES_DIR, f))
    df["Video"] = video
    df["Run_time"] = pd.to_numeric(df["Run_time"], errors="coerce").astype(float)

    # --- Load KDE boundary file ---
    kde_path = os.path.join(BOUNDARY_DIR, f"{video}_kde_timeseries.csv")
    if not os.path.exists(kde_path):
        print(f"⚠️ Skipping {video}: {kde_path} not found")
        continue

    kde = pd.read_csv(kde_path)
    kde = kde.rename(columns={kde.columns[0]: "Run_time", kde.columns[1]: "BoundaryStrength"})
    kde["Run_time"] = pd.to_numeric(kde["Run_time"], errors="coerce").astype(float)

    # --- Merge by nearest Run_time ---
    merged = pd.merge_asof(
        df.sort_values("Run_time"),
        kde.sort_values("Run_time"),
        on="Run_time"
    ).dropna(subset=["BoundaryStrength"])

    if merged.empty:
        print(f"⚠️ No overlapping timepoints for {video}")
        continue

    # --- Z-score within video ---
    zcols = ["BoundaryStrength"] + [c for c in merged.columns if c.startswith("Mean_PCA_")]
    scaler = StandardScaler()
    merged[zcols] = scaler.fit_transform(merged[zcols])

    all_dfs.append(merged)

# ==========================================================
# COMBINE ALL VIDEOS
# ==========================================================
if not all_dfs:
    raise SystemExit("⚠️ No valid video datasets found.")
data = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Combined dataset: {len(data)} rows across {data['Video'].nunique()} videos.")
data.to_csv(os.path.join(OUTPUT_DIR, "combined_data.csv"), index=False)
print(f"✅ Saved combined dataset to: {os.path.join(OUTPUT_DIR, 'combined_data.csv')}")

# ==========================================================
# GLOBAL MIXED-EFFECTS MODELS
# ==========================================================
results = []
for comp in [c for c in data.columns if c.startswith("Mean_PCA_")]:
    print(f"\nRunning global LMM for {comp}")
    try:
        model = smf.mixedlm(
            f"{comp} ~ BoundaryStrength",
            data,
            groups=data["Video"],
            re_formula="~BoundaryStrength"   # random intercept + slope
        )
        fit = model.fit()
        results.append({
            "Component": comp,
            "Coef_BoundaryStrength": fit.params.get("BoundaryStrength", float("nan")),
            "SE_BoundaryStrength": fit.bse.get("BoundaryStrength", float("nan")),
            "pval_BoundaryStrength": fit.pvalues.get("BoundaryStrength", float("nan")),
            "AIC": fit.aic,
            "BIC": fit.bic,
            "LLF": fit.llf
        })
    except Exception as e:
        print(f"⚠️ Model failed for {comp}: {e}")

# ==========================================================
# SAVE RESULTS
# ==========================================================
if results:
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "MixedLM_summary_all_components_zscored.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved z-scored results to: {out_path}")
else:
    print("\n⚠️ No successful models fit.")
