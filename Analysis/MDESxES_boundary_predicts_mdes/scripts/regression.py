import os
import pandas as pd
from statsmodels.formula.api import ols

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# MAIN LOOP
# ==========================================================
for mfile in os.listdir(MDES_DIR):
    if not mfile.endswith("_PCA_timecourse.csv"):
        continue

    base = os.path.basename(mfile)
    video_id = base.replace("_PCA_timecourse.csv", "").replace(".mp4", "")
    print(f"\n🎬 Processing {video_id}")

    kde_file = os.path.join(BOUNDARY_DIR, f"{video_id}_kde_timeseries.csv")
    if not os.path.exists(kde_file):
        print(f"⚠️ Skipping {video_id}: KDE file not found at {kde_file}")
        continue

    # --- Load data ---
    df_mdes = pd.read_csv(os.path.join(MDES_DIR, mfile))
    df_bound = pd.read_csv(kde_file)

    # --- Identify boundary column ---
    possible_cols = [c for c in df_bound.columns if "boundary" in c.lower() or "kde" in c.lower()]
    if not possible_cols:
        print(f"⚠️ No boundary column found in {kde_file}")
        continue
    df_bound = df_bound.rename(columns={possible_cols[0]: "BoundaryStrength"})

    # --- Align on Run_time ---
    if "Run_time" not in df_mdes.columns:
        print(f"⚠️ No Run_time in {mfile}")
        continue
    if "Run_time" not in df_bound.columns:
        time_col = next((c for c in df_bound.columns if c.lower() == "time"), None)
        if time_col:
            df_bound = df_bound.rename(columns={time_col: "Run_time"})
        else:
            df_bound["Run_time"] = range(len(df_bound))

    df_mdes["Run_time_rounded"] = df_mdes["Run_time"].round(1)
    df_bound["Run_time_rounded"] = df_bound["Run_time"].round(1)
    df = pd.merge(df_mdes, df_bound[["Run_time_rounded", "BoundaryStrength"]],
                  on="Run_time_rounded", how="inner")

    if df.empty:
        print(f"⚠️ No overlap between mDES and boundaries for {video_id}")
        continue

    # --- Regress each PCA component on boundary strength ---
    pca_cols = [c for c in df.columns if "pca" in c.lower()]
    results = []

    for pca_col in pca_cols:
        formula = f"{pca_col} ~ BoundaryStrength"
        try:
            model = ols(formula, data=df).fit()
            results.append({
                "video": video_id,
                "component": pca_col,
                "coef": model.params["BoundaryStrength"],
                "pval": model.pvalues["BoundaryStrength"],
                "r2": model.rsquared
            })
        except Exception as e:
            print(f"⚠️ Regression failed for {video_id} {pca_col}: {e}")

    # --- Save summary ---
    if results:
        out_df = pd.DataFrame(results)
        out_csv = os.path.join(OUTPUT_DIR, f"{video_id}_boundary_predicts_mdes.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"✅ Saved regression results → {out_csv}")

print("\n🎬 All regressions complete! Boundary → mDES direction.")
