"""
=====================================================================
 Visual Change → mDES PCA (Global Linear Mixed Model)
=====================================================================
Computes a global mixed model testing whether visual change
(time-varying cosine distance from embeddings) predicts mDES PCA
component scores across all videos.

Performs:
 - Merge of *_PCA_timecourse.csv (from mDES) and *_visualchange.csv
 - Z-scoring within each video
 - Mixed model per PCA component:
     Mean_PCA_k ~ VisualChange + (1 + VisualChange | Video)

Outputs:
 - Combined z-scored dataset
 - LMM summaries per PCA
 - MixedLM_summary_all_components_visual.csv

Author: Sam Ketcheson
=====================================================================
"""

import os
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
VIS_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Neural_model\Frame_Embeddings"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\PCA_Analysis\Neural_model\lmm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_FIRST_SEC = 60

# ==========================================================
# Map mDES video names → corresponding visualchange filenames
# ==========================================================
vis_name_map = {
    "12 years_6m": "12",
    "back to the future_6m": "backToFuture",
    "Movie Task-c4": "c4",
    "Movie Task-lms": "lms",
    "Movie Task-summer": "500Days",
    "pulp_fiction_6m": "pulpFiction",
    "shawshank clip_6m": "shawshank",
    "the_prestige_6m": "prestige"
}

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

    # --- Find corresponding visual change file ---
    vis_prefix = vis_name_map.get(video, video)
    vis_path = os.path.join(VIS_DIR, f"{vis_prefix}_visualchange.csv")
    if not os.path.exists(vis_path):
        print(f"⚠️ Skipping {video}: {vis_path} not found")
        continue

    vis = pd.read_csv(vis_path)
    vis = vis.rename(columns={vis.columns[0]: "Run_time", vis.columns[1]: "VisualChange"})
    vis["Run_time"] = pd.to_numeric(vis["Run_time"], errors="coerce").astype(float)

    # --- Merge by nearest Run_time ---
    merged = pd.merge_asof(
        df.sort_values("Run_time"),
        vis.sort_values("Run_time"),
        on="Run_time"
    ).dropna(subset=["VisualChange"])

    # --- Exclude first 60 seconds ---
    merged = merged[merged["Run_time"] >= IGNORE_FIRST_SEC].reset_index(drop=True)

    if merged.empty:
        print(f"⚠️ No overlapping timepoints for {video}")
        continue

    # --- Z-score within video ---
    zcols = ["VisualChange"] + [c for c in merged.columns if c.startswith("Mean_PCA_")]
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

# Save merged dataset (optional debug)
data.to_csv(os.path.join(OUTPUT_DIR, "merged_dataset_preview.csv"), index=False)

# ==========================================================
# GLOBAL MIXED-EFFECTS MODELS
# ==========================================================
results = []
for comp in [c for c in data.columns if c.startswith("Mean_PCA_")]:
    print(f"\n🌎 Running global LMM for {comp}")
    try:
        model = smf.mixedlm(
            f"{comp} ~ VisualChange",
            data,
            groups=data["Video"],
            re_formula="~VisualChange"   # random intercept + slope by video
        )
        fit = model.fit()
        results.append({
            "Component": comp,
            "Coef_VisualChange": fit.params.get("VisualChange", float("nan")),
            "SE_VisualChange": fit.bse.get("VisualChange", float("nan")),
            "pval_VisualChange": fit.pvalues.get("VisualChange", float("nan")),
            "AIC": fit.aic,
            "BIC": fit.bic,
            "LLF": fit.llf
        })

        # Save per-component summary text
        with open(os.path.join(OUTPUT_DIR, f"LMM_summary_{comp}.txt"), "w") as f:
            f.write(str(fit.summary()))

        print(f"✅ {comp}: β={fit.params.get('VisualChange', float('nan')):.3f}, p={fit.pvalues.get('VisualChange', float('nan')):.4f}")

    except Exception as e:
        print(f"⚠️ Model failed for {comp}: {e}")

# ==========================================================
# SAVE RESULTS
# ==========================================================
if results:
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "MixedLM_summary_all_components_visual.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved z-scored results to: {out_path}")
else:
    print("\n⚠️ No successful models fit.")
