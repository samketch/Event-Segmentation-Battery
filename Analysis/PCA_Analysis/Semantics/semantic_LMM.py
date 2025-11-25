"""
=====================================================================
 Semantic Change → mDES PCA (Global Linear Mixed Model)
=====================================================================
Computes global mixed models testing whether semantic change
(time-varying 1 - cosine similarity from embeddings) predicts
mDES PCA component scores across all videos.

Performs:
 - Merge of *_PCA_timecourse.csv (from mDES) and *_aligned_semantic.csv
 - Computes SemanticChange = 1 - similarity_prev
 - Z-scoring within each video
 - Mixed model per PCA component:
     Mean_PCA_k ~ SemanticChange + (1 + SemanticChange | Video)

Outputs:
 - Combined z-scored dataset
 - Per-component LMM summaries
 - MixedLM_summary_all_components_semantic.csv

Author: Sam Ketcheson
=====================================================================
"""

import os
import glob
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
SEM_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Semantics"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\PCA_Analysis\Semantics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_FIRST_SEC = 60

# ==========================================================
# Helper: standardize video name formats
# ==========================================================
def standardize_video_name(sem_file):
    """Infer mDES video name from semantic filename."""
    base = os.path.basename(sem_file)
    vid = base.replace("_aligned_semantic.csv", "")
    mapping = {
        "12": "12 years_6m.mp4",
        "12_years": "12 years_6m.mp4",
        "backToFuture": "back to the future_6m.mp4",
        "c4": "Movie Task-c4",
        "lms": "Movie Task-lms",
        "500Days": "Movie Task-summer",
        "pulpFiction": "pulp_fiction_6m.mp4",
        "shawshank": "shawshank clip_6m.mp4",
        "prestige": "the_prestige_6m.mp4"
    }
    return mapping.get(vid, vid)

# ==========================================================
# MERGE ALL VIDEO DATA
# ==========================================================
all_dfs = []
sem_files = glob.glob(os.path.join(SEM_DIR, "*_aligned_semantic.csv"))

for sem_path in sem_files:
    video = standardize_video_name(sem_path)
    print(f"🎬 Processing {video}")

    # --- Load mDES PCA timecourse ---
    mdes_path = os.path.join(MDES_DIR, f"{video}_PCA_timecourse.csv")
    if not os.path.exists(mdes_path):
        print(f"⚠️ Skipping {video}: missing PCA file at {mdes_path}")
        continue

    df = pd.read_csv(mdes_path)
    df["Video"] = video
    df["Run_time"] = pd.to_numeric(df["Run_time"], errors="coerce")

    # --- Load semantic data ---
    sem = pd.read_csv(sem_path)
    if "time" not in sem.columns or "similarity_prev" not in sem.columns:
        print(f"⚠️ Skipping {video}: semantic file missing expected columns.")
        continue

    sem["time"] = pd.to_numeric(sem["time"], errors="coerce")
    sem["SemanticChange"] = 1 - pd.to_numeric(sem["similarity_prev"], errors="coerce").fillna(1)

    # --- Merge by nearest time ---
    df["Run_time"] = pd.to_numeric(df["Run_time"], errors="coerce").astype(float)
    sem["time"] = pd.to_numeric(sem["time"], errors="coerce").astype(float)

    merged = pd.merge_asof(
        df.sort_values("Run_time").reset_index(drop=True),
        sem[["time", "SemanticChange"]].sort_values("time").reset_index(drop=True),
        left_on="Run_time",
        right_on="time"
    ).dropna(subset=["SemanticChange"])


    merged = merged[merged["Run_time"] >= IGNORE_FIRST_SEC].reset_index(drop=True)
    if merged.empty:
        print(f"⚠️ No overlapping timepoints for {video}")
        continue

    # --- Z-score within video ---
    zcols = ["SemanticChange"] + [c for c in merged.columns if c.startswith("Mean_PCA_")]
    scaler = StandardScaler()
    merged[zcols] = scaler.fit_transform(merged[zcols])

    all_dfs.append(merged)

# ==========================================================
# COMBINE ALL VIDEOS
# ==========================================================
if not all_dfs:
    raise SystemExit("⚠️ No valid merged datasets found.")
data = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Combined dataset: {len(data)} rows across {data['Video'].nunique()} videos.")
data.to_csv(os.path.join(OUTPUT_DIR, "merged_dataset_preview.csv"), index=False)

# ==========================================================
# GLOBAL MIXED-EFFECTS MODELS
# ==========================================================
results = []
for comp in [c for c in data.columns if c.startswith("Mean_PCA_")]:
    print(f"\n🌎 Running global LMM for {comp}")
    try:
        model = smf.mixedlm(
            f"{comp} ~ SemanticChange",
            data,
            groups=data["Video"],
            re_formula="~SemanticChange"
        )
        fit = model.fit()
        results.append({
            "Component": comp,
            "Coef_SemanticChange": fit.params.get("SemanticChange", float("nan")),
            "SE_SemanticChange": fit.bse.get("SemanticChange", float("nan")),
            "pval_SemanticChange": fit.pvalues.get("SemanticChange", float("nan")),
            "AIC": fit.aic,
            "BIC": fit.bic,
            "LLF": fit.llf
        })

        # Save per-component summary text
        with open(os.path.join(OUTPUT_DIR, f"LMM_summary_{comp}.txt"), "w") as f:
            f.write(str(fit.summary()))

        print(f"✅ {comp}: β={fit.params.get('SemanticChange', float('nan')):.3f}, p={fit.pvalues.get('SemanticChange', float('nan')):.4f}")
    except Exception as e:
        print(f"⚠️ Model failed for {comp}: {e}")

# ==========================================================
# SAVE RESULTS
# ==========================================================
if results:
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "MixedLM_summary_all_components_semantic.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved all LMM summaries to: {out_path}")
else:
    print("\n⚠️ No successful models fit.")
