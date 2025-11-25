import os
import pandas as pd
import statsmodels.formula.api as smf

# ==========================================================
# CONFIG
# ==========================================================
INPUT = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All.csv"
OUTPUT = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\MixedEffects_PCA_Events\MixedModel_BinnedResults.csv"

BIN_SIZE = 10  # seconds per bin — match MDES resolution 

# ==========================================================
# LOAD & BIN DATA
# ==========================================================
df = pd.read_csv(INPUT)

# Clean numeric columns
df = df.rename(columns={c: c.strip() for c in df.columns})
df = df.dropna(subset=["BoundaryDensity","Mean_PCA_1","Mean_PCA_2","Mean_PCA_3","Mean_PCA_4", "Mean_PCA_5"])

# Bin by time
df["bin"] = (df["Time_rounded"] // BIN_SIZE) * BIN_SIZE

# Average within bins
binned = (
    df.groupby(["Task_name","bin"], as_index=False)
    [["BoundaryDensity","Mean_PCA_1","Mean_PCA_2","Mean_PCA_3","Mean_PCA_4", "Mean_PCA_5"]]
    .mean()
)
print(f"✅ Binned data created: {len(binned)} rows, {binned['Task_name'].nunique()} videos")

if "Task_name" == "Movie Task-lms":
    BIN_SIZE = 15
if "Task_name" == "Movie Task-summer":
    BIN_SIZE = 15
if "Task_name" == "Movie Task-c4":
    BIN_SIZE = 15
# ==========================================================
# RUN MIXED MODELS
# ==========================================================
results = []
for pca in ["Mean_PCA_1","Mean_PCA_2","Mean_PCA_3","Mean_PCA_4", "Mean_PCA_5"]:
    try:
        m = smf.mixedlm(f"{pca} ~ BoundaryDensity", data=binned, groups=binned["Task_name"]).fit()
        coef = m.params["BoundaryDensity"]
        se = m.bse["BoundaryDensity"]
        z = coef / se
        p = m.pvalues["BoundaryDensity"]
        ci_low, ci_high = m.conf_int().loc["BoundaryDensity"]
        results.append({
            "PCA": pca,
            "Beta_BoundaryDensity": coef,
            "SE": se,
            "z": z,
            "p": p,
            "CI_lower": ci_low,
            "CI_upper": ci_high,
            "N_obs": len(binned),
            "N_groups": binned["Task_name"].nunique(),
            "LogLik": m.llf
        })
        print(f"✅ {pca}: β = {coef:.3f}, p = {p:.3f}")
    except Exception as e:
        print(f"⚠️ Model failed for {pca}: {e}")

# ==========================================================
# SAVE RESULTS
# ==========================================================
results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
results_df.to_csv(OUTPUT, index=False)
print(f"\n🎉 Mixed-effects summary saved → {OUTPUT}")
print(results_df)
