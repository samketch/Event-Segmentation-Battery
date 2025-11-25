import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\Across vs Within\LMM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_FIRST_SEC = 60

# ==========================================================
# LOAD & ALIGN DATA PER VIDEO
# ==========================================================
def load_one(video):
    """Load and align 1-Hz z-scored PCA and KDE time series for one video."""
    p_path = os.path.join(MDES_DIR, f"{video}_PCA_timecourse.csv")
    k_path = os.path.join(KDE_DIR, f"{video}_kde_timeseries.csv")
    if not (os.path.exists(p_path) and os.path.exists(k_path)):
        print(f"⚠️ Missing file for {video}")
        return None

    # Load
    p = pd.read_csv(p_path)
    k = pd.read_csv(k_path)
    p["Run_time"] = pd.to_numeric(p["Run_time"], errors="coerce")
    k = k.rename(columns={k.columns[0]: "Time", k.columns[1]: "KDE"})
    k["Time"] = pd.to_numeric(k["Time"], errors="coerce")

    # Interpolate to 1 Hz
    tmax = int(min(p["Run_time"].max(), k["Time"].max()))
    t = np.arange(0, tmax + 1)
    out = pd.DataFrame({"Video": video, "Time": t})

    kde = np.interp(t, k["Time"], k["KDE"])
    kde_z = (kde - np.mean(kde)) / (np.std(kde) + 1e-12)
    out["KDE_z"] = kde_z

    for c in [c for c in p.columns if c.startswith("Mean_PCA_")]:
        v = np.interp(t, p["Run_time"], p[c])
        v = (v - np.mean(v)) / (np.std(v) + 1e-12)
        out[c] = v

    return out[out["Time"] >= IGNORE_FIRST_SEC].reset_index(drop=True)

# ==========================================================
# LOAD ALL VIDEOS
# ==========================================================
videos = [
    f.replace("_PCA_timecourse.csv", "")
    for f in os.listdir(MDES_DIR)
    if f.endswith("_PCA_timecourse.csv")
]

dfs = []
for v in videos:
    kde_path = os.path.join(KDE_DIR, f"{v}_kde_timeseries.csv")
    if os.path.exists(kde_path):
        df_v = load_one(v)
        if df_v is not None:
            dfs.append(df_v)

if not dfs:
    raise RuntimeError("No valid video datasets found!")

df = pd.concat(dfs, ignore_index=True)

# ==========================================================
# ADD PARTICIPANT ID
# ==========================================================
# Try to extract participant ID from video names like 'P01_MovieTask' or 'sub-01'
df["Participant"] = df["Video"].str.extract(r"(P\d+|sub-\d+|participant\d+)", expand=False)
df["Participant"] = df["Participant"].fillna("Unknown")

df["Video"] = df["Video"].astype("category")
df["Participant"] = df["Participant"].astype("category")

# ==========================================================
# FIT LMM PER PCA COMPONENT
# ==========================================================
results = []

for comp in [c for c in df.columns if c.startswith("Mean_PCA_")]:
    print(f"\n🧠 Running LMM for {comp} ...")
    sub = df[["Video", "Participant", "KDE_z", comp]].dropna()

    try:
        # Mixed model with random intercepts for Participant and Video
        m = smf.mixedlm(
            f"{comp} ~ KDE_z",
            data=sub,
            groups=sub["Participant"],
            re_formula="~1",
            vc_formula={"Video": "0 + Video"}
        ).fit()

        beta = m.params.get("KDE_z", np.nan)
        se = m.bse.get("KDE_z", np.nan)
        pval = m.pvalues.get("KDE_z", np.nan)

        results.append({
            "Component": comp,
            "Beta_KDE": beta,
            "SE": se,
            "p": pval,
            "N_timepoints": len(sub)
        })

        # Save full summary for transparency
        with open(os.path.join(OUTPUT_DIR, f"{comp}_LMM_summary.txt"), "w") as f:
            f.write(m.summary().as_text())

        print(f"   ✅ β={beta:.4f}, p={pval:.4g}")

    except Exception as e:
        print(f"   ⚠️ LMM failed for {comp}: {e}")
        results.append({
            "Component": comp,
            "Beta_KDE": np.nan,
            "SE": np.nan,
            "p": np.nan,
            "N_timepoints": len(sub),
            "error": str(e)
        })

# ==========================================================
# SAVE SUMMARY TABLE
# ==========================================================
res_df = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, "timepoint_LMM_KDE.csv")
res_df.to_csv(out_path, index=False)

print("\n✅ Timepoint-level LMM analysis complete.")
print(f"Results saved to: {out_path}")
