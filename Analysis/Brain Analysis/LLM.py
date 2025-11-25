# ==========================================================
# Gradient-space speed vs KDE boundary strength (LMM)
# ==========================================================
# Author: Sam Ketcheson
# ----------------------------------------------------------
# This script:
#   • Loads all gradient data + all per-video KDE timeseries
#   • Computes z-scored gradient-space "speed" (Euclidean)
#   • Aligns each movie's speed with KDE boundary strength
#   • Combines all into one dataframe
#   • Fits LMM: Speed_z ~ KDE_value + (1|Video)
#   • Saves combined CSV, model summary, and plot
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm

# ------------------ CONFIG ------------------
GRADIENT_FILE = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs.csv"
KDE_DIR       = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR    = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\State Space\LLM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_FIRST_SEC = 0
WINDOW = 0  # not used here, continuous model

# map names so gradient file ↔ KDE filenames align
NAME_MAP = {
    "pulp_fiction_6m":      "pulp_fiction_6m",
    "the_prestige_6m":      "the_prestige_6m",
    "12 years_6m":          "12 years_6m",
    "back to the future_6m":"back to the future_6m",
    "shawshank clip_6m":    "shawshank clip_6m",
    "Movie Task-lms":       "Movie Task-lms",
    "Movie Task-c4":        "Movie Task-c4",
    "Movie Task-summer":    "Movie Task-summer",
}

# ------------------ LOAD GRADIENT DATA ------------------
df = pd.read_csv(GRADIENT_FILE)
df = df.rename(columns=lambda x: x.strip())
df["Task_name"] = df["Task_name"].str.replace(".mp4", "", regex=False)
grad_cols = [c for c in df.columns if c.startswith("gradient_")]
videos = df["Task_name"].unique()
print(f"✅ Found {len(videos)} videos with gradient data.")

# ------------------ HELPERS ------------------
def zscore(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def compute_speed(G):
    diffs = np.diff(G, axis=0)
    spd = np.sqrt((diffs ** 2).sum(axis=1))
    spd = np.concatenate([[spd[0]], spd])
    return zscore(spd)

def resample_to_1hz(times_s, mat):
    times_s = np.asarray(times_s, float)
    order = np.argsort(times_s)
    times_s, mat = times_s[order], mat[order]
    t_end = int(np.floor(times_s.max()))
    if t_end < 2:
        return None, None
    t1 = np.arange(0, t_end + 1, 1.0)
    M1 = np.empty((t1.size, mat.shape[1]))
    for j in range(mat.shape[1]):
        M1[:, j] = np.interp(t1, times_s, mat[:, j])
    return t1, M1

def load_kde(video_name):
    """Load that video's KDE timeseries if it exists."""
    for f in os.listdir(KDE_DIR):
        if f.lower().startswith(video_name.lower()) and f.endswith("_kde_timeseries.csv"):
            path = os.path.join(KDE_DIR, f)
            kde = pd.read_csv(path)
            kde = kde.rename(columns=lambda x: x.strip())
            # detect time column
            time_col = [c for c in kde.columns if "time(s)" in c.lower()][0]
            val_col  = [c for c in kde.columns if "boundary" in c.lower() or "value" in c.lower()][0]
            return kde[[time_col, val_col]].rename(columns={time_col: "Time", val_col: "KDE_value"})
    return None

# ------------------ BUILD COMBINED DATA ------------------
all_rows = []

for vid_key in videos:
    vid_df = df[df["Task_name"] == vid_key].copy()
    if vid_df.empty:
        continue

    kde_key = NAME_MAP.get(vid_key, vid_key)
    kde_df = load_kde(kde_key)
    if kde_df is None:
        print(f"⚠️ No KDE timeseries for {vid_key}")
        continue

    # gradients → 1 Hz
    t_raw = vid_df["Run_time"].astype(float).values
    G_raw = vid_df[grad_cols].to_numpy(float)
    t1, G1 = resample_to_1hz(t_raw, G_raw)
    if t1 is None:
        continue
    speed_z = compute_speed(G1)

    # resample KDE to same time base
    kde_interp = np.interp(t1, kde_df["Time"].values, kde_df["KDE_value"].values)
    # clip to same length
    n = min(len(speed_z), len(kde_interp))
    t1, speed_z, kde_interp = t1[:n], speed_z[:n], kde_interp[:n]

    # assemble long dataframe
    tmp = pd.DataFrame({
        "Video": vid_key,
        "Time": t1,
        "Speed_z": speed_z,
        "KDE_value": kde_interp
    })
    tmp = tmp[tmp["Time"] > IGNORE_FIRST_SEC]
    all_rows.append(tmp)

df_all = pd.concat(all_rows, ignore_index=True)
print(f"✅ Combined dataset: {len(df_all)} timepoints across {df_all['Video'].nunique()} videos.")

# ------------------ RUN LMM ------------------
model = smf.mixedlm("Speed_z ~ KDE_value", df_all, groups=df_all["Video"])
result = model.fit()
print(result.summary())

# Compute approximate p-values
from scipy.stats import norm
params = result.params
ses = result.bse
zvals = params / ses
pvals = 2 * (1 - norm.cdf(np.abs(zvals)))

fixed_effects = pd.DataFrame({
    "Effect": params.index,
    "Estimate": params.values,
    "StdErr": ses.values,
    "z": zvals.values,
    "p": pvals
})
fixed_effects.to_csv(os.path.join(OUTPUT_DIR, "GradientSpace_vs_KDE_LMM_FixedEffects.csv"), index=False)

print("\n✅ LMM fixed effects with p-values:")
print(fixed_effects)

# ------------------ SAVE COMBINED DATA ------------------
combined_path = os.path.join(OUTPUT_DIR, "GradientSpace_vs_KDE_combined.csv")
df_all.to_csv(combined_path, index=False)
print(f"✅ Combined dataframe saved: {combined_path}")

# ------------------ VISUALIZATION ------------------
plt.figure(figsize=(6,4))
plt.scatter(df_all["KDE_value"], df_all["Speed_z"], s=5, alpha=0.3)
x = np.linspace(df_all["KDE_value"].min(), df_all["KDE_value"].max(), 100)
y = result.params["Intercept"] + result.params["KDE_value"] * x
plt.plot(x, y, color="red", lw=2)
plt.xlabel("KDE boundary strength")
plt.ylabel("Gradient-space movement (z)")
plt.title("Relationship between boundary likelihood and brain-state speed")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "GradientSpace_vs_KDE_LMM_scatter.png"), dpi=300)
plt.close()

print("\n✅ Finished LMM gradient-speed vs KDE analysis.")
