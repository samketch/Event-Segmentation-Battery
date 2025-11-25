import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
GRAD_FILE = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs.csv"
SPEED_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\State Space\LLM\GradientSpace_vs_KDE_combined.csv"
OUTPUT = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\State Space\Gradients\Figures\GradientSpace_GlobalTrajectory_smooth_rescaled.png"

SMOOTH_SIGMA = 3     # Gaussian smoothing kernel (~3 s)
ADD_OFFSETS = True   # small per-movie offset to avoid complete overlap

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
grad = pd.read_csv(GRAD_FILE)
grad["Task_name"] = grad["Task_name"].str.replace(".mp4", "", regex=False)
speed = pd.read_csv(SPEED_FILE)
speed["Video"] = speed["Video"].str.replace(".mp4", "", regex=False)

videos = sorted(set(speed["Video"]).intersection(set(grad["Task_name"])))
print(f"✅ Found {len(videos)} overlapping videos")

# ----------------------------------------------------
# BUILD GLOBAL DATASET
# ----------------------------------------------------
all_rows, boundary_rows = [], []

for vid in videos:
    g = grad[grad["Task_name"] == vid].copy()
    s = speed[speed["Video"] == vid].copy()
    if g.empty or s.empty:
        continue

    # interpolate KDE + speed to gradient time base
    kde_interp = np.interp(g["Run_time"], s["run_time"], s["KDE_value"])
    spd_interp = np.interp(g["Run_time"], s["run_time"], s["Speed_z"])

    # determine top 10 % KDE *within this movie*
    kde_thresh = np.percentile(kde_interp, 90)
    boundary_mask = kde_interp >= kde_thresh

    # smooth gradients & speed for nicer trajectories
    for col in ["gradient_1", "gradient_2", "gradient_3"]:
        raw = g[col].to_numpy()
        smoothed = gaussian_filter1d(raw, sigma=SMOOTH_SIGMA)
        # Rescale to match original variance
        smoothed = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-9)
        smoothed *= raw.std()
        smoothed += raw.mean()
        g[col] = smoothed

    spd_interp = gaussian_filter1d(spd_interp, sigma=SMOOTH_SIGMA)
    kde_interp = gaussian_filter1d(kde_interp, sigma=SMOOTH_SIGMA)

    g["Video"] = vid
    g["KDE_value"] = kde_interp
    g["Speed_z"] = spd_interp
    g["is_boundary"] = boundary_mask

    all_rows.append(g)
    boundary_rows.append(g[g["is_boundary"]])

df_all = pd.concat(all_rows, ignore_index=True)
df_bound = pd.concat(boundary_rows, ignore_index=True)
print(f"✅ Combined dataset: {len(df_all)} points, {len(df_bound)} boundaries")



# ----------------------------------------------------
# OPTIONAL: small offsets to separate movies
# ----------------------------------------------------
if ADD_OFFSETS:
    offsets = np.linspace(-1, 1, len(videos))
    for off, vid in zip(offsets, videos):
        df_all.loc[df_all["Video"] == vid, "gradient_2"] += off * 0.4  # adjust magnitude if needed
        df_bound.loc[df_bound["Video"] == vid, "gradient_2"] += off * 0.4

# ----------------------------------------------------
# 3D TRAJECTORY PLOT
# ----------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for vid in videos:
    g = df_all[df_all["Video"] == vid].sort_values("Run_time")
    spd_norm = (g["Speed_z"] - g["Speed_z"].min()) / (g["Speed_z"].max() - g["Speed_z"].min() + 1e-9)
    colormap = cm.viridis(spd_norm)

    for i in range(len(g) - 1):
        ax.plot(
            g["gradient_1"].iloc[i:i+2],
            g["gradient_2"].iloc[i:i+2],
            g["gradient_3"].iloc[i:i+2],
            color=colormap[i],
            linewidth=1.6,
            alpha=0.5
        )

# Event boundaries
ax.scatter(
    df_bound["gradient_1"], df_bound["gradient_2"], df_bound["gradient_3"],
    s=60, c="red", edgecolor="black", linewidth=0.5, alpha=0.9,
    label="Top 10 % KDE (per movie)"
)

# Camera and labels
ax.view_init(elev=25, azim=35)
ax.set_xlabel("Gradient 1 (Sensory → DMN)")
ax.set_ylabel("Gradient 2 (Visual → Motor)")
ax.set_zlabel("Gradient 3 (Attention/Control)")
ax.set_title("Global Neural Trajectory in Gradient Space — Smoothed & Rescaled (All Movies)")
ax.legend()

# Colorbar for speed
mappable = plt.cm.ScalarMappable(cmap="viridis")
mappable.set_array(df_all["Speed_z"])
cbar = plt.colorbar(mappable, ax=ax, shrink=0.4, pad=0.1)
cbar.set_label("Gradient-space speed (z)")



plt.savefig(OUTPUT, dpi=400)
plt.close()

print(f"✅ Saved cleaned global trajectory figure → {OUTPUT}")

print(df_all[["gradient_1","gradient_2","gradient_3"]].corr())

