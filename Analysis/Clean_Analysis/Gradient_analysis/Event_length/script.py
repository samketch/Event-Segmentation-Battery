import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import zscore
import os

# ----------------------------------------------------------
# LOAD FILES
# ----------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
)
bounds = pd.read_csv(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
)

# ----------------------------------------------------------
# BUILD EVENT-LEVEL DATAFRAME
# ----------------------------------------------------------
event_rows = []
for vid, sub in df.groupby("VideoName"):
    these_bounds = np.sort(bounds.loc[bounds["VideoName"] == vid, "ConsensusTime(s)"].values)
    if len(these_bounds) == 0:
        continue

    these_bounds = np.concatenate(([sub["Run_time"].min()], these_bounds, [sub["Run_time"].max()]))
    for i in range(len(these_bounds) - 1):
        start, end = these_bounds[i], these_bounds[i + 1]
        event = sub[(sub["Run_time"] >= start) & (sub["Run_time"] < end)]
        if event.empty:
            continue

        event_mean = event[[f"Mean_gradient_{j}" for j in range(1, 6)]].mean()
        event_rows.append(
            {
                "VideoName": vid,
                "EventID": i + 1,
                "Start_time": start,
                "End_time": end,
                "EventLength": end - start,
                **event_mean.to_dict(),
            }
        )

events = pd.DataFrame(event_rows)
events = events.reset_index(drop=True)

print(f"✅ Built {len(events)} event rows from {events['VideoName'].nunique()} videos")

# ----------------------------------------------------------
# PREPROCESSING FOR MODEL
# ----------------------------------------------------------
grad_cols = [f"Mean_gradient_{i}" for i in range(1, 6)]

# Drop any all-NaN rows
events = events.dropna(subset=["EventLength"] + grad_cols)

# Z-score within each video
for g in grad_cols:
    events[g + "_z"] = events.groupby("VideoName")[g].transform(
        lambda x: zscore(x, nan_policy="omit")
    )

# Drop any rows with NaNs introduced by z-scoring
events = events.dropna(subset=[f"{g}_z" for g in grad_cols]).reset_index(drop=True)

# Ensure grouping variable is categorical and index is fresh
events["VideoName"] = events["VideoName"].astype("category")
events = events.reset_index(drop=True)

print(events.groupby("VideoName").size())

# ----------------------------------------------------------
# FIT MIXED MODEL (RANDOM INTERCEPT BY VIDEO)
# ----------------------------------------------------------
formula = "EventLength ~ " + " + ".join([f"{g}_z" for g in grad_cols])

try:
    model = smf.mixedlm(formula, data=events, groups=events["VideoName"])
    fit = model.fit(method="lbfgs", reml=False)
    print(fit.summary())
except Exception as e:
    print("⚠️ MixedLM failed, falling back to OLS for debugging:")
    print(e)
    import statsmodels.api as sm
    fit = sm.OLS.from_formula(formula, data=events).fit()
    print(fit.summary())

# ----------------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------------
os.makedirs(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Gradient_analysis\Event_length",
    exist_ok=True,
)
out_path = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Gradient_analysis\Event_length\Gradient_vs_EventLength_LMM.txt"
with open(out_path, "w") as f:
    f.write(fit.summary().as_text())

print(f"\n✅ Model summary saved to:\n{out_path}")
# ==========================================================
# 3D scatterplots of event length in gradient space
# ==========================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm
import numpy as np
import os

# Directory for saving
out_dir = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Gradient_analysis\Event_length"
os.makedirs(out_dir, exist_ok=True)

# Normalize event length for coloring
c = events["EventLength"]
norm = plt.Normalize(vmin=c.min(), vmax=c.max())
colors = cm.viridis(norm(c))

# -------- Plot 1: Gradients 1–3 --------
fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(111, projection="3d")
p1 = ax1.scatter(
    events["Mean_gradient_1_z"],
    events["Mean_gradient_2_z"],
    events["Mean_gradient_3_z"],
    c=c, cmap="viridis", s=60, edgecolor="k", alpha=0.8
)
ax1.set_xlabel("Gradient 1 (z)")
ax1.set_ylabel("Gradient 2 (z)")
ax1.set_zlabel("Gradient 3 (z)")
cb1 = fig1.colorbar(p1, ax=ax1, pad=0.1)
cb1.set_label("Event Length (s)")
ax1.set_title("Event Length in Gradient Space (1–3)")
plt.tight_layout()
path1 = os.path.join(out_dir, "EventLength_Gradients1_2_3.png")
plt.savefig(path1, dpi=300)
plt.show()
print(f"✅ Saved: {path1}")

# -------- Plot 2: Gradients 3–5 --------
fig2 = plt.figure(figsize=(7, 6))
ax2 = fig2.add_subplot(111, projection="3d")
p2 = ax2.scatter(
    events["Mean_gradient_3_z"],
    events["Mean_gradient_4_z"],
    events["Mean_gradient_5_z"],
    c=c, cmap="viridis", s=60, edgecolor="k", alpha=0.8
)
ax2.set_xlabel("Gradient 3 (z)")
ax2.set_ylabel("Gradient 4 (z)")
ax2.set_zlabel("Gradient 5 (z)")
cb2 = fig2.colorbar(p2, ax=ax2, pad=0.1)
cb2.set_label("Event Length (s)")
ax2.set_title("Event Length in Gradient Space (3–5)")
plt.tight_layout()
path2 = os.path.join(out_dir, "EventLength_Gradients3_4_5.png")
plt.savefig(path2, dpi=300)
plt.show()
print(f"✅ Saved: {path2}")

