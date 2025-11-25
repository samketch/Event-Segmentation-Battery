# =====================================================================
# Plot_PCA1_perVideo_with_ConsensusBoundaries.py
# =====================================================================
# Generates one PCA 1 timecourse plot per video.
# Each figure includes red dashed lines marking the
# consensus boundaries from the boundary CSV.
# =====================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
BOUNDARY_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
OUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\PCA1_Further_analysis\PCA1_Plot_perVideo"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH).sort_values(["VideoName", "Run_time"]).reset_index(drop=True)
bound_df = pd.read_csv(BOUNDARY_FILE)

# Clean video names in boundary file
bound_df["Video_clean"] = (
    bound_df.iloc[:, 0].astype(str)
    .str.lower()
    .str.replace(".mp4", "", regex=False)
    .str.strip()
)

# Detect boundary time column
time_col = [c for c in bound_df.columns if any(k in c.lower() for k in ["time", "sec", "boundary"])][0]

# ---------------- LOOP THROUGH VIDEOS ----------------
for vid, sub in df.groupby("VideoName"):
    sub = sub.sort_values("Run_time")
    stem = Path(vid).stem.lower()

    # Find consensus boundaries for this video
    b_times = bound_df.loc[bound_df["Video_clean"].str.contains(stem, case=False), time_col].dropna().values

    # Skip if no boundaries found
    if len(b_times) == 0:
        print(f"⚠️ No consensus boundaries found for {vid}, skipping.")
        continue

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10, 4))
    plt.plot(sub["Run_time"], sub["Mean_PCA_1"], color="royalblue", lw=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean_PCA_1 (z)")
    plt.title(f"PCA 1 timecourse with consensus boundaries — {vid}")

    # Add boundary lines
    for b in b_times:
        plt.axvline(b, color="red", linestyle="--", lw=0.8, alpha=0.7)

    plt.tight_layout()

    # Save each figure
    out_path = os.path.join(OUT_DIR, f"{stem}PCA1_withBoundaries.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot for {vid} → {out_path}")

print("\n✅ All video plots saved successfully.")
