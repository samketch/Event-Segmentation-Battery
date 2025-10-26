"""
==========================================================
ThoughtSpace PCA — Global PCA → Per-Video Time Series
==========================================================
Uses global PCA scores (same decomposition for all data)
to compute per-video average time series trajectories of
each PCA component (PCA_1, PCA_2, etc.) across probes.

Inputs:
 - ThoughtSpace output/pca_scores_original_format.csv
 - Metadata in original dataset (ProbeVersion, ProbeNumber, VideoName)
Outputs:
 - Time_Series/<Video>_Averaged_PCA_TimeSeries.csv
 - Time_Series/<Video>_ComponentDynamics.png

Author: Sam Ketcheson
==========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Config
# ==========================================================
PCA_RESULTS = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\ThoughtSpace_Results\grouped_PCA_byVideo\csvdata\fitted_pca_scores.csv"
MASTER_DATA = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\output.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Time_Series"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================
df_pca = pd.read_csv(PCA_RESULTS)
df_meta = pd.read_csv(MASTER_DATA)

# Merge metadata back into PCA scores
meta_cols = ["VideoName", "ProbeVersion", "ProbeNumber"]
df_meta = df_meta[meta_cols]
df = pd.concat([df_meta, df_pca.filter(like="PCA_")], axis=1)

pca_cols = [c for c in df.columns if c.startswith("PCA_")]
videos = df["VideoName"].dropna().unique()

print(f"\nFound {len(videos)} videos: {videos}")

# ==========================================================
# Compute per-video averages and plots
# ==========================================================
for video in videos:
    dfv = df[df["VideoName"] == video].dropna(subset=["ProbeNumber"])
    averaged = (
        dfv.groupby(["ProbeVersion", "ProbeNumber"])[pca_cols]
        .mean()
        .reset_index()
        .sort_values(["ProbeVersion", "ProbeNumber"])
    )

    # Save per-video CSV
    save_csv = os.path.join(OUTPUT_DIR, f"{video}_Averaged_PCA_TimeSeries.csv")
    averaged.to_csv(save_csv, index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    for col in pca_cols:
        plt.plot(
            averaged["ProbeNumber"],
            averaged[col],
            marker="o",
            linewidth=2,
            label=col,
        )
    plt.title(f"{video} — PCA Component Trajectories")
    plt.xlabel("Probe Number")
    plt.ylabel("Average PCA Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_plot = os.path.join(OUTPUT_DIR, f"{video}_ComponentDynamics.png")
    plt.savefig(save_plot, dpi=300)
    plt.close()

    print(f"Saved: {video} averages → {save_csv}")

print("\nAll done! Averaged PCA time series saved in:")
print(OUTPUT_DIR)
