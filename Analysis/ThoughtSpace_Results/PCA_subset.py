import os
import pandas as pd
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================
MASTER = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_perVideo_averages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(MASTER)

# Keep only Movie Task rows (if you have multiple tasks)
#df = df[df["Task_name"].str.contains("movie", case=False, na=False)]

# Identify PCA columns
pca_cols = [c for c in df.columns if c.startswith("PCA_")]
print(f"PCA columns found: {pca_cols}")

# Define full grid of possible ProbeVersions and ProbeNumbers
probe_versions = range(0, 16)  # 0–15
probe_numbers = range(0, 6)    # 0–5

# ==========================================================
# LOOP OVER VIDEOS
# ==========================================================
videos = df["Task_name"].dropna().unique()
print(f"\nFound {len(videos)} videos: {videos}")

for video in videos:
    dsv = df[df["Task_name"] == video]

    # Average PCA components per ProbeVersion × ProbeNumber
    avg = (
        dsv.groupby(["ProbeVersion", "ProbeNumber"])[pca_cols]
        .mean()
        .reset_index()
    )

    # Create complete 16×6 grid (fill missing with NaN)
    grid = pd.MultiIndex.from_product(
        [probe_versions, probe_numbers],
        names=["ProbeVersion", "ProbeNumber"]
    )
    grid_df = pd.DataFrame(index=grid).reset_index()

    merged = pd.merge(grid_df, avg, on=["ProbeVersion", "ProbeNumber"], how="left")
    merged.columns = ["ProbeVersion", "ProbeNumber"] + [f"Average_{c}" for c in pca_cols]

    # Save per-video file
    save_path = os.path.join(OUTPUT_DIR, f"{video}_averages.csv")
    merged.to_csv(save_path, index=False)
    print(f"✅ Saved: {save_path}")

print("\nAll done — one averaged PCA file per video created.")
