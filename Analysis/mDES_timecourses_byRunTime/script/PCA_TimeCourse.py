import os
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================
MASTER_DATA = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv"  # your full mDES+PCA file
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD & PREP
# ==========================================================
df = pd.read_csv(MASTER_DATA)
pca_cols = [c for c in df.columns if c.startswith("PCA_")]
videos = df["Task_name"].dropna().unique()

print(f"Found {len(videos)} videos → {videos}")

# ==========================================================
# LOOP OVER MOVIES
# ==========================================================
for video in videos:
    dfv = df[df["Task_name"] == video].copy()

    # drop missing times or components
    dfv = dfv.dropna(subset=["Run_time"] + pca_cols)

    # average across participants at each Run_time
    avg_df = dfv.groupby("Run_time")[pca_cols].mean().reset_index()

    # rename for clarity
    avg_df = avg_df.rename(columns={c: f"Mean_{c}" for c in pca_cols})

    # save output
    out_csv = os.path.join(OUTPUT_DIR, f"{video}_PCA_timecourse.csv")
    avg_df.to_csv(out_csv, index=False)
    print(f"✅ Saved {video} → {out_csv}")

print("\n🎬 All PCA timecourses (Run_time-based) saved!")
