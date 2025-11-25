import os
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================
MASTER_DATA = r"C:\Users\Smallwood Lab\Downloads\Master_file.csv"  # your full mDES+Gradient file
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Gradient_timecourses_byRunTime"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD & PREP
# ==========================================================
df = pd.read_csv(MASTER_DATA)
grad_cols = [c for c in df.columns if c.startswith("gradient_")]
videos = df["Task_name"].dropna().unique()

print(f"Found {len(videos)} videos → {videos}")

# ==========================================================
# LOOP OVER MOVIES
# ==========================================================
for video in videos:
    dfv = df[df["Task_name"] == video].copy()

    # drop missing times or gradients
    dfv = dfv.dropna(subset=["Run_time"] + grad_cols)

    # average across participants at each Run_time
    avg_df = dfv.groupby("Run_time")[grad_cols].mean().reset_index()

    # rename for clarity
    avg_df = avg_df.rename(columns={c: f"Mean_{c}" for c in grad_cols})

    # save output
    out_csv = os.path.join(OUTPUT_DIR, f"{video}_GRADIENT_timecourse.csv")
    avg_df.to_csv(out_csv, index=False)
    print(f"✅ Saved {video} → {out_csv}")

print("\n🧠 All Gradient timecourses (Run_time-based) saved!")
