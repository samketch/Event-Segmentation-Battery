import os
import glob
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================
MASTER_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All.csv"
KDE_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_TRIMMED = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All_trimmed.csv"

# ==========================================================
# LOAD MASTER FILE
# ==========================================================
print(f"🔹 Loading master file: {MASTER_FILE}")
master = pd.read_csv(MASTER_FILE)
print(f"Loaded {len(master)} rows, {len(master.columns)} columns")

# ==========================================================
# LOAD ALL KDE TIMES FROM SOURCE FILES
# ==========================================================
print(f"\n🔹 Collecting all KDE timepoints from: {KDE_DIR}")
kde_times_all = []
for f in glob.glob(os.path.join(KDE_DIR, "*.csv")):
    df = pd.read_csv(f)
    # Dynamically detect time column
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if not time_cols:
        continue
    time_col = time_cols[0]
    kde_times_all.extend(df[time_col].round().tolist())
kde_times_all = sorted(set(kde_times_all))
print(f"✅ Found {len(kde_times_all)} unique KDE timepoints across all videos")

# ==========================================================
# FIND ROWS WITH NO KDE DATA
# ==========================================================
fake_rows = master[master['BoundaryDensity'].isna()]
print(f"\n🔍 Found {len(fake_rows)} rows with missing KDE data (possible fake boundaries)")

# ==========================================================
# GROUP BY VIDEO
# ==========================================================
if len(fake_rows) > 0:
    print("\n🧠 Breakdown by video:")
    fake_summary = fake_rows.groupby("Task_name")["Run_time"].apply(list)
    for vid, times in fake_summary.items():
        print(f"  {vid}: {len(times)} rows → {times[:10]}{'...' if len(times) > 10 else ''}")

# ==========================================================
# CHECK IF THESE TIMES EXIST IN TRUE KDE FILES
# ==========================================================
nonexistent = fake_rows[~fake_rows["Run_time"].isin(kde_times_all)]
print(f"\n⚠️ {len(nonexistent)} of these rows occur at times not found in any KDE file (i.e., merge artifacts).")

# ==========================================================
# SAVE TRIMMED MASTER FILE
# ==========================================================
if len(nonexistent) > 0:
    clean_master = master[~master.index.isin(nonexistent.index)]
    clean_master.to_csv(OUTPUT_TRIMMED, index=False)
    print(f"✅ Trimmed master file saved → {OUTPUT_TRIMMED}")
    print(f"   {len(clean_master)} rows remain after cleanup.")
else:
    print("✅ No fake KDE rows detected — master file already clean.")

print("\n🎬 Done.")
