import os
import pandas as pd
import glob

# ==========================================================
# CONFIG
# ==========================================================
MDES_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
GRAD_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Gradient_timecourses_byRunTime"
KDE_DIR  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"

OUTPUT = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All.csv"
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ==========================================================
# HELPER: Load folder into dict of {video_name: dataframe}
# ==========================================================
def load_timecourses(folder, suffix):
    data_dict = {}
    for f in glob.glob(os.path.join(folder, "*.csv")):
        name = os.path.basename(f).replace(suffix, "").replace(".csv", "")
        df = pd.read_csv(f)
        data_dict[name] = df
    print(f"✅ Loaded {len(data_dict)} files from {folder}")
    return data_dict

# ==========================================================
# LOAD DATASETS
# ==========================================================
mdes_data = load_timecourses(MDES_DIR, "_PCA_timecourse")
grad_data = load_timecourses(GRAD_DIR, "_GRADIENT_timecourse")

# KDE naming varies — use generic loader
kde_data = {}
for f in glob.glob(os.path.join(KDE_DIR, "*.csv")):
    name = os.path.basename(f).replace("_kde_timeseries", "").replace(".csv", "")
    df = pd.read_csv(f)
    kde_data[name] = df
print(f"✅ Loaded {len(kde_data)} KDE files")

# ==========================================================
# MERGE ACROSS SOURCES
# ==========================================================
merged_list = []

for video in mdes_data.keys():
    mdes_df = mdes_data[video]
    grad_df = grad_data.get(video)
    kde_df = kde_data.get(video)

    if grad_df is None:
        print(f"⚠️ Skipping {video}: missing gradient file")
        continue

    if kde_df is None:
        print(f"⚠️ Skipping {video}: missing KDE file")
        continue

    # Harmonize time column names
    if "Run_time" in kde_df.columns:
        kde_df = kde_df.rename(columns={"Run_time": "Time_rounded"})
    if "Time_rounded" not in kde_df.columns:
        kde_df = kde_df.rename(columns={kde_df.columns[0]: "Time_rounded"})

    # Merge on nearest second (round if needed)
    mdes_df["Run_time"] = mdes_df["Run_time"].round()
    grad_df["Run_time"] = grad_df["Run_time"].round()
    kde_df["Time_rounded"] = kde_df["Time_rounded"].round()

    # Merge
    merged = (
        mdes_df.merge(grad_df, on="Run_time", suffixes=("_PCA", "_GRAD"))
               .merge(kde_df, left_on="Run_time", right_on="Time_rounded", how="outer")
    )
    merged["Task_name"] = video
    merged_list.append(merged)

# ==========================================================
# CONCATENATE & SAVE
# ==========================================================
if merged_list:
    master = pd.concat(merged_list, ignore_index=True)
    master.to_csv(OUTPUT, index=False)
    print(f"\n🎉 Master file saved → {OUTPUT}")
    print(f"Total rows: {len(master)} | Columns: {len(master.columns)}")
else:
    print("❌ No data merged — check your input folders.")

kde_all = pd.concat(kde_data.values(), ignore_index=True)

# Find the time column dynamically
possible_time_cols = [c for c in kde_all.columns if "time" in c.lower()]
if possible_time_cols:
    time_col = possible_time_cols[0]
    print(f"Using KDE time column: {time_col}")
    kde_all = kde_all.rename(columns={time_col: "Time_rounded"})
else:
    raise ValueError("No time-like column found in KDE data!")

master_times = master['Run_time'].unique()
kde_times = kde_all['Time_rounded'].unique()

extra_times = sorted(set(master_times) - set(kde_times))
print("Extra times not in KDE:", extra_times)



