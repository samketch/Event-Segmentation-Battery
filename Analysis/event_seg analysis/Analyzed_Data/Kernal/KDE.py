# ===============================================================
# Event Segmentation KDE Generator (with combined output)
# ===============================================================
# Computes per-video kernel density estimates and consensus boundaries,
# and also outputs a combined KDE timeline file for reliability analyses.
# ===============================================================

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import os

# ---- Parameters ----
input_csv = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\master_data.csv"
output_path = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"

dedup_window = 0.5       # Double-tap removal window (s)
kernel_sd = 2.5          # Gaussian kernel width (σ)
percentile_cutoff = 90   # Top X% of boundary density to define consensus
grid_res = 1.0           # Time resolution (s) for output grid

# ---- Load data ----
df = pd.read_csv(input_csv)
df = df[['ParticipantID', 'VideoName', 'BoundaryTime(s)']]

# ---- Remove double-taps ----
deduped_rows = []
for (pid, video), group in df.groupby(['ParticipantID', 'VideoName']):
    times = np.sort(group['BoundaryTime(s)'].values)
    if len(times) == 0:
        continue
    filtered_times = [times[0]]
    for t in times[1:]:
        if t - filtered_times[-1] >= dedup_window:
            filtered_times.append(t)
    for t in filtered_times:
        deduped_rows.append({'ParticipantID': pid, 'VideoName': video, 'BoundaryTime(s)': t})
df_dedup = pd.DataFrame(deduped_rows)

# ---- Ensure output folder exists ----
os.makedirs(output_path, exist_ok=True)

all_results = []          # Consensus results
all_kde_results = []      # Combined KDE time series

# ---- Process each video ----
for video_name, group in df_dedup.groupby('VideoName'):
    times = group['BoundaryTime(s)'].values
    if len(times) < 2:
        print(f"⚠️ Skipping {video_name}: not enough boundaries")
        continue

    # KDE with bandwidth = kernel_sd
    try:
        kde = gaussian_kde(times, bw_method=kernel_sd / np.std(times, ddof=1))
    except Exception as e:
        print(f"⚠️ Failed KDE for {video_name}: {e}")
        continue

    # Evaluate KDE across uniform grid (full movie range)
    t_min, t_max = times.min(), times.max()
    grid = np.arange(t_min, t_max, grid_res)
    density = kde(grid)
    density /= np.trapz(density, grid)  # normalize to integrate = 1

    # ---- Save per-video KDE ----
    kde_df = pd.DataFrame({'Time(s)': grid, 'BoundaryDensity': density})
    safe_name = video_name.replace(".mp4", "")
    if safe_name == "12_years":
        safe_name = "12 years_6m.mp4"
    elif safe_name == "500Days":
        safe_name = "Movie Task-summer"
    elif safe_name == "backToFuture":
        safe_name = "back to the future_6m.mp4"
    elif safe_name == "c4":
        safe_name = "Movie Task-c4"
    elif safe_name == "lms":
        safe_name = "Movie Task-lms"
    elif safe_name == "prestige":
        safe_name = "the_prestige_6m.mp4"
    elif safe_name == "pulpFiction":
        safe_name = "pulp_fiction_6m.mp4"
    elif safe_name == "shawshank":
        safe_name = "shawshank clip_6m.mp4"

    out_csv = os.path.join(output_path, f"{safe_name}_kde_timeseries.csv")
    kde_df.to_csv(out_csv, index=False)
    print(f"💾 Saved KDE timeseries: {out_csv}")

    # Add to combined list (include video name)
    kde_df['VideoName'] = safe_name
    all_kde_results.append(kde_df)

    # ---- Threshold at percentile ----
    cutoff = np.percentile(density, percentile_cutoff)
    peaks, _ = find_peaks(density, height=cutoff)
    consensus_times = grid[peaks]

    results = []
    for ct in consensus_times:
        results.append({
            'VideoName': safe_name,
            'ConsensusTime(s)': round(ct, 3),
            'PercentileCutoff': percentile_cutoff
        })
        all_results.append(results[-1])

    # ---- Save per-video consensus ----
    if results:
        out_df = pd.DataFrame(results)
        out_path = os.path.join(output_path, f"{safe_name}_consensus_boundaries.csv")
        out_df.to_csv(out_path, index=False)
        print(f"💾 Saved consensus: {out_path}")
    else:
        print(f"⚠️ No consensus events above threshold for {safe_name}")

# ---- Save combined consensus and KDE ----
if all_results:
    all_df = pd.DataFrame(all_results)
    combined_file = os.path.join(output_path, "all_videos_consensus_boundaries.csv")
    all_df.to_csv(combined_file, index=False)
    print(f"✅ Saved combined consensus file: {combined_file}")

if all_kde_results:
    all_kde_df = pd.concat(all_kde_results, ignore_index=True)
    combined_kde_file = os.path.join(output_path, "all_videos_kde_timeseries.csv")
    all_kde_df.to_csv(combined_kde_file, index=False)
    print(f"✅ Saved combined KDE file: {combined_kde_file}")
