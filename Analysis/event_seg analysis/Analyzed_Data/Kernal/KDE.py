import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import os

# ---- Parameters ----
input_csv = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\master_data.csv"
output_path = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"

dedup_window = 0.5       # Double-tap removal window (s)
kernel_sd = 2.5          # Gaussian kernel width in seconds (σ)
percentile_cutoff = 90   # Only keep top X% of boundary density

# ---- Load data ----
df = pd.read_csv(input_csv)
df = df[['ParticipantID', 'VideoName', 'BoundaryTime(s)']]

# ---- Remove double-taps ----
deduped_rows = []
for (pid, video), group in df.groupby(['ParticipantID', 'VideoName']):
    times = group['BoundaryTime(s)'].sort_values().values
    filtered_times = []
    if len(times) > 0:
        filtered_times.append(times[0])
        for t in times[1:]:
            if t - filtered_times[-1] >= dedup_window:
                filtered_times.append(t)
    for t in filtered_times:
        deduped_rows.append({'ParticipantID': pid,
                             'VideoName': video,
                             'BoundaryTime(s)': t})

df_dedup = pd.DataFrame(deduped_rows)

# ---- Ensure output folder exists ----
os.makedirs(output_path, exist_ok=True)

all_results = []

# ---- Process each video ----
for video_name, group in df_dedup.groupby('VideoName'):
    times = group['BoundaryTime(s)'].values

    # KDE with bandwidth = kernel_sd
    kde = gaussian_kde(times, bw_method=kernel_sd/np.std(times, ddof=1))

    # Evaluate KDE across time range
    t_min, t_max = times.min(), times.max()
    grid = np.linspace(t_min, t_max, 1000)  # dense sampling
    density = kde(grid)

    # ---- Save KDE timeline ----
    kde_df = pd.DataFrame({
        'Time(s)': grid,
        'BoundaryDensity': density
    })
    safe_name = video_name.replace(".mp4", "").replace(" ", "_")
    kde_out_path = os.path.join(output_path, f"{safe_name}_kde_timeseries.csv")
    kde_df.to_csv(kde_out_path, index=False)
    print(f"Saved KDE timeseries: {kde_out_path}")

    # ---- Threshold at percentile ----
    cutoff = np.percentile(density, percentile_cutoff)
    peaks, _ = find_peaks(density, height=cutoff)
    consensus_times = grid[peaks]

    results = []
    for ct in consensus_times:
        results.append({
            'VideoName': video_name,
            'ConsensusTime(s)': round(ct, 3),
            'PercentileCutoff': percentile_cutoff
        })
        all_results.append(results[-1])

    # ---- Save per-video consensus ----
    if results:
        out_df = pd.DataFrame(results)
        out_path = os.path.join(output_path, f"{safe_name}_consensus_boundaries.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved consensus: {out_path}")
    else:
        print(f"No consensus events above threshold for {video_name}")

# ---- Save combined consensus file ----
if all_results:
    all_df = pd.DataFrame(all_results)
    combined_file = os.path.join(output_path, "all_videos_consensus_boundaries.csv")
    all_df.to_csv(combined_file, index=False)
    print(f"Saved combined file: {combined_file}")
