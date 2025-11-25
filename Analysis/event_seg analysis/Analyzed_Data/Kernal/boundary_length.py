import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ==========================================================
# CONFIG — match your KDE settings
# ==========================================================
KDE_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\Event_Lengths"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERCENTILE_CUTOFF = 90  # same threshold used for consensus boundaries
SPECIAL_11MIN = ["Movie Task-summer", "Movie Task-lms", "Movie Task-c4"]  # same special cases as before

# ==========================================================
# MAIN LOOP
# ==========================================================
results = []

for f in os.listdir(KDE_DIR):
    if not f.endswith("_kde_timeseries.csv"):
        continue

    video = f.replace("_kde_timeseries.csv", "")
    path = os.path.join(KDE_DIR, f)
    kde = pd.read_csv(path)

    # --- Handle duration logic (6 vs 11 min)
    if any(x in video for x in SPECIAL_11MIN):
        dur = 11 * 60
    else:
        dur = 6 * 60

    # --- Load and clean
    kde = kde.rename(columns={kde.columns[0]: "Time", kde.columns[1]: "BoundaryStrength"})
    kde["Time"] = pd.to_numeric(kde["Time"], errors="coerce")
    kde["BoundaryStrength"] = pd.to_numeric(kde["BoundaryStrength"], errors="coerce")
    kde = kde.dropna()

    # --- Apply cutoff and detect peaks
    cutoff = np.percentile(kde["BoundaryStrength"], PERCENTILE_CUTOFF)
    peaks, props = find_peaks(kde["BoundaryStrength"], height=cutoff)
    peak_times = kde["Time"].values[peaks]

    # --- Compute event lengths (time between consecutive peaks)
    if len(peak_times) > 1:
        diffs = np.diff(peak_times)
        mean_len = np.mean(diffs)
        median_len = np.median(diffs)
        std_len = np.std(diffs)
        n_events = len(diffs) + 1  # number of segments
    else:
        diffs = []
        mean_len = np.nan
        median_len = np.nan
        std_len = np.nan
        n_events = 0

    results.append({
        "Video": video,
        "Num_ConsensusBoundaries": len(peak_times),
        "Num_Events": n_events,
        "Mean_EventLength_s": mean_len,
        "Median_EventLength_s": median_len,
        "SD_EventLength_s": std_len
    })

    # --- Save per-video event lengths
    per_video_csv = os.path.join(OUTPUT_DIR, f"{video}_event_lengths.csv")
    pd.DataFrame({
        "BoundaryIndex": np.arange(1, len(peak_times) + 1),
        "BoundaryTime_s": peak_times,
        "IntervalToNext_s": np.append(diffs, np.nan)
    }).to_csv(per_video_csv, index=False)

    print(f"✅ {video}: {len(peak_times)} boundaries, mean event = {mean_len:.1f}s")

# ==========================================================
# Save summary
# ==========================================================
summary = pd.DataFrame(results)
summary.to_csv(os.path.join(OUTPUT_DIR, "event_length_summary.csv"), index=False)

print("\n🎬 Done! Summary saved to:")
print(os.path.join(OUTPUT_DIR, "event_length_summary.csv"))
