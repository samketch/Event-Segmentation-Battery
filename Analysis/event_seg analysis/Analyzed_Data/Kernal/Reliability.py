# ===============================================================
# GLOBAL Split-Half Reliability Across All Movies
# ===============================================================
# Combines boundaries from all videos, performs global split-half
# reliability analysis:
#   1. Split participants randomly into halves (N_SPLITS times)
#   2. Compute KDE per video for each half
#   3. Concatenate/average across videos -> global KDEs
#   4. Calculate:
#        - Split-half correlation
#        - Peak timing precision
#        - Peak overlap (Jaccard)
# ===============================================================

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
from scipy.signal import find_peaks
import random

# ---------------------------
# CONFIG
# ---------------------------
INPUT_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\master_data.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\Split-Half"
BANDWIDTH = 2.5       # KDE kernel width (σ, seconds)
DT = 1.0              # Temporal resolution
N_SPLITS = 100        # Number of random splits
PEAK_PROM = 0.002     # Minimum prominence for peaks
MATCH_TOL = 2.0       # Seconds tolerance for overlap

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(INPUT_FILE)
df = df[['ParticipantID', 'VideoName', 'BoundaryTime(s)']]

participants = df['ParticipantID'].unique()
videos = df['VideoName'].unique()

# ---------------------------
# FUNCTIONS
# ---------------------------
def compute_kde(times, grid):
    """Compute normalized KDE on given grid."""
    if len(times) < 2:
        return np.zeros_like(grid)
    kde = gaussian_kde(times, bw_method=BANDWIDTH / np.std(times, ddof=1))
    y = kde(grid)
    return y / np.trapz(y, grid)

def get_peaks(y, grid):
    """Extract KDE peak times."""
    p, _ = find_peaks(y, prominence=PEAK_PROM)
    return grid[p]

# ---------------------------
# PREP GLOBAL TIME GRID PER VIDEO
# ---------------------------
video_grids = {}
for v in videos:
    g = df[df["VideoName"] == v]
    t_min, t_max = g['BoundaryTime(s)'].min(), g['BoundaryTime(s)'].max()
    video_grids[v] = np.arange(t_min, t_max, DT)

# ---------------------------
# SPLIT-HALF RELIABILITY
# ---------------------------
split_rs = []
peak_diffs = []
overlaps = []

for _ in range(N_SPLITS):
    random.shuffle(participants)
    half1 = participants[:len(participants)//2]
    half2 = participants[len(participants)//2:]

    # Build concatenated global KDEs
    kde1_all, kde2_all = [], []
    time_all = []

    for v in videos:
        grid = video_grids[v]
        sub = df[df["VideoName"] == v]
        times1 = sub.loc[sub["ParticipantID"].isin(half1), "BoundaryTime(s)"].values
        times2 = sub.loc[sub["ParticipantID"].isin(half2), "BoundaryTime(s)"].values

        kde1 = compute_kde(times1, grid)
        kde2 = compute_kde(times2, grid)

        # Append with movie offset (preserve ordering)
        offset = len(time_all)
        kde1_all.extend(kde1)
        kde2_all.extend(kde2)
        time_all.extend(grid + offset * DT * 10000)  # ensures unique time axis per movie

    kde1_all, kde2_all = np.array(kde1_all), np.array(kde2_all)
    time_all = np.array(time_all)

    # Compute correlation
    r, _ = pearsonr(kde1_all, kde2_all)
    split_rs.append(r)

    # Peak metrics (on concatenated KDEs)
    p1, p2 = get_peaks(kde1_all, time_all), get_peaks(kde2_all, time_all)
    if len(p1) > 0 and len(p2) > 0:
        diffs = [np.min(np.abs(p1 - t)) for t in p2]
        peak_diffs.append(np.median(diffs))
        overlap_count = sum([np.any(np.abs(p1 - t) <= MATCH_TOL) for t in p2])
        union_count = len(np.unique(np.concatenate([p1, p2])))
        if union_count > 0:
            overlaps.append(overlap_count / union_count)

# ---------------------------
# OUTPUT SUMMARY
# ---------------------------
print("\n=== GLOBAL KDE STABILITY ===")
print(f"Participants: {len(participants)} across {len(videos)} movies")
print(f"Iterations: {N_SPLITS}")
print(f"Mean split-half r: {np.mean(split_rs):.3f} ± {np.std(split_rs):.2f}")
print(f"Median peak timing error: {np.median(peak_diffs):.2f} s")
print(f"Mean Jaccard overlap: {np.mean(overlaps):.3f}")
print("=============================")
