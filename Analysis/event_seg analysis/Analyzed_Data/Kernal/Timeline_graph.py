import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# =========================
# Configuration
# =========================
MASTER_FILE   = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\master_data.csv"
OUTPUT_DIR    = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Graphs"
VIDEO_DURATION = 6 * 60        # 22 minutes = 1320 seconds
DEDUP_WINDOW   = 0.5            # remove double taps within this window
KERNEL_SD      = 2.5            # Gaussian kernel width (σ in seconds)
PERCENTILE_CUTOFF = 90          # threshold for consensus peaks (optional)
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Formatter for mm:ss ----
def seconds_to_mmss(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{minutes:02d}:{seconds:02d}"

time_formatter = FuncFormatter(seconds_to_mmss)

# ---- Load master ----
df = pd.read_csv(MASTER_FILE)[["ParticipantID", "VideoName", "BoundaryTime(s)"]].dropna()

# ---- Remove double taps within DEDUP_WINDOW per participant and video ----
dedup_rows = []
for (pid, vid), grp in df.groupby(["ParticipantID", "VideoName"]):
    t = np.sort(grp["BoundaryTime(s)"].values)
    if t.size == 0:
        continue
    kept = [t[0]]
    for x in t[1:]:
        if x - kept[-1] >= DEDUP_WINDOW:
            kept.append(x)
    for x in kept:
        dedup_rows.append({"ParticipantID": pid, "VideoName": vid, "Time": x})
df_clean = pd.DataFrame(dedup_rows)

# ---- Process per video ----
for video, g in df_clean.groupby("VideoName"):
    print(video)
    times = g["Time"].values
    if times.size == 0:
        print(f"No data for {video}")
        continue

    # KDE with bandwidth = KERNEL_SD
    bw = KERNEL_SD / np.std(times, ddof=1) if np.std(times, ddof=1) > 0 else 1.0
    kde = gaussian_kde(times, bw_method=bw)

    # Evaluate KDE across full video duration
    if video == "500Days.mp4":
        VIDEO_DURATION = 11 * 60
    elif video == "lms.mp4":
        VIDEO_DURATION = 11 * 60
    elif video == "c4.mp4":
        VIDEO_DURATION = 11 * 60
    else: 
        VIDEO_DURATION = 6 *60
    grid = np.linspace(0, VIDEO_DURATION, VIDEO_DURATION)  # 1 sample per second
    density = kde(grid)

    # Determine cutoff for consensus peaks
    cutoff = np.percentile(density, PERCENTILE_CUTOFF)
    peaks, _ = find_peaks(density, height=cutoff)

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(grid, density, color="steelblue", label="KDE density")
    plt.axhline(cutoff, color="red", linestyle="--", label=f"{PERCENTILE_CUTOFF}th percentile")
    plt.scatter(grid[peaks], density[peaks], color="darkorange", zorder=5, label="Consensus peaks")

    plt.xlabel("Time (mm:ss)")
    plt.ylabel("Boundary density")
    plt.title(f"KDE Identified Boundaries – {video}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Apply mm:ss formatting
    ax = plt.gca()
    ax.xaxis.set_major_formatter(time_formatter)
    ax.xaxis.set_major_locator(MultipleLocator(60)) #tick every 60 seconds
    ax.set_xlim(-59.9,VIDEO_DURATION+59.9) #force X axis limits of 0 and video duration
    plt.tight_layout()

    safe = video.replace(".mp4", "").replace(" ", "_")
    out = os.path.join(OUTPUT_DIR, f"{safe}_kde.png")
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved: {out}")

print("\nDone. KDE plots saved.")
