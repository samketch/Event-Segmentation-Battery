import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import glob

# =========================
# Config
# =========================
SEMANTIC_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Semantics"
KDE_DIR      = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"

VIDEO_DURATION_DEFAULT = 6 * 60
N_PERM = 1000

BASE_OUT = os.path.join(SEMANTIC_DIR, "Combined_Semantics")
os.makedirs(BASE_OUT, exist_ok=True)

# =========================
# Load and combine all videos
# =========================

semantic_files = glob.glob(os.path.join(SEMANTIC_DIR, "*_aligned_semantic.csv"))
all_sem = []
all_kde = []

for sem_file in semantic_files:
    base = os.path.basename(sem_file)
    video_id = base.split("_")[0]

    if "12" in video_id:
        video_id = "12 years_6m.mp4"
        VIDEO_DURATION = VIDEO_DURATION_DEFAULT
    elif "500Days" in video_id:
        video_id = "Movie Task-summer"
        VIDEO_DURATION = 11 * 60
    elif "backToFuture" in video_id:
        video_id = "back to the future_6m.mp4"
        VIDEO_DURATION = VIDEO_DURATION_DEFAULT
    elif "c4" in video_id:
        video_id = "Movie Task-c4"
        VIDEO_DURATION = 11 * 60
    elif "lms" in video_id:
        video_id = "Movie Task-lms"
        VIDEO_DURATION = 11 * 60
    elif "prestige" in video_id:
        video_id = "the_prestige_6m.mp4"
        VIDEO_DURATION = VIDEO_DURATION_DEFAULT
    elif "pulpFiction" in video_id:
        video_id = "pulp_fiction_6m.mp4"
        VIDEO_DURATION = VIDEO_DURATION_DEFAULT
    elif "shawshank" in video_id:
        video_id = "shawshank clip_6m.mp4"
        VIDEO_DURATION = VIDEO_DURATION_DEFAULT
    else:
        print(f"⚠️ Skipping unrecognized video: {video_id}")
        continue

    kde_file = os.path.join(KDE_DIR, f"{video_id}_kde_timeseries.csv")
    if not os.path.exists(kde_file):
        print(f"⚠️ KDE not found for {video_id}")
        continue

    print(f"✅ Including {video_id}")

    # --- Load semantic data ---
    sem = pd.read_csv(sem_file)
    sem_time = pd.to_numeric(sem["time"], errors="coerce").values
    sem_shift = 1 - pd.to_numeric(sem["similarity_prev"], errors="coerce").fillna(1).values

    # --- Load KDE data ---
    kde = pd.read_csv(kde_file)
    kde_time = pd.to_numeric(kde["Time(s)"], errors="coerce").values
    kde_density = pd.to_numeric(kde["BoundaryDensity"], errors="coerce").values

    # --- Interpolate to 1s bins ---
    t = np.arange(0, VIDEO_DURATION + 1, 1)
    sem_interp = np.interp(t, sem_time, sem_shift)
    kde_interp = np.interp(t, kde_time, kde_density)

    # --- Normalize (z-score) within each video ---
    sem_z = (sem_interp - sem_interp.mean()) / sem_interp.std()
    kde_z = (kde_interp - kde_interp.mean()) / kde_interp.std()

    # --- Store for concatenation ---
    all_sem.append(sem_z)
    all_kde.append(kde_z)

# =========================
# Combine across all videos
# =========================
sem_all = np.concatenate(all_sem)
kde_all = np.concatenate(all_kde)
t_all = np.arange(len(sem_all))

print(f"\nCombined data: {len(sem_all)} total seconds across {len(all_sem)} videos")

# --- Global observed correlation ---
r_obs, _ = pearsonr(sem_all, kde_all)
print(f"🌎 Observed correlation (global): r = {r_obs:.3f}")

# =========================
# Permutation test (circular shift)
# =========================
rng = np.random.default_rng()
r_null = []
for i in range(N_PERM):
    shift = rng.integers(low=1, high=len(sem_all))
    sem_shifted = np.roll(sem_all, shift)
    r, _ = pearsonr(sem_shifted, kde_all)
    r_null.append(r)

r_null = np.array(r_null)
p_perm = (np.sum(r_null >= r_obs) + 1) / (N_PERM + 1)

print(f"Permutation p-value = {p_perm:.4f} (N={N_PERM})")

# =========================
# Plot results
# =========================
plt.figure(figsize=(8,5))
plt.hist(r_null, bins=40, alpha=0.7, color="gray", edgecolor="black")
plt.axvline(r_obs, color="red", linestyle="--", linewidth=2, label=f"Observed r={r_obs:.3f}")
plt.xlabel("Correlation (r)")
plt.ylabel("Frequency")
plt.title(f"Global Permutation Test (N={N_PERM})\nObserved r={r_obs:.3f}, p={p_perm:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUT, "global_permutation_hist.png"))
plt.close()

# --- Plot global overlapping timecourses ---
plt.figure(figsize=(14,5))
plt.plot(t_all, sem_all, label="Semantic shift (1 - similarity)", alpha=0.8)
plt.plot(t_all, kde_all, label="KDE boundary density", alpha=0.8)
plt.xlabel("Time (s, concatenated across videos)")
plt.ylabel("Z-scored value")
plt.title(f"Global Semantic vs Boundary Density\nObserved r={r_obs:.2f}, p_perm={p_perm:.3g}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUT, "global_timecourse.png"))
plt.close()

# --- Save results summary ---
out_csv = os.path.join(BASE_OUT, "global_permutation_summary.csv")
pd.DataFrame([{"r_obs": r_obs, "p_perm": p_perm, "N_perm": N_PERM, "n_videos": len(all_sem)}]).to_csv(out_csv, index=False)
print(f"\n✅ Saved global permutation summary to {out_csv}")
