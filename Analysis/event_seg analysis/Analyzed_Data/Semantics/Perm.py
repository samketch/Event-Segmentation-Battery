import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import glob

# =========================
# Config
# =========================
SEMANTIC_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Semantics"
KDE_DIR      = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
VIDEO_DURATION = 22 * 60  # 22 minutes
N_PERM = 1000             # number of permutations

BASE_OUT = os.path.join(SEMANTIC_DIR, "permutation")
HIST_DIR = os.path.join(BASE_OUT, "perm_hists")
CORR_DIR = os.path.join(BASE_OUT, "perm_correlation")

os.makedirs(BASE_OUT, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)
# =========================

results = []

semantic_files = glob.glob(os.path.join(SEMANTIC_DIR, "*_aligned_semantic.csv"))

for sem_file in semantic_files:
    base = os.path.basename(sem_file)
    video_id = base.split("_")[0]  # e.g., "friends1"
    kde_file = os.path.join(KDE_DIR, f"{video_id}_kde_timeseries.csv")

    if not os.path.exists(kde_file):
        print(f"Skipping {video_id}: KDE file not found at {kde_file}")
        continue

    print(f"\n=== Processing {video_id} ===")

    # --- Load semantic data ---
    sem = pd.read_csv(sem_file)
    sem_time = pd.to_numeric(sem["time"], errors="coerce").values
    sem_shift = 1 - pd.to_numeric(sem["similarity_prev"], errors="coerce").fillna(1).values

    # --- Load KDE data ---
    kde = pd.read_csv(kde_file)
    kde_time = pd.to_numeric(kde["Time(s)"], errors="coerce").values
    kde_density = pd.to_numeric(kde["BoundaryDensity"], errors="coerce").values

    # --- Common time axis (1s bins) ---
    t = np.arange(0, VIDEO_DURATION+1, 1)
    sem_interp = np.interp(t, sem_time, sem_shift)
    kde_interp = np.interp(t, kde_time, kde_density)

    # --- Normalize (z-score) ---
    sem_z = (sem_interp - sem_interp.mean()) / sem_interp.std()
    kde_z = (kde_interp - kde_interp.mean()) / kde_interp.std()

    # --- Observed correlation ---
    r_obs, _ = pearsonr(sem_z, kde_z)
    print(f"Observed correlation: r={r_obs:.3f}")

    # --- Permutation test (circular shift) ---
    rng = np.random.default_rng()
    r_null = []
    for i in range(N_PERM):
        shift = rng.integers(low=1, high=len(sem_z))  # random shift
        sem_shifted = np.roll(sem_z, shift)
        r, _ = pearsonr(sem_shifted, kde_z)
        r_null.append(r)

    r_null = np.array(r_null)

    # --- Permutation p-value ---
    p_perm = (np.sum(r_null >= r_obs) + 1) / (N_PERM + 1)
    print(f"Permutation p-value = {p_perm:.4f} (N={N_PERM})")

    # --- Save results row ---
    results.append({
        "video": video_id,
        "r_obs": r_obs,
        "p_perm": p_perm
    })

    # --- Plot 1: null distribution ---
    plt.figure(figsize=(8,5))
    plt.hist(r_null, bins=40, alpha=0.7, color="gray", edgecolor="black")
    plt.axvline(r_obs, color="red", linestyle="--", linewidth=2, label=f"Observed r={r_obs:.3f}")
    plt.xlabel("Correlation (r)")
    plt.ylabel("Frequency")
    plt.title(f"{video_id} Permutation Test (N={N_PERM})\nObserved r={r_obs:.3f}, p={p_perm:.4f}")
    plt.legend()
    plt.tight_layout()
    out_fig_perm = os.path.join(HIST_DIR, f"{video_id}_permutation_hist.png")
    plt.savefig(out_fig_perm)
    plt.close()
    print(f"Saved null distribution plot: {out_fig_perm}")

    # --- Plot 2: overlapping timecourses ---
    plt.figure(figsize=(12,5))
    plt.plot(t, sem_z, label="Semantic shift (1 - similarity)", alpha=0.8)
    plt.plot(t, kde_z, label="KDE boundary density", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Z-scored value")
    plt.title(f"{video_id}: Semantic vs Boundary Density\nObserved r={r_obs:.2f}, p_perm={p_perm:.3g}")
    plt.legend()
    plt.tight_layout()
    out_fig_line = os.path.join(CORR_DIR, f"{video_id}_timecourse.png")
    plt.savefig(out_fig_line)
    plt.close()
    print(f"Saved timecourse plot: {out_fig_line}")

# --- Save summary table ---
out_csv = os.path.join(BASE_OUT, "permutation_summary.csv")
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\nSaved summary results to {out_csv}")
