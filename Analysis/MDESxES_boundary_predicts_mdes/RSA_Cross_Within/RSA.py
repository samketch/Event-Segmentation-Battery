# ============================================================================
# Within–Across Event RSA Analyses for GLOBAL PCA RSAs
# Author: Sam Ketcheson
# ============================================================================
"""
Runs bootstrap, middlepoint, and fake-boundary tests on global PCA RSA matrices.

Inputs:
  - global_Mean_PCA_*_RSA_matrix.npy : cosine similarity matrices
  - all_videos_consensus_boundaries.csv : boundaries per movie (seconds)

Outputs:
  RSA_Cross_Within/Within_Across_Bootstrap/PCA_#/
      Bootstrap/, Middlepoint/, Fake_Boundary/
      Summary CSVs and histograms
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
RSA_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\RSA_Cross_Within"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"

STEP_SIZE = 15
WINDOW_SIZE = 45
BOOTSTRAPS = 1000
COMBINE_LEFT_RIGHT = True
SHIFT_RADIUS = 0
MIDPOINT_FRAC = 0.25

# ----------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------
def load_boundaries(epsilon=1e-3):
    """Load and normalize pooled consensus boundaries from all videos."""
    path = os.path.join(BOUNDARY_DIR, "all_videos_consensus_boundaries.csv")
    if not os.path.exists(path):
        print("⚠️ No all_videos_consensus_boundaries.csv found.")
        return None

    df = pd.read_csv(path)
    col_time = next((c for c in df.columns if "consensus" in c.lower() and "time" in c.lower()), "ConsensusTime(s)")
    col_video = next((c for c in df.columns if "video" in c.lower()), "VideoName")

    df = df[[col_video, col_time]].dropna()
    df[col_time] = pd.to_numeric(df[col_time], errors="coerce")
    df = df.dropna(subset=[col_time])

    # normalize per-video to [0,1]
    durations = df.groupby(col_video)[col_time].transform("max").replace(0, np.nan)
    df = df[durations.notna()]
    props = (df[col_time] / durations).clip(0, 1).to_numpy()

    props = np.sort(props)
    deduped = [props[0]]
    for x in props[1:]:
        if abs(x - deduped[-1]) > epsilon:
            deduped.append(x)
    props = np.array(deduped)

    print(f"[global] Loaded {len(props)} normalized boundaries in [0,1]. "
          f"Min={props.min():.3f}, Max={props.max():.3f}")
    return props


def bootstrap_across(vals, n_samples, bootstraps=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.array([])
    return np.array([np.mean(rng.choice(vals, size=n_samples, replace=True)) for _ in range(bootstraps)])


def upper_triangle_mean(block, min_lag=0):
    if block.size == 0:
        return np.nan, 0
    iu = np.triu_indices_from(block, k=min_lag)
    if iu[0].size == 0:
        return np.nan, 0
    return np.mean(block[iu]), iu[0].size


def compute_middle_point_values(M, start, end, n_time, shift_radius=0, frac=0.25, seg_len=5):
    length = end - start
    if length < (seg_len * 2 + 1):
        return None
    half_len = length // 2
    d = max(1, int(np.floor(half_len * frac)))
    center = start + half_len
    min_sep = int(np.ceil(WINDOW_SIZE / STEP_SIZE))
    d = max(d, min_sep)
    clamp = lambda a, lo, hi: max(lo, min(hi, a))

    within_vals, across_vals = [], []
    for s in range(-shift_radius, shift_radius + 1):
        seg1_start = clamp(center - d - seg_len + s, start, end - seg_len)
        seg1_end = seg1_start + seg_len
        seg2_start = clamp(center + d + s, start, end - seg_len)
        seg2_end = seg2_start + seg_len
        within_block = M[seg1_start:seg1_end, seg2_start:seg2_end]
        if within_block.size > 0:
            within_vals.append(np.mean(within_block))

        b = end
        seg3_start = clamp(b - d - seg_len + s, 0, n_time - seg_len)
        seg3_end = seg3_start + seg_len
        seg4_start = clamp(b + d + s, 0, n_time - seg_len)
        seg4_end = seg4_start + seg_len
        if seg4_end <= n_time:
            across_block = M[seg3_start:seg3_end, seg4_start:seg4_end]
            if across_block.size > 0:
                across_vals.append(np.mean(across_block))

    if not within_vals:
        return None
    return {
        "within_value": float(np.mean(within_vals)),
        "across_value": float(np.mean(across_vals)) if across_vals else np.nan,
        "lag_used": int(d),
        "center_idx": int(center),
        "segment_length": seg_len,
        "type": "segment_middle"
    }


def compute_fake_boundary_midpoint_values(M, start, end, next_start, n_time):
    length = end - start
    if length < 4:
        return None
    mid = start + length // 2
    mid1 = start + (mid - start) // 2
    mid2 = mid + (end - mid) // 2
    patch, half = 5, 2
    r1 = slice(max(0, mid1 - half), min(n_time, mid1 + half + 1))
    r2 = slice(max(0, mid2 - half), min(n_time, mid2 + half + 1))
    within_value = float(np.mean(M[r1, r2]))
    across_value = np.nan
    if next_start is not None and next_start < n_time:
        next_mid = next_start + (end - start) // 4
        r_next = slice(max(0, next_mid - half), min(n_time, next_mid + half + 1))
        if mid2 < n_time and next_mid < n_time:
            across_value = float(np.mean(M[r2, r_next]))
    return {
        "within_value": within_value,
        "across_value": across_value,
        "difference": within_value - across_value,
        "mid1": mid1,
        "mid2": mid2,
        "type": "fake_boundary"
    }

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
summary, summary_middle = [], []
boundaries = load_boundaries()
if boundaries is None:
    raise RuntimeError("No valid boundaries loaded.")

for file in os.listdir(RSA_DIR):
    if not (file.endswith("_RSA_matrix.npy") and file.startswith("global_")):
        continue

    base = file.replace("_RSA_matrix.npy", "")
    component = base.split("_")[-2]  # e.g., PCA_1

    print(f"\nAnalyzing {base}...")

    OUT_ROOT = os.path.join(RSA_DIR, "Within_Across_Bootstrap", component)
    BOOTSTRAP_DIR = os.path.join(OUT_ROOT, "Bootstrap")
    MIDDLEPOINT_DIR = os.path.join(OUT_ROOT, "Middlepoint")
    FAKEBOUNDARY_DIR = os.path.join(OUT_ROOT, "Fake_Boundary")
    for d in [OUT_ROOT, BOOTSTRAP_DIR, MIDDLEPOINT_DIR, FAKEBOUNDARY_DIR]:
        os.makedirs(d, exist_ok=True)

    M = np.load(os.path.join(RSA_DIR, file))
    n_time = M.shape[0]

    indices = np.round(boundaries * n_time).astype(int)
    indices = indices[(indices > 0) & (indices < n_time)]
    indices = np.unique(np.append(indices, n_time))

    print(f"{base}: using {len(indices)-1} events (n_time={n_time}).")

    event_starts = np.concatenate(([0], indices[:-1]))
    event_ends = indices
    event_results = []

    for e, (start, end) in enumerate(zip(event_starts, event_ends)):
        if end - start < 1:
            continue
        within_block = M[start:end, start:end]
        within_mean, n_pairs = upper_triangle_mean(within_block)
        if n_pairs == 0 or np.isnan(within_mean):
            continue

        right_block = M[start:end, end:n_time]
        left_block = M[0:start, start:end]
        across_vals = (np.concatenate([left_block.flatten(), right_block.flatten()])
                       if COMBINE_LEFT_RIGHT else right_block.flatten())

        if across_vals.size == 0:
            z, p, across_mean = np.nan, np.nan, np.nan
        else:
            boot = bootstrap_across(across_vals, n_pairs, BOOTSTRAPS)
            across_mean = np.mean(boot)
            z = (within_mean - np.mean(boot)) / np.std(boot)
            p = 1 - percentileofscore(boot, within_mean) / 100.0

        event_results.append({
            "video": base,
            "event_index": e + 1,
            "start_idx": start,
            "end_idx": end,
            "within_mean": within_mean,
            "across_mean": across_mean,
            "z": z,
            "p": p,
            "n_pairs": n_pairs
        })

        mp = compute_middle_point_values(M, start, end, n_time,
                                         shift_radius=SHIFT_RADIUS,
                                         frac=MIDPOINT_FRAC)
        if mp is not None:
            summary_middle.append({"video": base, "event_index": e + 1, **mp})

        next_start = event_starts[e + 1] if e + 1 < len(event_starts) else None
        fb = compute_fake_boundary_midpoint_values(M, start, end, next_start, n_time)
        if fb is not None:
            summary_middle.append({"video": base, "event_index": e + 1, **fb})

    if event_results:
        df = pd.DataFrame(event_results)
        df.to_csv(os.path.join(BOOTSTRAP_DIR, f"{base}_within_across_bootstrap.csv"), index=False)
        summary.append(df)
        print(f"{len(df)} events analyzed and saved for {base}.")

# ----------------------------------------------------------------------------
# SAVE + HISTOGRAMS
# ----------------------------------------------------------------------------
if summary:
    df_all = pd.concat(summary, ignore_index=True)
    df_all.to_csv(os.path.join(RSA_DIR, "Within_Across_Bootstrap", "global_within_across_summary.csv"), index=False)
    print("✅ Combined bootstrap summary saved.")

if summary_middle:
    df_middle = pd.DataFrame(summary_middle)
    df_middle.to_csv(os.path.join(RSA_DIR, "Within_Across_Bootstrap", "global_middlepoint_summary.csv"), index=False)
    print("✅ Middle point summary saved.")

    df_fake = df_middle[df_middle["type"] == "fake_boundary"].copy()
    if not df_fake.empty:
        keep = ["video", "event_index", "within_value", "across_value", "difference", "mid1", "mid2", "type"]
        df_fake = df_fake[[c for c in keep if c in df_fake.columns]]
        df_fake.to_csv(os.path.join(RSA_DIR, "Within_Across_Bootstrap", "global_fakeboundary_summary.csv"), index=False)
        print("✅ Fake-boundary summary saved.")

    # -- Histograms per type per video
    for vid in df_middle["video"].unique():
        subset = df_middle[df_middle["video"] == vid]
        for t in subset["type"].unique():
            sub = subset[subset["type"] == t]
            diff = sub["within_value"] - sub["across_value"]
            plt.hist(diff.dropna(), bins=30, edgecolor="k")
            plt.xlabel("Within - Across similarity")
            plt.ylabel("Count")
            plt.title(f"{vid}: {t}")
            plt.tight_layout()
            save_dir = (FAKEBOUNDARY_DIR if "fake" in t.lower()
                        else MIDDLEPOINT_DIR if "middle" in t.lower()
                        else BOOTSTRAP_DIR)
            plt.savefig(os.path.join(save_dir, f"{vid}_{t}_difference_distribution.png"))
            plt.close()

    # -- Combined histograms
    if not df_fake.empty:
        plt.hist(df_fake["difference"].dropna(), bins=30, edgecolor="k")
        plt.xlabel("Within - Across similarity")
        plt.ylabel("Count")
        plt.title("Combined Fake-Boundary Differences")
        plt.tight_layout()
        plt.savefig(os.path.join(RSA_DIR, "Within_Across_Bootstrap", "combined_fakeboundary_difference_distribution.png"))
        plt.close()

    df_middlepoint = df_middle[df_middle["type"] == "segment_middle"].copy()
    if not df_middlepoint.empty:
        diff = df_middlepoint["within_value"] - df_middlepoint["across_value"]
        plt.hist(diff.dropna(), bins=30, edgecolor="k")
        plt.xlabel("Within - Across similarity")
        plt.ylabel("Count")
        plt.title("Combined Middle-Point Differences")
        plt.tight_layout()
        plt.savefig(os.path.join(RSA_DIR, "Within_Across_Bootstrap", "combined_middlepoint_difference_distribution.png"))
        plt.close()

    # Bootstrap combined histogram
    all_boot_path = os.path.join(RSA_DIR, "Within_Across_Bootstrap", "global_within_across_summary.csv")
    if os.path.exists(all_boot_path):
        df_boot = pd.read_csv(all_boot_path)
        if not df_boot.empty:
            diff = df_boot["within_mean"] - df_boot["across_mean"]
            plt.hist(diff.dropna(), bins=30, edgecolor="k")
            plt.xlabel("Within - Across similarity")
            plt.ylabel("Count")
            plt.title("Combined Bootstrap Differences")
            plt.tight_layout()
            plt.savefig(os.path.join(RSA_DIR, "Within_Across_Bootstrap", "combined_bootstrap_difference_distribution.png"))
            plt.close()

print("\n✅ Global PCA RSA within–across analysis complete (with histograms).")
