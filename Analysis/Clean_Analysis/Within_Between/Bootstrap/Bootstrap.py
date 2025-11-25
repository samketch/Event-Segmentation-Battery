# =============================================================================
# PCA_RSA_within_across_bootstrap_v5.py
#
# Performs within/across bootstrap analysis with lag-matched sampling:
#   1) Per video (output: Bootstrap/<PCA>/<video>_bootstrap.csv)
#   2) Global per PCA (output: Bootstrap/Global_Bootstrap/<PCA>/global_bootstrap.csv)
#
# This version removes temporal autocorrelation bias by matching lags
# between within-event and across-event pairs.
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
RSA_ROOT = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\PCA_RSA"
BOUNDARY_FILE = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\Bootstrap"

RUN_BOOTSTRAP = True
BOOTSTRAPS = 1000
COMBINE_LEFT_RIGHT = True

# Lag filter
MIN_LAG = 1    # exclude diagonal (lag 0)
MAX_LAG = None # None means unlimited; set to 10 to restrict

# Directories
BOOTSTRAP_ROOT = os.path.join(OUTPUT_DIR, "Bootstrap")
GLOBAL_ROOT = os.path.join(BOOTSTRAP_ROOT, "Global_Bootstrap")
os.makedirs(BOOTSTRAP_ROOT, exist_ok=True)
os.makedirs(GLOBAL_ROOT, exist_ok=True)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_boundaries(video_name):
    if not os.path.exists(BOUNDARY_FILE):
        print("  ⚠️ Boundary file not found.")
        return None
    df = pd.read_csv(BOUNDARY_FILE)
    df.columns = [c.strip() for c in df.columns]
    df["VideoName_norm"] = (
        df[df.columns[0]]
        .astype(str)
        .str.strip()
        .str.replace("\u00A0", " ", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.lower()
    )
    video_norm = (
        str(video_name)
        .strip()
        .replace("\u00A0", " ")
        .replace("\xa0", " ")
        .lower()
    )
    mask = df["VideoName_norm"] == video_norm
    vals = df.loc[mask, [c for c in df.columns if "time" in c.lower()][0]].dropna().to_numpy()
    return np.sort(vals) if len(vals) > 0 else None


def pairs_by_lag_within(M, start, end, min_lag=1, max_lag=None):
    vals_by_lag = {}
    for i in range(start, end):
        for j in range(i + min_lag, end):
            lag = j - i
            if max_lag is not None and lag > max_lag:
                continue
            v = M[i, j]
            if not np.isnan(v):
                vals_by_lag.setdefault(lag, []).append(v)
    return vals_by_lag


def pairs_by_lag_across(M, start, end, n_time, min_lag=1, max_lag=None):
    vals_by_lag = {}
    # left side (earlier)
    for i in range(0, start):
        for j in range(start, end):
            lag = abs(j - i)
            if lag < min_lag:
                continue
            if max_lag is not None and lag > max_lag:
                continue
            v = M[i, j] if i < j else M[j, i]
            if not np.isnan(v):
                vals_by_lag.setdefault(lag, []).append(v)
    # right side (later)
    for i in range(start, end):
        for j in range(end, n_time):
            lag = abs(j - i)
            if lag < min_lag:
                continue
            if max_lag is not None and lag > max_lag:
                continue
            v = M[i, j] if i < j else M[j, i]
            if not np.isnan(v):
                vals_by_lag.setdefault(lag, []).append(v)
    return vals_by_lag


def lag_matched_bootstrap(across_by_lag, within_counts_by_lag, bootstraps=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    usable = {}
    for lag, cnt in within_counts_by_lag.items():
        pool = across_by_lag.get(lag, [])
        if len(pool) >= 1:
            usable[lag] = (cnt, np.asarray(pool))
    if not usable:
        return np.array([])

    boot = np.empty(bootstraps, dtype=float)
    lags = list(usable.keys())

    for b in range(bootstraps):
        acc = []
        for lag in lags:
            cnt, pool = usable[lag]
            draw = rng.choice(pool, size=cnt, replace=True)
            acc.append(draw)
        acc = np.concatenate(acc)
        boot[b] = np.mean(acc) if acc.size else np.nan

    boot = boot[~np.isnan(boot)]
    return boot


def mean_upper_triangle(block, min_lag=1, max_lag=None):
    if block.size == 0:
        return np.nan, 0
    rows, cols = block.shape
    vals = []
    for i in range(rows):
        for j in range(i + min_lag, rows):
            lag = j - i
            if max_lag is not None and lag > max_lag:
                continue
            v = block[i, j]
            if not np.isnan(v):
                vals.append(v)
    if not vals:
        return np.nan, 0
    return float(np.mean(vals)), len(vals)


# -----------------------------------------------------------------------------
# MAIN ANALYSIS
# -----------------------------------------------------------------------------
summary_all = []

for pca_dir in sorted([d for d in os.listdir(RSA_ROOT) if d.startswith("Mean_PCA_")]):
    pca_path = os.path.join(RSA_ROOT, pca_dir)
    if not os.path.isdir(pca_path):
        continue
    print(f"\n=== Processing {pca_dir} ===")

    pca_outdir = os.path.join(BOOTSTRAP_ROOT, pca_dir)
    os.makedirs(pca_outdir, exist_ok=True)
    global_records = []

    for file in os.listdir(pca_path):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(pca_path, file)
        video_name = file.replace("_PCA_timecourse_RSA.csv", "")
        print(f"Analyzing {video_name}")

        M = pd.read_csv(file_path).to_numpy()
        n_time = M.shape[0]

        boundaries = load_boundaries(video_name)
        if boundaries is None or len(boundaries) == 0:
            print(f"  No boundaries found for {video_name}")
            continue

        long_flag = any(tag in video_name.lower() for tag in ["lms", "c4", "summer"])
        interval_len = 15 if long_flag else 10
        cutoff_s = 75 if long_flag else 60
        boundaries = boundaries[boundaries > cutoff_s]
        indices = np.floor((boundaries - cutoff_s) / interval_len).astype(int)
        indices = np.unique(indices[(indices >= 0) & (indices < n_time)])
        if len(indices) == 0:
            continue
        if indices[-1] < n_time - 1:
            indices = np.append(indices, n_time - 1)

        event_starts = np.concatenate(([0], indices[:-1]))
        event_ends = indices
        event_results = []

        for e, (start, end) in enumerate(zip(event_starts, event_ends)):
            if end - start < 1:
                continue

            # Within-event mean (lag filtered)
            within_block = M[start:end, start:end]
            within_mean, n_pairs = mean_upper_triangle(within_block, min_lag=MIN_LAG, max_lag=MAX_LAG)
            if n_pairs == 0 or np.isnan(within_mean):
                continue

            within_by_lag = pairs_by_lag_within(M, start, end, min_lag=MIN_LAG, max_lag=MAX_LAG)
            within_counts_by_lag = {lag: len(vals) for lag, vals in within_by_lag.items()}
            total_within_pairs = sum(within_counts_by_lag.values())
            if total_within_pairs == 0:
                continue

            across_by_lag = pairs_by_lag_across(M, start, end, n_time, min_lag=MIN_LAG, max_lag=MAX_LAG)
            rng = np.random.default_rng(0)
            boot = lag_matched_bootstrap(across_by_lag, within_counts_by_lag, bootstraps=BOOTSTRAPS, rng=rng)

            if boot.size == 0 or np.std(boot) == 0:
                z, p, across_mean = np.nan, np.nan, np.nan
            else:
                across_mean = float(np.mean(boot))
                z = (within_mean - across_mean) / np.std(boot)
                p_lower = percentileofscore(boot, within_mean) / 100.0
                p_upper = 1 - p_lower
                p = 2 * min(p_lower, p_upper)
                p = min(p, 1.0)

            record = {
                "PCA": pca_dir,
                "video": video_name,
                "event_index": e + 1,
                "within_mean": within_mean,
                "across_mean": across_mean,
                "z": z,
                "p": p,
                "n_pairs": total_within_pairs,
            }
            event_results.append(record)
            global_records.append(record)

        if event_results:
            df = pd.DataFrame(event_results)
            out_file = os.path.join(pca_outdir, f"{video_name}_bootstrap.csv")
            df.to_csv(out_file, index=False)
            summary_all.append(df)
            print(f"  Saved {len(df)} events → {out_file}")

    # -------------------------------------------------------------------------
    # GLOBAL (EVENT-LEVEL PERMUTATION)
    # -------------------------------------------------------------------------
    if len(global_records) > 0:
        df_global = pd.DataFrame(global_records)
        global_outdir = os.path.join(GLOBAL_ROOT, pca_dir)
        os.makedirs(global_outdir, exist_ok=True)

        all_within = df_global["within_mean"].dropna().values
        all_across = df_global["across_mean"].dropna().values

        if len(all_within) > 0 and len(all_across) > 0:
            rng = np.random.default_rng(0)
            observed = np.mean(all_within) - np.mean(all_across)
            combined = np.concatenate([all_within, all_across])
            n_within = len(all_within)
            boot = np.empty(BOOTSTRAPS)

            for b in range(BOOTSTRAPS):
                rng.shuffle(combined)
                boot_within = combined[:n_within]
                boot_across = combined[n_within:]
                boot[b] = np.mean(boot_within) - np.mean(boot_across)

            z = (observed - np.mean(boot)) / np.std(boot)
            p_lower = percentileofscore(boot, observed) / 100.0
            p_upper = 1 - p_lower
            p = 2 * min(p_lower, p_upper)
            p = min(p, 1.0)

            df_summary = pd.DataFrame([{
                "PCA": pca_dir,
                "n_events": len(all_within),
                "within_mean": np.mean(all_within),
                "across_mean": np.mean(all_across),
                "mean_diff": observed,
                "z": z,
                "p_two_tailed": p
            }])
            df_summary.to_csv(os.path.join(global_outdir, "global_bootstrap.csv"), index=False)
            print(f"  🌍 Global event-level permutation saved → {os.path.join(global_outdir, 'global_bootstrap.csv')}")
        else:
            print(f"  ⚠️ Skipped global test for {pca_dir}: insufficient data")

# -----------------------------------------------------------------------------
# SAVE COMBINED SUMMARY
# -----------------------------------------------------------------------------
if summary_all:
    df_all = pd.concat(summary_all, ignore_index=True)
    out_summary = os.path.join(BOOTSTRAP_ROOT, "PCA_within_across_summary.csv")
    df_all.to_csv(out_summary, index=False)
    print(f"\n✅ Combined per-video summary saved to: {out_summary}")
else:
    print("\n⚠️ No per-video results generated.")
