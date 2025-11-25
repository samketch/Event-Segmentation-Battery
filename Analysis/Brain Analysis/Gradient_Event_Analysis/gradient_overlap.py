"""
gradient_overlap.py
Compute absolute and relative overlap between gradient boundaries and event boundaries,
test whether stronger (high-speed) boundaries align more with events,
and compute correlations between gradient speed and event density.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
def run_gradient_overlap(cfg):
    print("→ Running gradient–event overlap analysis (using master timecourse)")

    # ---------------------------------------------------------------
    # LOAD MASTER DATA
    master = pd.read_csv(cfg["master_timecourse_csv"])
    print(f"✅ Loaded {len(master):,} rows from master timecourse file.")

    # Ensure necessary columns exist
    required_cols = cfg["gradient_cols"] + ["BoundaryDensity", "Task_name"]
    missing = [c for c in required_cols if c not in master.columns]
    if missing:
        raise ValueError(f"Missing required columns in master file: {missing}")

    # ---------------------------------------------------------------
    # OUTPUT DIRECTORY
    outdir = os.path.join(cfg["output_root"], "Overlap")
    os.makedirs(outdir, exist_ok=True)

    summary_rows = []

    # ---------------------------------------------------------------
    # RUN PER VIDEO
    for vid, sub in master.groupby("Task_name"):
        print(f"\n▶ Processing video: {vid}")

        # Drop NaNs in gradient or boundary columns
        sub = sub.dropna(subset=cfg["gradient_cols"] + ["BoundaryDensity"]).reset_index(drop=True)

        # Z-score gradients
        sub[cfg["gradient_cols"]] = sub[cfg["gradient_cols"]].apply(zscore)

        # Compute gradient speed (Euclidean distance in gradient space)
        speed = np.sqrt(np.sum(np.diff(sub[cfg["gradient_cols"]], axis=0) ** 2, axis=1))
        speed = np.append(speed, speed[-1])
        sub["speed_z"] = zscore(speed)
        sub["speed_z"] = gaussian_filter1d(sub["speed_z"], sigma=cfg["smoothing_sigma"])

        # Normalize BoundaryDensity
        sub["BoundaryDensity_z"] = zscore(sub["BoundaryDensity"])

        # -----------------------------------------------------------
        # DEFINE BOUNDARIES (top X% gradients and top 5% events)
        grad_thresh = np.quantile(sub["speed_z"], cfg["speed_threshold"])
        sub["grad_boundary"] = (sub["speed_z"] > grad_thresh).astype(int)

        event_thresh = np.quantile(sub["BoundaryDensity_z"], 0.95)
        sub["event_boundary"] = (sub["BoundaryDensity_z"] > event_thresh).astype(int)

        # -----------------------------------------------------------
        # COMPUTE OVERLAP
        E = sub["event_boundary"].values
        S = sub["grad_boundary"].values
        S_strength = sub["speed_z"].values * S
        n = len(E)

        O = np.sum(E * S)
        OE = (np.sum(E) * np.sum(S)) / n
        OA = (O - OE) / (np.sum(S) - OE)
        OR = (O - OE) / (min(np.sum(E), np.sum(S)) - OE)

        O_strength = np.sum(E * S_strength)
        OE_strength = (np.sum(E) * np.sum(S_strength)) / n
        OA_strength = (O_strength - OE_strength) / (np.sum(S_strength) - OE_strength)

        # -----------------------------------------------------------
        # CORRELATION METRICS
        r, p = pearsonr(sub["speed_z"], sub["BoundaryDensity_z"])
        window = 60  # rolling window in seconds
        rolling_r = sub["speed_z"].rolling(window).corr(sub["BoundaryDensity_z"])

        both = np.sum((sub["grad_boundary"] == 1) & (sub["event_boundary"] == 1))
        total_grad_boundaries = np.sum(sub["grad_boundary"])
        overlap_pct = 100 * both / total_grad_boundaries if total_grad_boundaries > 0 else 0

        print(f"  r = {r:.3f}, p = {p:.5f}, overlap = {overlap_pct:.1f}%")

        summary_rows.append({
            "Task_name": vid,
            "AbsoluteOverlap": OA,
            "RelativeOverlap": OR,
            "StrengthWeightedOverlap": OA_strength,
            "Correlation_r": r,
            "Correlation_p": p,
            "SharedBoundaryPct": overlap_pct,
            "GradThreshold": grad_thresh,
            "EventThreshold": event_thresh
        })

        # -----------------------------------------------------------
        # PLOT (OPTIONAL)
        if cfg["save_plots"]:
            plt.figure(figsize=(12, 6))
            grad_smooth = sub["speed_z"].rolling(10, center=True, min_periods=1).mean()
            event_smooth = sub["BoundaryDensity_z"].rolling(10, center=True, min_periods=1).mean()

            plt.plot(grad_smooth, label="Gradient Speed (smoothed)", color="royalblue")
            plt.plot(event_smooth, label="Event Density (smoothed)", color="orange", alpha=0.8)

            for t in sub.index[sub["grad_boundary"] == 1]:
                plt.axvline(t, color="blue", alpha=0.15, lw=0.7)
            for t in sub.index[sub["event_boundary"] == 1]:
                plt.axvline(t, color="red", alpha=0.15, lw=0.7)

            plt.title(f"{vid}\nGradient Speed vs Event Boundary Density\n"
                      f"r = {r:.3f}, p = {p:.5f}, {overlap_pct:.1f}% shared")
            plt.xlabel("Time (s)")
            plt.ylabel("Z-scaled (smoothed)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{vid}_gradient_event_correlation.png"), dpi=300)
            plt.close()

        # -----------------------------------------------------------
        # SAVE PER-VIDEO TIMESERIES CSV
        out_csv = os.path.join(outdir, f"{vid}_gradient_event_timeseries.csv")
        sub_out = pd.DataFrame({
            "Time": np.arange(len(sub)),
            "Speed_z": sub["speed_z"],
            "BoundaryDensity_z": sub["BoundaryDensity_z"],
            "GradBoundary": sub["grad_boundary"],
            "EventBoundary": sub["event_boundary"],
            "RollingCorr_60s": rolling_r
        })
        sub_out.to_csv(out_csv, index=False)

    # ---------------------------------------------------------------
    # SAVE SUMMARY
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(outdir, "gradient_event_overlap_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved per-video summary → {summary_csv}\n")
