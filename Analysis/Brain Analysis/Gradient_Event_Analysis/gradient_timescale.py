import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks


def compute_event_lengths(cfg):
    """
    Compute inter-boundary (event) durations for each gradient using robust peak detection.

    Saves:
        - gradient_event_lengths.csv
        - gradient_event_lengths.png
    """

    def robust_boundary_peaks(series, pct=0.95, min_separation_s=5, prominence=None, fs=1.0):
        """Find distinct boundary peaks separated by at least min_separation_s seconds."""
        series = np.asarray(series)
        thr = np.quantile(series, pct)
        distance = int(np.round(min_separation_s * fs))
        peaks, _ = find_peaks(series, height=thr, distance=distance, prominence=prominence)
        return peaks

    # ------------------------------------------------------------------
    grad = pd.read_csv(cfg["gradient_csv"])
    fs = cfg.get("sampling_rate", 1.0)
    outdir = os.path.join(cfg["output_root"], "Hierarchy")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for g in cfg["gradient_cols"]:
        # Derivative of gradient component
        deriv = np.abs(np.diff(grad[g], prepend=grad[g].iloc[0]))
        peaks = robust_boundary_peaks(deriv, pct=cfg["speed_threshold"],
                                      min_separation_s=5, fs=fs)
        gaps = np.diff(peaks) / fs  # inter-boundary intervals in seconds

        rows.append({
            "gradient": g,
            "n_boundaries": int(peaks.size),
            "mean_event_length_s": float(np.mean(gaps)) if gaps.size else np.nan,
            "median_event_length_s": float(np.median(gaps)) if gaps.size else np.nan
        })

    df = pd.DataFrame(rows)
    df["overall_mean_event_length_s"] = df["mean_event_length_s"].mean()

    # ------------------------------------------------------------------
    # Save CSV
    csv_path = os.path.join(outdir, "gradient_event_lengths.csv")
    df.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Plot mean event length per gradient
    plt.figure(figsize=(6, 4))
    plt.bar(df["gradient"], df["mean_event_length_s"], color="teal")
    plt.ylabel("Mean Event Length (s)")
    plt.title("Mean Event Duration per Gradient")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gradient_event_lengths.png"), dpi=300)
    plt.close()

    print(f"✓ Saved event length metrics and plot → {csv_path}")
    return df
