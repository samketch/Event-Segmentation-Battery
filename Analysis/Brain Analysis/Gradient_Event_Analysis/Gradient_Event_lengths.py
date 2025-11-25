# =====================================================================
# Gradient_Event_lengths_consensus3D_FIXED_MASTER.py
# Uses a unified MASTER timecourse file and computes event metrics
# STRICTLY within each Task_name (video).
#
# Outputs:
#   - kde_event_summary.csv
#   - eventlength_gradient_correlations.csv
#   - eventlength_multivariate_summary.csv
#   - gradientspace_3D_by_eventlength.png
#   - eventlength_gradient3D_summary.csv
#   - console diagnostics with per-video counts
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import statsmodels.api as sm


# ---------------------------------------------------------------------
def analyze_kde_events_in_gradient_space(cfg):
    print("→ KDE × Gradient-space event structure analysis (MASTER PER-VIDEO)")

    master = pd.read_csv(cfg["master_timecourse_csv"])
    gradients = cfg["gradient_cols"]
    fs = cfg.get("sampling_rate", 1.0)
    outdir = os.path.join(cfg["output_root"], "KDE_Event_Structure")
    os.makedirs(outdir, exist_ok=True)

    if "Task_name" not in master.columns:
        raise ValueError("Master file must include 'Task_name' column.")
    if "BoundaryDensity" not in master.columns:
        raise ValueError("Master file must include 'BoundaryDensity' column.")

    # -----------------------------------------------------------------
    # DETERMINE PER-VIDEO BOUNDARY INDICES BASED ON KDE PEAKS
    vid_col = "Task_name"
    min_sep_s = cfg.get("min_sep_s", 0)
    distance = int(max(1, round(min_sep_s * fs))) if (min_sep_s and min_sep_s > 0) else 0

    by_video = {}
    print("\nPer-video boundary diagnostics:")
    total_segments = 0

    for vid, dfv in master.groupby(vid_col, sort=False):
        dfv = dfv.reset_index(drop=True)
        vmin, vmax = dfv.index.min(), dfv.index.max()

        # Find top 5% BoundaryDensity timepoints as boundaries
        thresh = np.quantile(dfv["BoundaryDensity"], 0.95)
        b_idx = np.where(dfv["BoundaryDensity"] > thresh)[0]

        # Optional minimum separation
        if distance > 0 and len(b_idx) > 1:
            kept = [b_idx[0]]
            for x in b_idx[1:]:
                if x - kept[-1] >= distance:
                    kept.append(x)
            b_idx = np.array(kept, dtype=int)

        # Add start/end sentinels
        bounds = np.unique(np.concatenate([[vmin], b_idx, [vmax]]))
        by_video[vid] = bounds
        print(f"  • {vid}: consensus-like K={len(b_idx)}, segments={len(bounds)-1}")
        total_segments += (len(bounds) - 1)

    print(f"\nTOTAL segments across all videos = {total_segments}\n")

    # -----------------------------------------------------------------
    # COMPUTE EVENT METRICS STRICTLY WITHIN EACH VIDEO
    all_event_lengths = []
    all_event_means = []
    all_event_vars = []
    video_tag = []
    local_index = []

    for vid, bounds in by_video.items():
        sub = master[master[vid_col] == vid]
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            seg = sub[gradients].iloc[start:end]
            if seg.shape[0] == 0:
                continue
            all_event_lengths.append((end - start) / fs)
            all_event_means.append(seg.mean().values)
            all_event_vars.append(seg.var().values)
            video_tag.append(vid)
            local_index.append(i)

    event_lengths = np.array(all_event_lengths)
    event_means = np.vstack(all_event_means) if all_event_means else np.empty((0, len(gradients)))
    event_vars = np.vstack(all_event_vars) if all_event_vars else np.empty((0, len(gradients)))

    n_events = len(event_lengths)
    print(f"✓ Computed {n_events} total events (strictly within-video)\n")

    # -----------------------------------------------------------------
    # CORRELATIONS: EVENT LENGTH vs MEAN / VAR
    corrs = []
    for i, g in enumerate(gradients):
        r_pos, p_pos = spearmanr(event_lengths, event_means[:, i]) if n_events > 1 else (np.nan, np.nan)
        r_var, p_var = spearmanr(event_lengths, event_vars[:, i]) if n_events > 1 else (np.nan, np.nan)
        corrs.append({
            "gradient": g,
            "r_eventlen_vs_meanpos": r_pos,
            "p_meanpos": p_pos,
            "r_eventlen_vs_withinvar": r_var,
            "p_withinvar": p_var
        })
    pd.DataFrame(corrs).to_csv(os.path.join(outdir, "eventlength_gradient_correlations.csv"), index=False)

    # -----------------------------------------------------------------
    # DISTANCE BETWEEN CONSECUTIVE EVENTS (WITHIN VIDEO)
    dists_to_next = []
    for vid in sorted(set(video_tag), key=list(dict.fromkeys(video_tag)).index):
        idxs = [j for j, v in enumerate(video_tag) if v == vid]
        if len(idxs) < 2:
            continue
        em = event_means[idxs, :]
        dists = np.sqrt(np.sum(np.diff(em, axis=0)**2, axis=1))
        dists_to_next.extend(dists.tolist())

    if len(dists_to_next) > 1:
        aligned_lengths = []
        for vid in sorted(set(video_tag), key=list(dict.fromkeys(video_tag)).index):
            idxs = [j for j, v in enumerate(video_tag) if v == vid]
            aligned_lengths.extend(event_lengths[idxs[:-1]].tolist())
        r_dist, p_dist = spearmanr(aligned_lengths, dists_to_next)
    else:
        r_dist, p_dist = (np.nan, np.nan)

    # Mantel-style correlation between event-length and gradient-space distances
    grad_dist = pdist(event_means, metric="euclidean") if n_events > 1 else np.array([np.nan])
    len_dist = pdist(event_lengths[:, None], metric="euclidean") if n_events > 1 else np.array([np.nan])
    if grad_dist.size > 1:
        r_mantel, p_mantel = spearmanr(grad_dist, len_dist)
    else:
        r_mantel, p_mantel = (np.nan, np.nan)

    pd.DataFrame({
        "r_mantel": [r_mantel],
        "p_mantel": [p_mantel],
        "r_distlen_to_next": [r_dist],
        "p_distlen_to_next": [p_dist]
    }).to_csv(os.path.join(outdir, "eventlength_multivariate_summary.csv"), index=False)

    # -----------------------------------------------------------------
    # PER-EVENT SUMMARY
    summary = pd.DataFrame({
        "video": video_tag,
        "event_within_video": local_index,
        "event_length_s": event_lengths,
        **{f"mean_{g}": event_means[:, i] for i, g in enumerate(gradients)},
        **{f"withinvar_{g}": event_vars[:, i] for i, g in enumerate(gradients)},
    })
    summary["mean_withinvar"] = event_vars.mean(axis=1) if n_events > 0 else np.nan
    summary.to_csv(os.path.join(outdir, "kde_event_summary.csv"), index=False)

    # -----------------------------------------------------------------
    # 3D SCATTER (FIRST THREE GRADIENTS)
    if event_means.shape[1] >= 3 and n_events > 0:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            event_means[:, 0], event_means[:, 1], event_means[:, 2],
            c=event_lengths, cmap="viridis", s=70, alpha=0.9, edgecolor="k", linewidth=0.4
        )
        ax.set_xlabel("Gradient 1"); ax.set_ylabel("Gradient 2"); ax.set_zlabel("Gradient 3")
        ax.set_title("Event locations in 3D gradient space\ncolored by event length")
        fig.colorbar(sc, ax=ax, label="Event Length (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "gradientspace_3D_by_eventlength.png"), dpi=300)
        plt.close()

    # -----------------------------------------------------------------
    # ADDITIONAL QUANTITATIVE 3D SUMMARY
    if event_means.shape[1] >= 3 and n_events > 1:
        X = event_means[:, :3]
        y = event_lengths

        # 1. Spearman correlations
        corrs_3d = [spearmanr(y, X[:, i]) for i in range(3)]
        corr_table = pd.DataFrame({
            "gradient": ["Mean_gradient_1", "Mean_gradient_2", "Mean_gradient_3"],
            "rho_length_vs_meanpos": [r[0] for r in corrs_3d],
            "p_value": [r[1] for r in corrs_3d]
        })

        # 2. Multiple regression (standardized betas)
        # --- 2. Multiple regression (standardized betas, safe handling)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0)
        y_z = (y - y.mean()) / y.std()

        # Replace infs and nans caused by division-by-zero or empty variance
        X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)
        y_z = np.nan_to_num(y_z, nan=0.0, posinf=0.0, neginf=0.0)

        # Drop any rows that are entirely zeros (no usable data)
        valid_rows = np.all(np.isfinite(X_z), axis=1)
        X_z = X_z[valid_rows]
        y_z = y_z[valid_rows]

        if len(y_z) > 3:
            X_z = sm.add_constant(X_z)
            model = sm.OLS(y_z, X_z).fit()
            reg_summary = pd.DataFrame({
                "gradient": ["Intercept", "Gradient 1", "Gradient 2", "Gradient 3"],
                "beta": model.params,
                "p_value": model.pvalues
            })
            r2 = model.rsquared
        else:
            reg_summary = pd.DataFrame({
                "gradient": ["Intercept", "Gradient 1", "Gradient 2", "Gradient 3"],
                "beta": [np.nan]*4,
                "p_value": [np.nan]*4
            })
            r2 = np.nan

        reg_summary = pd.DataFrame({
            "gradient": ["Intercept", "Gradient 1", "Gradient 2", "Gradient 3"],
            "beta": model.params,
            "p_value": model.pvalues
        })
        r2 = model.rsquared

        # 3. 3D correlation vector
        vec = np.corrcoef(np.vstack([y, X.T]))[0, 1:4]
        vec_norm = vec / np.linalg.norm(vec)

        vector_summary = pd.DataFrame({
            "axis": ["gradient_1", "gradient_2", "gradient_3"],
            "correlation": vec,
            "normalized_direction": vec_norm
        })
        vector_summary.loc[0, "model_R2"] = r2

        outpath = os.path.join(outdir, "eventlength_gradient3D_summary.csv")
        with open(outpath, "w") as f:
            f.write("# Spearman correlations (length vs each gradient)\n")
        corr_table.to_csv(outpath, index=False, mode="a")

        with open(outpath, "a") as f:
            f.write("\n# Multiple regression (standardized betas predicting event length)\n")
        reg_summary.to_csv(outpath, index=False, mode="a")

        with open(outpath, "a") as f:
            f.write("\n# 3D correlation vector (direction of length increase in gradient space)\n")
        vector_summary.to_csv(outpath, index=False, mode="a")

        print(f"✓ Saved quantitative 3D summary → {outpath}")

    print("✓ All outputs saved to:", outdir)
    return summary


# ---------------------------------------------------------------------
# Example config
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = {
        "master_timecourse_csv": r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All.csv",
        "gradient_cols": ["Mean_gradient_1", "Mean_gradient_2", "Mean_gradient_3", "Mean_gradient_4", "Mean_gradient_5"],
        "sampling_rate": 1.0,
        "output_root": r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\Gradient_Event_Analysis\KDE_Event_Structure",
        "min_sep_s": 0
    }
    analyze_kde_events_in_gradient_space(cfg)
