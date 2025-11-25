# =====================================================================
# EventLength_vs_PCA_and_Gradients.py
#
# Do the exact same event-length analysis you run on gradients,
# but on PCA timecourses. Optionally, run both (PCA + gradients)
# and write side-by-side summaries.
#
# Outputs (per analysis label "PCA" or "GRAD"):
#   - <label>_kde_event_summary.csv
#   - <label>_eventlength_feature_correlations.csv
#   - <label>_eventlength_multivariate_summary.csv
#   - <label>_feature3D_by_eventlength.png  (3D scatter of first 3 dims)
#   - <label>_eventlength_3D_summary.csv    (Spearman, regression, vector)
#
# If run_both=True, also:
#   - COMPARISON_eventlength_PCA_vs_GRAD_summary.csv  (key stats side-by-side)
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import statsmodels.api as sm

import pandas as pd




# -----------------------------
# Small utilities
# -----------------------------
def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def boundaries_by_video_from_consensus(
    consensus_df,
    events_df,
    fs=1.0,
    video_col_candidates=("VideoName","Task_name","Movie","stimulus"),
    idx_col_candidates=("boundary_idx","index"),
    time_col_candidates=("time_s","time","sec","seconds", "run_time", "ConsensusTime(s)"),
    min_sep_s=0
):
    vid_col = _find_col(events_df, video_col_candidates)
    if vid_col is None or vid_col not in consensus_df.columns:
        raise ValueError("Could not find a shared video column in consensus/events.")

    idx_col = _find_col(consensus_df, idx_col_candidates)
    time_col = _find_col(consensus_df, time_col_candidates)
    if idx_col is None and time_col is None:
        raise ValueError("Consensus file needs boundary index column OR time (s) column.")

    distance = int(max(1, round(min_sep_s * fs))) if (min_sep_s and min_sep_s > 0) else 0

    by_video, totals_K, totals_segments = {}, 0, 0
    for vid, dfv in events_df.groupby(vid_col, sort=False):
        vmin, vmax = dfv.index.min(), dfv.index.max()
        sub = consensus_df[consensus_df[vid_col] == vid].copy()

        if sub.empty:
            bounds = np.array([vmin, vmax], dtype=int)
            by_video[vid] = bounds
            totals_segments += 1
            continue

        if idx_col is not None:
            b_idx = sub[idx_col].to_numpy(dtype=int)
        else:
            b_idx = np.round(sub[time_col].to_numpy(dtype=float) * fs).astype(int)

        if b_idx.min() >= 0 and b_idx.max() <= (vmax - vmin + 1):
            b_idx = b_idx + vmin

        b_idx = b_idx[(b_idx >= vmin) & (b_idx <= vmax)]
        b_idx = np.unique(np.sort(b_idx))

        if distance > 0 and b_idx.size > 1:
            kept = [b_idx[0]]
            for x in b_idx[1:]:
                if x - kept[-1] >= distance:
                    kept.append(x)
            b_idx = np.array(kept, dtype=int)

        bounds = np.unique(np.concatenate([[vmin], b_idx, [vmax]]))
        by_video[vid] = bounds
        K = len(sub)
        totals_K += K
        totals_segments += (K + 1)

    return by_video, totals_K, totals_segments, vid_col

# -----------------------------
# Core analysis (generic: features = gradients OR PCA)
# -----------------------------
def analyze_eventlength_vs_features(
    features_df,
    feature_cols,
    events_df,
    consensus_df,
    out_root,
    label="Mean_PCA",
    fs=1.0,
    min_sep_s=0,
    make_3d_plot=True
):
    """
    features_df: time-resolved table with columns feature_cols (e.g., PCA_1..)
    feature_cols: list[str] of feature column names
    events_df: dataframe used to infer per-video spans and shared video column
    consensus_df: consensus boundaries table (per video)
    out_root: output directory for this label
    """
    os.makedirs(out_root, exist_ok=True)



    by_video, totalK, totalSegs, vid_col = boundaries_by_video_from_consensus(
        consensus_df, events_df, fs=fs, min_sep_s=min_sep_s
    )

    # --- Compute per-event metrics strictly within each video
    all_event_lengths, all_event_means, all_event_vars = [], [], []
    video_tag, local_index = [], []

    # --- Normalize video names across sources ---
    def normalize_label(x):
        return (
            str(x)
            .lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace(".mp4", "")
            .replace("pca", "")
            .replace("timecourse", "")
            .replace("movietask", "")
        )

    features_df["video"] = features_df["video"].apply(normalize_label)
    events_df[vid_col] = events_df[vid_col].apply(normalize_label)
    consensus_df[vid_col] = consensus_df[vid_col].apply(normalize_label)


    for vid, bounds in by_video.items():
        # --- select PCA rows for this video ---
        dfv = features_df[features_df["video"] == vid].reset_index(drop=True)
        n_frames = len(dfv)
        if n_frames == 0:
            continue

        # --- get consensus boundaries for this movie ---
        sub_cons = consensus_df[consensus_df[vid_col] == vid].copy()
        if sub_cons.empty:
            # if no boundaries, treat the whole clip as one segment
            local_bounds = np.array([0, n_frames - 1])
        else:
            # use either boundary_idx or time column, whichever exists
            if "boundary_idx" in sub_cons.columns:
                b_idx = sub_cons["boundary_idx"].astype(int).to_numpy()
            elif any(col in sub_cons.columns for col in ["time_s","ConsensusTime(s)","run_time"]):
                tcol = [c for c in ["time_s","ConsensusTime(s)","run_time"] if c in sub_cons.columns][0]
                b_idx = np.round(sub_cons[tcol].astype(float) * fs).astype(int)
            else:
                continue

            # clean and clamp to this clip's length
            b_idx = b_idx[(b_idx >= 0) & (b_idx < n_frames)]
            local_bounds = np.unique(np.concatenate([[0], b_idx, [n_frames - 1]]))

        # --- iterate through events within this clip ---
        for i in range(len(local_bounds) - 1):
            start, end = local_bounds[i], local_bounds[i + 1]
            if end <= start:
                continue
            seg = dfv[feature_cols].iloc[start:end]
            if seg.empty:
                continue

            all_event_lengths.append((end - start) / fs)
            all_event_means.append(seg.mean().values)
            all_event_vars.append(seg.var().values)
            video_tag.append(vid)
            local_index.append(i)




    event_lengths = np.array(all_event_lengths)
    event_means   = np.vstack(all_event_means) if all_event_means else np.empty((0, len(feature_cols)))
    event_vars    = np.vstack(all_event_vars)  if all_event_vars  else np.empty((0, len(feature_cols)))
    n_events = len(event_lengths)

    print(f"[{label}] TOTAL consensus boundaries = {totalK}")
    print(f"[{label}] Expected TOTAL segments   = {totalSegs}")
    print(f"[{label}] Computed events (strict per-video) = {n_events}\n")

    # ---- Per-feature correlations (length ↔ mean, length ↔ within-var)
    corrs = []
    for i, g in enumerate(feature_cols):
        r_pos, p_pos = spearmanr(event_lengths, event_means[:, i]) if n_events > 1 else (np.nan, np.nan)
        r_var, p_var = spearmanr(event_lengths, event_vars[:, i])  if n_events > 1 else (np.nan, np.nan)
        corrs.append({
            "feature": g,
            "r_eventlen_vs_meanpos": r_pos,
            "p_meanpos": p_pos,
            "r_eventlen_vs_withinvar": r_var,
            "p_withinvar": p_var
        })
    corr_df = pd.DataFrame(corrs)
    corr_df.to_csv(os.path.join(out_root, f"{label}_eventlength_feature_correlations.csv"), index=False)

    # ---- Multivariate distance correlation (Mantel-style) across ALL features
    if n_events > 1:
        feat_dist = pdist(event_means, metric="euclidean")
        len_dist  = pdist(event_lengths[:, None], metric="euclidean")
        r_mantel, p_mantel = spearmanr(feat_dist, len_dist)
    else:
        r_mantel, p_mantel = (np.nan, np.nan)

    pd.DataFrame({
        "r_mantel": [r_mantel],
        "p_mantel": [p_mantel]
    }).to_csv(os.path.join(out_root, f"{label}_eventlength_multivariate_summary.csv"), index=False)

    # ---- Event summary table
    summary = pd.DataFrame({
        "video": video_tag,
        "event_within_video": local_index,
        "event_length_s": event_lengths,
        **{f"mean_{c}": event_means[:, i] for i, c in enumerate(feature_cols)},
        **{f"withinvar_{c}": event_vars[:, i] for i, c in enumerate(feature_cols)},
    })
    if n_events > 0:
        summary["mean_withinvar"] = event_vars.mean(axis=1)
    summary.to_csv(os.path.join(out_root, f"{label}_kde_event_summary.csv"), index=False)

    # ---- 3D analysis (first 3 dims) + scatter
    three = min(3, len(feature_cols))
    if three >= 2 and n_events > 0:
        X3 = event_means[:, :three]
        y  = event_lengths

        # 1) Spearman (per axis)
        corrs_3d = [spearmanr(y, X3[:, i]) for i in range(three)]
        corr_table = pd.DataFrame({
            "feature": feature_cols[:three],
            "rho_length_vs_meanpos": [r[0] for r in corrs_3d],
            "p_value": [r[1] for r in corrs_3d]
        })

        # 2) Multiple regression (standardized betas)
        Xz = (X3 - X3.mean(axis=0)) / X3.std(axis=0)
        yz = (y - y.mean()) / y.std()
        Xz = sm.add_constant(Xz)
        model = sm.OLS(yz, Xz).fit()
        reg_summary = pd.DataFrame({
            "term": ["Intercept"] + feature_cols[:three],
            "beta": list(model.params),
            "p_value": list(model.pvalues)
        })
        r2 = model.rsquared

        # 3) Correlation vector + normalized direction
        #    (simple corr with each axis; normalized for direction)
        vec = np.array([np.corrcoef(y, X3[:, i])[0, 1] for i in range(three)])
        vec_norm = vec / np.linalg.norm(vec) if np.isfinite(vec).all() and np.linalg.norm(vec) > 0 else np.full_like(vec, np.nan)

        vector_summary = pd.DataFrame({
            "axis": feature_cols[:three],
            "correlation": vec,
            "normalized_direction": vec_norm
        })
        vector_summary.loc[0, "model_R2"] = r2

        # Save the combined 3D summary as a single CSV (stacked sections)
        outpath = os.path.join(out_root, f"{label}_eventlength_3D_summary.csv")
        with open(outpath, "w") as f:
            f.write("# Spearman correlations (length vs first 3 features)\n")
        corr_table.to_csv(outpath, index=False, mode="a")
        with open(outpath, "a") as f:
            f.write("\n# Multiple regression (standardized betas predicting event length)\n")
        reg_summary.to_csv(outpath, index=False, mode="a")
        with open(outpath, "a") as f:
            f.write("\n# 3D correlation vector (direction of length increase in feature space)\n")
        vector_summary.to_csv(outpath, index=False, mode="a")

        # 3D plot
        if make_3d_plot and three == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                X3[:, 0], X3[:, 1], X3[:, 2],
                c=y, cmap="viridis", s=70, alpha=0.9, edgecolor="k", linewidth=0.4
            )
            ax.set_xlabel(feature_cols[0]); ax.set_ylabel(feature_cols[1]); ax.set_zlabel(feature_cols[2])
            ax.set_title(f"{label}: Event locations (3D) colored by event length")
            fig.colorbar(sc, ax=ax, label="Event Length (s)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_root, f"{label}_feature3D_by_eventlength.png"), dpi=300)
            plt.close()

    print(f"[{label}] ✓ Saved outputs to: {out_root}\n")
    return {
        "summary": summary,
        "corr_df": corr_df,
        "r_mantel": r_mantel,
        "p_mantel": p_mantel,
        "r2_3d": r2 if three >= 2 and n_events > 0 else np.nan
    }

# -----------------------------
# Optional: compare PCA vs Gradients side-by-side
# -----------------------------
def write_side_by_side_comparison(pca_pack, grad_pack, out_dir):
    """
    pca_pack / grad_pack: dicts returned by analyze_eventlength_vs_features
    """
    # Pull the 3D summaries (first three dims) if present
    # We’ll summarize Spearman r for first three and model R² and Mantel r.
    def pull_first3(corr_df, prefix):
        # Expect 'feature' and 'rho_length_vs_meanpos'
        take = corr_df.head(3).copy()
        take = take[["feature", "rho_length_vs_meanpos"]]
        take.columns = [f"{prefix}_feature", f"{prefix}_rho"]
        return take

    p_first = pull_first3(pca_pack["corr_df"], "PCA")
    g_first = pull_first3(grad_pack["corr_df"], "GRAD")

    comp = pd.concat([p_first.reset_index(drop=True), g_first.reset_index(drop=True)], axis=1)
    comp["PCA_r_mantel"]  = pca_pack["r_mantel"]
    comp["PCA_r2_3d"]     = pca_pack["r2_3d"]
    comp["GRAD_r_mantel"] = grad_pack["r_mantel"]
    comp["GRAD_r2_3d"]    = grad_pack["r2_3d"]

    outpath = os.path.join(out_dir, "COMPARISON_eventlength_PCA_vs_GRAD_summary.csv")
    comp.to_csv(outpath, index=False)
    print("✓ Wrote side-by-side comparison →", outpath)

# -----------------------------
# Run config
# -----------------------------
if __name__ == "__main__":
    cfg = {
        # Time-resolved tables (same length & indexing)
        "pca_csv": r"C:\path\to\your\PCA_timecourse.csv",
        "grad_csv": r"C:\path\to\your\gradients.csv",

        # This file is just to provide per-video spans + video ids (same as before)
        "events_csv": r"C:\path\to\your\kde_density.csv",

        # All-videos consensus boundaries
        "consensus_csv": r"C:\path\to\your\consensus_boundaries.csv",

        # Column lists
        "pca_cols": ["Mea_PCA_1","Mean_PCA_2","Mean_PCA_3","Mean_PCA_4","Mean_PCA_5"],
        "grad_cols": ["gradient_1","gradient_2","gradient_3","gradient_4","gradient_5"],

        # Output root
        "output_root": r"C:\path\to\output\EventLength_Analyses",

        # Sampling
        "fs": 1.0,
        "min_sep_s": 0,

        # Whether to also run gradients for a side-by-side comparison table
        "run_both": True
    }

    os.makedirs(cfg["output_root"], exist_ok=True)

    # Load inputs
    pca_df = pd.read_csv(cfg["pca_csv"])
    grad_df = pd.read_csv(cfg["grad_csv"])
    events_df = pd.read_csv(cfg["events_csv"])
    consensus_df = pd.read_csv(cfg["consensus_csv"])

    # PCA analysis
    pca_outdir = os.path.join(cfg["output_root"], "PCA_Event_Structure")
    pca_pack = analyze_eventlength_vs_features(
        features_df=pca_df,
        feature_cols=cfg["pca_cols"],
        events_df=events_df,
        consensus_df=consensus_df,
        out_root=pca_outdir,
        label="PCA",
        fs=cfg["fs"],
        min_sep_s=cfg["min_sep_s"]
    )

    if cfg.get("run_both", False):
        # Gradient analysis (optional, for direct comparison)
        grad_outdir = os.path.join(cfg["output_root"], "GRAD_Event_Structure")
        grad_pack = analyze_eventlength_vs_features(
            features_df=grad_df,
            feature_cols=cfg["grad_cols"],
            events_df=events_df,
            consensus_df=consensus_df,
            out_root=grad_outdir,
            label="GRAD",
            fs=cfg["fs"],
            min_sep_s=cfg["min_sep_s"]
        )
        write_side_by_side_comparison(pca_pack, grad_pack, cfg["output_root"])
