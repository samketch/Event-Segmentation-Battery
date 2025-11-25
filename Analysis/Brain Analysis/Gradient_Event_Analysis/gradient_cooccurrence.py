"""
gradient_cooccurrence.py
Quantify how often gradient boundaries co-occur within and across gradients.
Analogous to Geerligs et al. (2022) Figures 3 & 5.

Outputs:
 - per-video co-occurrence matrices (npy + png)
 - global average co-occurrence matrix across all videos
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
def run_gradient_cooccurrence(cfg):
    print("→ Running gradient co-occurrence / network overlap analysis (using master timecourse)")

    # -----------------------------------------------------------------
    # LOAD DATA
    master = pd.read_csv(cfg["master_timecourse_csv"])
    print(f"✅ Loaded {len(master):,} rows from master timecourse file.")

    # Confirm required columns
    networks = cfg["gradient_cols"]
    missing = [c for c in networks if c not in master.columns]
    if missing:
        raise ValueError(f"Missing required gradient columns: {missing}")

    outdir = os.path.join(cfg["output_root"], "Cooccurrence")
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------------------------------------------
    # CONTAINER FOR GLOBAL AVERAGE
    all_mats = []

    # -----------------------------------------------------------------
    # PER-VIDEO LOOP
    for vid, sub in master.groupby("Task_name"):
        print(f"\n▶ Processing video: {vid}")

        sub = sub.dropna(subset=networks).reset_index(drop=True)
        n_time = len(sub)
        if n_time < 5:
            print(f"Skipping {vid}: too few timepoints ({n_time})")
            continue

        rel_overlap = np.zeros((len(networks), len(networks)))

        # -------------------------------------------------------------
        # Compute boundary series per gradient
        boundary_series = {}
        for g in networks:
            grad_speed_g = np.abs(np.diff(sub[g], prepend=sub[g].iloc[0]))
            grad_thresh_g = np.quantile(grad_speed_g, cfg["speed_threshold"])
            grad_boundary_g = (grad_speed_g > grad_thresh_g).astype(int)
            boundary_series[g] = grad_boundary_g

        # -------------------------------------------------------------
        # Pairwise relative overlap (Geerligs et al. 2022)
        for i, gi in enumerate(networks):
            Si = boundary_series[gi]
            for j, gj in enumerate(networks):
                Sj = boundary_series[gj]
                O = np.sum(Si * Sj)
                OE = (np.sum(Si) * np.sum(Sj)) / n_time
                denom = (min(np.sum(Si), np.sum(Sj)) - OE)
                OR = (O - OE) / denom if denom != 0 else np.nan
                rel_overlap[i, j] = OR

        all_mats.append(rel_overlap)

        # -------------------------------------------------------------
        # SAVE PER-VIDEO MATRIX
        np.save(os.path.join(outdir, f"{vid}_boundary_cooccurrence_matrix.npy"), rel_overlap)

        # -------------------------------------------------------------
        # PLOT HEATMAP
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(rel_overlap, cmap="coolwarm", vmin=0, vmax=np.nanmax(rel_overlap))
        ax.set_xticks(range(len(networks)))
        ax.set_yticks(range(len(networks)))
        ax.set_xticklabels(networks, rotation=45, ha="right")
        ax.set_yticklabels(networks)
        fig.colorbar(im, ax=ax, label="Relative Boundary Overlap")
        plt.title(f"Gradient Boundary Co-occurrence: {vid}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{vid}_cooccurrence_heatmap.png"), dpi=300)
        plt.close()

    # -----------------------------------------------------------------
    # GLOBAL AVERAGE MATRIX
    if len(all_mats) > 0:
        global_mean = np.nanmean(np.stack(all_mats), axis=0)
        np.save(os.path.join(outdir, "global_mean_cooccurrence_matrix.npy"), global_mean)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(global_mean, cmap="coolwarm", vmin=0, vmax=np.nanmax(global_mean))
        ax.set_xticks(range(len(networks)))
        ax.set_yticks(range(len(networks)))
        ax.set_xticklabels(networks, rotation=45, ha="right")
        ax.set_yticklabels(networks)
        fig.colorbar(im, ax=ax, label="Mean Relative Overlap")
        plt.title("Global Average Gradient Co-occurrence")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "global_mean_cooccurrence_heatmap.png"), dpi=300)
        plt.close()

        print(f"\n✓ Saved global average co-occurrence matrix and heatmap → {outdir}")
    else:
        print("\n⚠️ No valid videos processed, global average not computed.")

    print("✓ Gradient co-occurrence analysis complete.\n")
