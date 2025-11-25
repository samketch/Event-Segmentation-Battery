"""
gradient_hierarchy.py
Quantify hierarchical organization of neural boundaries across gradients.

Implements three analyses:
 1. Boundary frequency per gradient (per video + global)
 2. Cross-gradient integration (network graph)
 3. Event-triggered gradient dynamics around behavioural boundaries

Inputs:
 - master_timecourse_csv (contains gradients, BoundaryDensity, Task_name)
 - global_mean_cooccurrence_matrix.npy (optional, from Cooccurrence step)

Outputs saved under <output_root>/Hierarchy/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import zscore


# ---------------------------------------------------------------------
def run_gradient_hierarchy(cfg):
    print("→ Running gradient hierarchy analysis (using master timecourse)")

    outdir = os.path.join(cfg["output_root"], "Hierarchy")
    os.makedirs(outdir, exist_ok=True)

    # ---------------------------------------------------------------
    # LOAD DATA
    master = pd.read_csv(cfg["master_timecourse_csv"])
    print(f"✅ Loaded {len(master):,} rows from master timecourse file.")

    gradients = cfg["gradient_cols"]
    missing = [g for g in gradients if g not in master.columns]
    if missing:
        raise ValueError(f"Missing gradient columns: {missing}")

    if "BoundaryDensity" not in master.columns:
        raise ValueError("Master file must contain 'BoundaryDensity' column.")

    # ---------------------------------------------------------------
    # 1️⃣  BOUNDARY FREQUENCY PER GRADIENT (PER VIDEO)
    all_freqs = []

    for vid, sub in master.groupby("Task_name"):
        sub = sub.dropna(subset=gradients).reset_index(drop=True)
        n_time = len(sub)
        if n_time < 5:
            continue

        vid_freqs = {}
        for g in gradients:
            grad_speed = np.abs(np.diff(sub[g], prepend=sub[g].iloc[0]))
            thresh = np.quantile(grad_speed, cfg["speed_threshold"])
            boundaries = (grad_speed > thresh).astype(int)
            vid_freqs[g] = np.sum(boundaries) / n_time

        row = {"Task_name": vid}
        row.update(vid_freqs)
        all_freqs.append(row)

    freq_df = pd.DataFrame(all_freqs)
    freq_df.to_csv(os.path.join(outdir, "boundary_frequency_perVideo.csv"), index=False)
    print(f"✓ Saved per-video boundary frequency metrics ({len(freq_df)} videos).")

    # Compute global mean frequency across videos
    mean_freq = freq_df[gradients].mean().to_frame("boundary_freq")
    mean_freq.to_csv(os.path.join(outdir, "boundary_frequency_global.csv"))

    plt.figure(figsize=(6, 4))
    plt.bar(mean_freq.index, mean_freq["boundary_freq"], color="steelblue")
    plt.ylabel("Boundary Frequency (proportion of timepoints)")
    plt.title("Boundary Frequency per Gradient (Global Mean)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boundary_frequency_barplot.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------------
    # 2️⃣  HIERARCHICAL INTEGRATION (NETWORK GRAPH)
    co_path = os.path.join(cfg["output_root"], "Cooccurrence", "global_mean_cooccurrence_matrix.npy")
    if os.path.exists(co_path):
        rel_overlap = np.load(co_path)
        G = nx.Graph()
        for i, g1 in enumerate(gradients):
            G.add_node(g1, freq=mean_freq.loc[g1, "boundary_freq"])
            for j, g2 in enumerate(gradients):
                if i < j:
                    weight = rel_overlap[i, j]
                    if not np.isnan(weight):
                        G.add_edge(g1, g2, weight=weight)

        plt.figure(figsize=(6, 5))
        pos = nx.circular_layout(G)
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=[4000 * mean_freq.loc[g, "boundary_freq"] for g in gradients],
            node_color="lightblue",
            width=3 * weights,
            edge_color=weights,
            edge_cmap=plt.cm.coolwarm,
            font_size=10
        )
        plt.title("Gradient Boundary Integration Network\n(Node size = frequency; Edge = co-occurrence)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "gradient_hierarchy_network.png"), dpi=300)
        plt.close()
        print("✓ Saved gradient integration network graph (global mean).")
    else:
        print("⚠️ No co-occurrence matrix found — skipping network graph.")

    # ---------------------------------------------------------------
    # 3️⃣  EVENT-TRIGGERED GRADIENT DYNAMICS (PER VIDEO + GLOBAL)
    print("→ Computing event-triggered gradient dynamics")
    window = 10  # ±10 s
    time_window = np.arange(-window, window + 1)
    all_curves = {g: [] for g in gradients}

    for vid, sub in master.groupby("Task_name"):
        sub = sub.dropna(subset=gradients + ["BoundaryDensity"]).reset_index(drop=True)
        if len(sub) < window * 2:
            continue

        event_thresh = np.quantile(sub["BoundaryDensity"], 0.95)
        event_idx = np.where(sub["BoundaryDensity"].values > event_thresh)[0]

        for g in gradients:
            grad_speed = np.abs(np.diff(sub[g], prepend=sub[g].iloc[0]))
            grad_speed_z = zscore(grad_speed)
            aligned = []
            for b in event_idx:
                start, end = b - window, b + window + 1
                if start >= 0 and end < len(grad_speed_z):
                    aligned.append(grad_speed_z[start:end])
            if len(aligned) > 0:
                mean_curve = np.mean(aligned, axis=0)
                all_curves[g].append(mean_curve)

    # Average across videos
    global_curves = {}
    for g in gradients:
        if len(all_curves[g]) > 0:
            global_curves[g] = np.mean(np.stack(all_curves[g]), axis=0)
        else:
            global_curves[g] = np.full(len(time_window), np.nan)

    plt.figure(figsize=(7, 5))
    for g in gradients:
        plt.plot(time_window, global_curves[g], label=g)
    plt.axvline(0, color="k", ls="--", lw=1)
    plt.xlabel("Time (s) relative to event boundary")
    plt.ylabel("Gradient Speed (z-scored)")
    plt.title("Event-Triggered Gradient Dynamics (Global Mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "event_triggered_gradients_global.png"), dpi=300)
    plt.close()

    np.save(os.path.join(outdir, "event_triggered_gradients_global.npy"), global_curves)
    print("✓ Saved global event-triggered gradient dynamics.\n")

    print("✓ Hierarchy analysis complete.\n")
