"""
MAIN GRADIENT–EVENT ANALYSIS PIPELINE
Author: Sam Ketcheson
Project: Event Segmentation Battery (Thought × Gradient Dynamics)

This script orchestrates gradient-space analyses that examine how cortical reconfiguration 
(gradient speed) aligns with event segmentation and thought patterns.
"""

import os
import json
from datetime import datetime
import pandas as pd

# Local analysis modules
from gradient_overlap import run_gradient_overlap
from gradient_cooccurrence import run_gradient_cooccurrence
from gradient_timescale import compute_event_lengths
from gradient_hierarchy import run_gradient_hierarchy
from mdes_partialcorr import run_mdes_partialcorr
from Gradient_Event_lengths import analyze_kde_events_in_gradient_space
from PCA_Gradients import analyze_eventlength_vs_features

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
CONFIG = {
    "master_timecourse_csv": r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Master_Data\Master_Timecourse_All.csv",
    "output_root": r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\Gradient_Event_Analysis",
    "gradient_cols": ["Mean_gradient_1", "Mean_gradient_2", "Mean_gradient_3", "Mean_gradient_4", "Mean_gradient_5"],
    "pca_cols": ["Mean_PCA_1", "Mean_PCA_2", "Mean_PCA_3", "Mean_PCA_4", "Mean_PCA_5"],
    "sampling_rate": 1.0,          # 1 Hz
    "speed_threshold": 0.90,       # top 10% = boundary
    "smoothing_sigma": 1,
    "save_plots": True
}

# ---------------------------------------------------------------------
# SELECT WHICH ANALYSIS TO RUN HERE
# ---------------------------------------------------------------------
# Options: "overlap", "cooccurrence", "timescale", "hierarchy", "mdes", "GradientxEvents", "PCA", or "all"
ANALYSIS_TO_RUN = "GradientxEvents"

# ---------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------
os.makedirs(CONFIG["output_root"], exist_ok=True)
with open(os.path.join(CONFIG["output_root"], "config_log.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting analysis: {ANALYSIS_TO_RUN}\n")

# ---------------------------------------------------------------------
# LOAD MASTER DATA
# ---------------------------------------------------------------------
print("→ Loading master timecourse file...")
master_df = pd.read_csv(CONFIG["master_timecourse_csv"])
print(f"✅ Loaded {len(master_df):,} rows and {len(master_df.columns)} columns.")

# ---------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------
if ANALYSIS_TO_RUN in ["overlap", "all"]:
    run_gradient_overlap(CONFIG)

if ANALYSIS_TO_RUN in ["cooccurrence", "all"]:
    run_gradient_cooccurrence(CONFIG)

if ANALYSIS_TO_RUN in ["timescale", "all"]:
    compute_event_lengths(CONFIG)

if ANALYSIS_TO_RUN in ["hierarchy", "all"]:
    run_gradient_hierarchy(CONFIG)

if ANALYSIS_TO_RUN in ["mdes", "all"]:
    run_mdes_partialcorr(CONFIG)

if ANALYSIS_TO_RUN in ["GradientxEvents", "all"]:
    analyze_kde_events_in_gradient_space(CONFIG)

if ANALYSIS_TO_RUN in ["PCA", "all"]:
    analyze_eventlength_vs_features(
        features_df=master_df,
        feature_cols=CONFIG["pca_cols"],
        events_df=master_df,              # now contained within the same file
        consensus_df=master_df,           # same master file (can use boundary density column)
        out_root=os.path.join(CONFIG["output_root"], "PCA_Event_Analysis"),
        label="Mean_PCA",
        fs=CONFIG["sampling_rate"],
        min_sep_s=0
    )

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All done.\n")
