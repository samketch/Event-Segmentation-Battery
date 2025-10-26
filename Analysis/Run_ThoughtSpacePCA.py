"""
==========================================================
 ThoughtSpace PCA (by Video)
==========================================================
Performs PCA decomposition on experience-sampling data
using the ThoughtSpace framework, excluding metadata
columns (e.g., ProbeVersion, ProbeNumber, etc.).

Outputs:
 - csvdata/  → PCA scores and loadings
 - wordclouds/ → Component word clouds
 - screeplots/ → Eigenvalue/variance plots
 - descriptives/ → Mean item plots

==========================================================
"""

import os
import pandas as pd
from ThoughtSpace.pca import groupedPCA

# ==========================================================
# Config
# ==========================================================
DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\output.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\ThoughtSpace_Results"

# Columns to exclude from PCA (non-thought variables)
EXCLUDE_COLS = ["ProbeVersion", "ProbeNumber", "Run_time", "correct_per_run", "correct_overall", "watched_before"]

GROUP_COL = "VideoName"  # PCA grouped by video file

# ==========================================================
# Load and clean data
# ==========================================================
df = pd.read_csv(DATA_PATH)

# Drop excluded columns if they exist
df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns], errors="ignore")

# Remove rows with missing group labels (if any)
df = df.dropna(subset=[GROUP_COL])

print(f"Loaded data shape: {df.shape}")
print(f"Grouping by: {GROUP_COL}")
print(f"Columns included in PCA: {df.select_dtypes(include='number').shape[1]} numeric")

# ==========================================================
# Run grouped PCA
# ==========================================================
model = groupedPCA(grouping_col=GROUP_COL, n_components="infer", rotation="varimax")

pca_scores = model.fit_transform(df)

# ==========================================================
# Save results
# ==========================================================
model.save(path=OUTPUT_DIR, pathprefix="PCA_byVideo", includetime=False)

print("\nThoughtSpace PCA complete!")
print(f"Results saved in:\n{OUTPUT_DIR}")
