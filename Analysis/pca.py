"""
============================================================
ThoughtSpace PCA — Event Segmentation Battery
============================================================
Runs PCA on your mDES dataset using the official ThoughtSpace
pipeline (with rotation, KMO/Bartlett tests, word clouds, and
variance reports). Saves all results in organized folders.

Author: Sam Ketcheson
============================================================
"""

import sys
import os
import pandas as pd
from ThoughtSpace.pca import groupedPCA  # now that it's installed correctly

# ============================================================
# 1. Paths and config
# ============================================================
DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\output.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\ThoughtSpace_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUP_COL = "VideoName"  # how to group data (each video = one PCA group)
N_COMPONENTS = "infer"   # ThoughtSpace infers components automatically

# ============================================================
# 2. Load data
# ============================================================
df = pd.read_csv(DATA_PATH)

# Drop any irrelevant columns that aren't numeric
exclude_cols = ["idno", "Task_name"]
df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

# ============================================================
# 3. Initialize and run grouped PCA
# ============================================================
print(f"Running ThoughtSpace grouped PCA on {df[GROUP_COL].nunique()} videos...\n")

pca_model = groupedPCA(grouping_col=GROUP_COL, n_components=N_COMPONENTS)
pca_model.fit(df)

# You must transform before saving so ThoughtSpace has projected scores
_ = pca_model.transform(df)

# ============================================================
# 4. Save outputs (loadings, scores, scree plots, wordclouds)
# ============================================================
pca_model.save(path=OUTPUT_DIR, pathprefix="PCA_byVideo", includetime=False)

print("\nThoughtSpace PCA complete! Results saved to:")
print(OUTPUT_DIR)
