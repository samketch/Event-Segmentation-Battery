import os
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
MASTER = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_averages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# Load and inspect
# ----------------------------------------------------------
df = pd.read_csv(MASTER)
print(df.columns.tolist())  # sanity check

# Optional: filter just for the Movie Task if needed
df = df[df["Task_name"].str.contains("movie", case=False, na=False)]

# Identify PCA columns
pca_cols = [c for c in df.columns if c.startswith("PCA_")]

# ----------------------------------------------------------
# Compute averages by ProbeVersion × ProbeNumber
# ----------------------------------------------------------
averages = (
    df.groupby(["ProbeVersion", "ProbeNumber"])[pca_cols]
    .mean()
    .reset_index()
)

# Fill missing combinations (0–15 versions, 0–5 probes)
probe_versions = range(0, 16)
probe_numbers = range(0, 6)
grid = pd.MultiIndex.from_product([probe_versions, probe_numbers], names=["ProbeVersion", "ProbeNumber"])
full = pd.DataFrame(index=grid).reset_index()

merged = pd.merge(full, averages, on=["ProbeVersion", "ProbeNumber"], how="left")

# Rename to match R naming
merged.columns = ["ProbeVersion", "ProbeNumber"] + [f"Average_{c}" for c in pca_cols]

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------
save_path = os.path.join(OUTPUT_DIR, "mDES_global_averages.csv")
merged.to_csv(save_path, index=False)
print(f"\n✅ Saved mean PCA-by-probe averages to:\n{save_path}")
