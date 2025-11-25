# ==========================================================
# Combine Event-Segmentation Bootstrap Results
# ----------------------------------------------------------
# Computes overall summary stats for a PCA bootstrap CSV:
#   • Weighted and unweighted mean within-across difference
#   • Weighted and unweighted mean z-score
#   • Combined p-value using Fisher’s method
# ----------------------------------------------------------
# Author: Sam Ketcheson
# ==========================================================

import pandas as pd
from scipy.stats import combine_pvalues
import os

# ===== CONFIG =====
INPUT_CSV = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\RSA_Cross_Within\Within_Across_Bootstrap\PCA_5\Bootstrap\global_Mean_PCA_5_within_across_bootstrap.csv"

# ===== LOAD =====
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df)} events from: {os.path.basename(INPUT_CSV)}")

# ===== BASIC CLEANUP =====
df["diff"] = df["within_mean"] - df["across_mean"]
df["p"] = df["p"].replace(0, 1e-6)  # avoid log(0) issues
df["n_pairs"] = df["n_pairs"].replace(0, 1)  # avoid divide by zero

# ===== UNWEIGHTED MEANS =====
mean_diff = df["diff"].mean()
mean_z = df["z"].mean()

# ===== WEIGHTED MEANS (by number of pairs) =====
w = df["n_pairs"]
weighted_diff = (df["diff"] * w).sum() / w.sum()
weighted_z = (df["z"] * w).sum() / w.sum()

# ===== COMBINED P-VALUE =====
stat, combined_p = combine_pvalues(df["p"], method="fisher")

# ===== DISPLAY RESULTS =====
print("\n=== GLOBAL SUMMARY ===")
print(f"Unweighted mean Δ (within - across): {mean_diff:.4f}")
print(f"Weighted mean Δ (within - across):   {weighted_diff:.4f}")
print(f"Unweighted mean z:                    {mean_z:.4f}")
print(f"Weighted mean z:                      {weighted_z:.4f}")
print(f"Fisher combined statistic:            {stat:.3f}")
print(f"Combined p-value (Fisher):            {combined_p:.6f}")

# ===== OPTIONAL SAVE =====
out_path = INPUT_CSV.replace(".csv", "_global_summary.csv")
summary = pd.DataFrame({
    "mean_diff": [mean_diff],
    "weighted_diff": [weighted_diff],
    "mean_z": [mean_z],
    "weighted_z": [weighted_z],
    "fisher_stat": [stat],
    "combined_p": [combined_p],
})
summary.to_csv(out_path, index=False)
print(f"\n✅ Saved summary to: {out_path}")
