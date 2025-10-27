"""
==========================================================
Per-Video ANOVA of PCA Time Series (Thought Dynamics)
==========================================================
For each movie clip, performs a one-way ANOVA testing
whether PCA component scores vary significantly across
probe numbers (time points).

Input:
 - Time_Series/*_Averaged_PCA_TimeSeries.csv
Output:
 - ANOVA_Results.csv summarizing F, p, eta²
 - Text printout matching the Smallwood et al. style

Author: Sam Ketcheson
==========================================================
"""

import os
import pandas as pd
import pingouin as pg

# ==========================================================
# Config
# ==========================================================
INPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Time_Series"
OUTPUT_FILE = os.path.join(INPUT_DIR, "ANOVA_Results.csv")

# ==========================================================
# Run ANOVAs
# ==========================================================
results = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith("_Averaged_PCA_TimeSeries.csv"):
        continue

    video = file.replace("_Averaged_PCA_TimeSeries.csv", "")
    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    # Identify PCA components
    pca_cols = [c for c in df.columns if c.startswith("PCA_")]

    for comp in pca_cols:
        try:
            # Run one-way ANOVA: PCA_i ~ ProbeNumber
            aov = pg.anova(dv=comp, between="ProbeNumber", data=df, detailed=True)

            F = aov["F"].values[0]
            p = aov["p-unc"].values[0]
            eta2 = aov["np2"].values[0] if "np2" in aov else None

            results.append({
                "Video": video,
                "Component": comp,
                "F": round(F, 3),
                "p": round(p, 5),
                "eta2": round(eta2, 3) if eta2 else None
            })

            sig = "significant" if p < 0.05 else "n.s."
            print(f"{video} — {comp}: F={F:.2f}, p={p:.3f}, η²={eta2:.2f if eta2 else 0} ({sig})")

        except Exception as e:
            print(f"ANOVA failed for {video}, {comp}: {e}")

# Save all results
if results:
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nANOVA results saved → {OUTPUT_FILE}")
else:
    print("No ANOVA results computed.")
