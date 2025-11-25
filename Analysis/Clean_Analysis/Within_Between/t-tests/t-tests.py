import os
import pandas as pd
from scipy.stats import ttest_ind
import csv

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
)
output_dir = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\t-tests"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "t-test_perPCA.csv")

# ----------------------------------------------------------
# Run t-tests and save results
# ----------------------------------------------------------
results = []
for col in [c for c in df.columns if "Mean_PCA" in c or "Mean_gradient" in c]:
    w = df.loc[df["Within_or_Across"] == 1, col]
    a = df.loc[df["Within_or_Across"] == 0, col]
    t, p = ttest_ind(w, a, equal_var=False)
    print(f"{col}: t={t:.3f}, p={p:.4f}")
    results.append([col, t, p, len(w), len(a)])

# ----------------------------------------------------------
# Write to CSV
# ----------------------------------------------------------
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Variable", "t", "p", "n_within", "n_across"])
    writer.writerows(results)

print(f"\n✅ Saved results to: {output_path}")
