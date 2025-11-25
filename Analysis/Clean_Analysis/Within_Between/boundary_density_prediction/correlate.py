import os
import pandas as pd
from scipy.stats import ttest_ind
import csv
import statsmodels.formula.api as smf

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
df = pd.read_csv(
    r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
)
output_dir = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\boundary_density_prediction"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "correlate_perPCA.csv")

# ----------------------------------------------------------
# Run t-tests and save results
# ----------------------------------------------------------
results = []
for col in [c for c in df.columns if "Mean_PCA" in c or "Mean_gradient" in c]:
    model = smf.ols(f"BoundaryDensity ~ {col}", data=df).fit()
    print(col, model.rsquared, model.pvalues[col])
    results.append([col,model.rsquared, model.pvalues[col]])

# ----------------------------------------------------------
# Write to CSV
# ----------------------------------------------------------
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Variable", "r", "p"])
    writer.writerows(results)

print(f"\n✅ Saved results to: {output_path}")
