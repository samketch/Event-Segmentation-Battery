import statsmodels.api as sm
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
output_dir = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\Predict_within_acrossLMM"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "correlate_perPCA.csv")

# ----------------------------------------------------------
# Run tests and save results
# ----------------------------------------------------------
results = []
X = df[[c for c in df.columns if "Mean_PCA" in c or "Mean_gradient" in c]]
X = sm.add_constant(X)
y = df["Within_or_Across"]
model = sm.Logit(y, X).fit()
print(model.summary())
results.append(model.summary)

# ----------------------------------------------------------
# Write to CSV
# ----------------------------------------------------------
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(results)

print(f"\n✅ Saved results to: {output_path}")

