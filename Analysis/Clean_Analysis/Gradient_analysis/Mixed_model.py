import os
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore

# ---------- CONFIG ----------
INPUT_CSV  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Gradient_analysis\Gradient_vs_Boundary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

grad_cols = [f"Mean_gradient_{i}" for i in range(1, 6)]

# ---------- LOAD ----------
df = pd.read_csv(INPUT_CSV)

# Keep only required columns
need = ["VideoName", "BoundaryDensity"] + grad_cols
df = df[need].dropna()

# Optional: z score gradients within each video to mirror standard practice
for g in grad_cols:
    df[g + "_z"] = df.groupby("VideoName")[g].transform(lambda x: zscore(x, nan_policy="omit"))

# If BoundaryDensity is very skewed, you could use log1p
# df["BoundaryDensity_log"] = np.log1p(df["BoundaryDensity"])
# response = "BoundaryDensity_log"
response = "BoundaryDensity"

# ---------- MIXED MODEL ----------
formula = f"{response} ~ " + " + ".join([f"{g}_z" for g in grad_cols])
model = smf.mixedlm(formula, data=df, groups=df["VideoName"])
fit = model.fit(method="lbfgs", reml=False)

# ---------- SAVE ----------
out_txt = os.path.join(OUTPUT_DIR, "Gradient_vs_Boundary_LMM.txt")
with open(out_txt, "w") as f:
    f.write(fit.summary().as_text())

print(fit.summary())
print(f"Saved: {out_txt}")
