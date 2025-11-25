import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ============================================================
# CONFIG
# ============================================================
FILE = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv"   # <-- change this to your CSV path
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\PCA_Components"  # <-- folder to save loadings tables


# ============================================================
# LOAD AND PREP DATA
# ============================================================
df = pd.read_csv(FILE)

# Identify relevant columns
mdes_cols = [c for c in df.columns if c.endswith("_response")]
pca_cols = [c for c in df.columns if c.startswith("PCA_")]

# ============================================================
# CALCULATE CORRELATIONS (MDES × PCA)
# ============================================================
loadings = pd.DataFrame(index=mdes_cols, columns=pca_cols)

for mdes_var in mdes_cols:
    for pca in pca_cols:
        r, _ = pearsonr(df[mdes_var], df[pca])
        loadings.loc[mdes_var, pca] = r

loadings = loadings.astype(float)

# ============================================================
# SAVE RESULTS
# ============================================================
loadings.to_csv(f"{OUTPUT_DIR}/mdes_pca_loadings.csv")

# Also print top contributing variables for each PCA
for pca in pca_cols:
    print(f"\n=== {pca} ===")
    top_vars = loadings[pca].sort_values(key=abs, ascending=False)
    print(top_vars)

# Optional: visualize as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(loadings.astype(float), cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("MDES–PCA Component Correlations (Loadings)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mdes_pca_heatmap.png", dpi=300)
plt.show()
