import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# CONFIG
# ==========================================================
RESULTS = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\MixedEffects_PCA_Events\MixedModel_BinnedResults.csv"
OUTPUT  = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\MixedEffects_PCA_Events\Figures\MixedModel_Betas.png"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================
df = pd.read_csv(RESULTS)

# Rename nicely for plotting
df["PCA_Label"] = df["PCA"].str.replace("Mean_", "").str.replace("_", " ")
df = df.sort_values("PCA_Label")

# ==========================================================
# PLOT BAR WITH ERRORBARS
# ==========================================================
plt.figure(figsize=(7,5))
sns.barplot(
    x="PCA_Label",
    y="Beta_BoundaryDensity",
    data=df,
    palette="RdBu_r",
    capsize=0.2,
    errcolor="black",
    errwidth=1.5,
)

# Add 95% CI whiskers manually (from CI_lower / CI_upper)
for i, row in enumerate(df.itertuples()):
    plt.plot([i, i], [row.CI_lower, row.CI_upper], color="black", linewidth=1.5)

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.ylabel("Effect of Boundary Density (β ± 95% CI)")
plt.xlabel("PCA Component")
plt.title("Mixed-effects model: Boundary influence on Thought Components")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=300)
plt.show()

print(f"✅ Figure saved → {OUTPUT}")

import seaborn as sns, matplotlib.pyplot as plt

for pca in ["Mean_PCA_1"]:
    sns.lmplot(x="BoundaryDensity", y=pca, lowess=True,
               scatter_kws={'alpha':0.5, 's':40})
    plt.title(f"{pca} vs BoundaryDensity")
    plt.xlabel("Boundary Density")
    plt.ylabel(pca)
    plt.show()

