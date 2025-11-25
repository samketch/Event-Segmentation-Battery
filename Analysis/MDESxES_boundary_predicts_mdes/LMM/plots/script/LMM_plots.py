import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==============================================================
# CONFIG
# ==============================================================
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\plots"
SUMMARY_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\MixedLM_summary_all_components_zscored.csv"
DATA_PATH = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\MDESxES_boundary_predicts_mdes\LMM\combined_data.csv"  # optional for predicted plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================
# LOAD RESULTS
# ==============================================================
res = pd.read_csv(SUMMARY_PATH)
res["Component"] = res["Component"].str.replace("Mean_", "", regex=False)

# ==============================================================
# 1. FIXED-EFFECT COEFFICIENT PLOT
# ==============================================================
plt.figure(figsize=(8, 5))
sns.pointplot(data=res, x="Component", y="Coef_BoundaryStrength", join=False, color="black")
plt.errorbar(
    x=range(len(res)),
    y=res["Coef_BoundaryStrength"],
    yerr=res["SE_BoundaryStrength"],
    fmt="none",
    ecolor="gray",
    capsize=4
)
plt.axhline(0, color="red", linestyle="--")
plt.title("Effect of Boundary Strength on PCA Components")
plt.ylabel("Fixed Effect (β for BoundaryStrength)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "LMM_FixedEffects_CoefficientPlot.png"), dpi=300)
plt.close()

print("✅ Saved: Coefficient plot")

# ==============================================================
# 2. HEATMAP OF COEFFICIENTS
# ==============================================================
plt.figure(figsize=(6, 4))
sns.heatmap(
    res.set_index("Component")[["Coef_BoundaryStrength"]],
    annot=True,
    cmap="RdBu_r",
    center=0,
    cbar_kws={"label": "β (BoundaryStrength)"}
)
plt.title("Fixed Effects Across PCA Components")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "LMM_FixedEffects_Heatmap.png"), dpi=300)
plt.close()

print("✅ Saved: Heatmap of coefficients")

# ==============================================================
# 3. PREDICTED RELATIONSHIP PLOTS (OPTIONAL)
# ==============================================================
# This assumes you have a combined dataset "data" used to fit the model.
# You can load it or skip this section if you only want summary plots.

try:
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        print("⚙️ Generating predicted relationship plots (example PCs)")
        for comp in res["Component"].head(5):  # top 3 PCs
            col = f"Mean_{comp}"
            sns.lmplot(
                data=data,
                x="BoundaryStrength",
                y=col,
                lowess=True,
                scatter=False,
                line_kws={"color": "black"},
            )
            plt.title(f"{col}: Relationship with Boundary Strength")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"LMM_Predicted_{comp}.png"), dpi=300)
            plt.close()
        print("✅ Saved: Predicted relationship plots (first 3 PCs)")
    else:
        print(f"⚠️ combined_data.csv not found at: {DATA_PATH}")

except Exception as e:
    print(f"⚠️ Skipped predicted relationship plots: {e}")

# ==============================================================
# 4. RANDOM SLOPE / VIDEO VARIABILITY PLACEHOLDER
# ==============================================================
# This section is included as a template — if you later extract random slopes,
# you can easily visualize them as caterpillar plots.

# Example format:
# re_df = pd.DataFrame({"Component": [...], "Video": [...], "Slope": [...]})
# sns.boxplot(data=re_df, x="Component", y="Slope")
# plt.axhline(0, color="red", linestyle="--")
# plt.title("Random Slopes by Component (Video-Level)")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "LMM_RandomSlope_Boxplot.png"), dpi=300)
# plt.close()

print(f"\n🎉 All plots saved to: {OUTPUT_DIR}")
