import os
import re
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.colors import sample_colorscale
from wordcloud import WordCloud
from scipy.stats import pearsonr

# ==========================================================
# CONFIG
# ==========================================================
FILE = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\PCA_Components\wordclouds"
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"   # or path to your Helvetica font
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(FILE, sep="\t")
df.columns = df.columns.str.strip()

# Identify MDES and PCA columns
mdes_cols = [c for c in df.columns if c.endswith("_response")]
pca_cols = [c for c in df.columns if re.match(r'.*\bPCA[_ ]?\d+\b', c, flags=re.IGNORECASE) or re.match(r'^PC\d+$', c, flags=re.IGNORECASE)]

print(f"Found {len(mdes_cols)} MDES cols, {len(pca_cols)} PCA cols")

if not mdes_cols:
    raise RuntimeError("No MDES columns found (expected columns ending with '_response').")
if not pca_cols:
    raise RuntimeError("No PCA columns found (expected names like 'PCA_1', 'NativePCA_1', 'PC1', etc.).")

# ==========================================================
# COMPUTE CORRELATIONS (MDES × PCA)
# ==========================================================
loadings = pd.DataFrame(index=mdes_cols, columns=pca_cols, dtype=float)
for mdes_var in mdes_cols:
    for pca in pca_cols:
        r, _ = pearsonr(df[mdes_var], df[pca])
        loadings.loc[mdes_var, pca] = r

# ==========================================================
# HELPERS
# ==========================================================
def clean_labels(labels):
    """Remove '_response' suffix."""
    return [re.sub(r"_response$", "", l) for l in labels]

def returnhighest(series: pd.Series, n_items: int = 3) -> str:
    """Return the top n absolute loadings as a string for filenames."""
    top_items = series.abs().sort_values(ascending=False).head(n_items)
    return "-".join([re.sub(r"_response$", "", n) for n in top_items.index])

# ==========================================================
# WORDCLOUD GENERATION
# ==========================================================
for col in loadings.columns:
    s = loadings[col].dropna()
    s.index = clean_labels(s.index)

    # Normalized magnitude (for word size)
    mag = np.abs(s.values)
    mag_scaled = MinMaxScaler().fit_transform(mag.reshape(-1, 1)).flatten()
    freq_dict = {k: v for k, v in zip(s.index, mag_scaled)}

    # Normalize values to [0,1] for color sampling
    # -1 → 0 (dark blue), 0 → 0.5 (white), +1 → 1 (dark red)
    color_norm = (s.values + 1) / 2.0
    color_vals = np.clip(color_norm, 0, 1)
    colours = sample_colorscale("RdBu_r", color_vals)
    colour_dict = {k: c for k, c in zip(s.index, colours)}

    def color_func(word, *args, **kwargs):
        return colour_dict.get(word, "#000000")

    wc = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        color_func=color_func,
        width=600,
        height=600,
        prefer_horizontal=1,
        min_font_size=10,
        max_font_size=250,
        random_state=42,
    ).generate_from_frequencies(freq_dict)

    highest = returnhighest(loadings[col])
    out_name = f"{col}_{highest}.png"
    wc.to_file(os.path.join(OUTPUT_DIR, out_name))
    print(f"✅ Saved {out_name}")

print(f"\n🎨 All PCA wordclouds saved to: {OUTPUT_DIR}")
