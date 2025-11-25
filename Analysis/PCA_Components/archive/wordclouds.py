import os
import math
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly.colors import sample_colorscale
from wordcloud import WordCloud
from scipy.stats import pearsonr

FILE = r"C:\Users\Smallwood Lab\Downloads\skipper_data_moviemanifold_5factor_gradient_Common_NativePCAs (2).csv"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\PCA_Components\wordclouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE)
# Make column matching robust
df = pd.read_csv(FILE, sep="\t")
df.columns = df.columns.str.strip()
print(df)

# MDES and PCA column detection (flexible)
mdes_cols = [c for c in df.columns if c.endswith("_response")]
# match: PCA_1, PCA_2, NativePCA_1, PC1, etc.
pca_cols = [c for c in df.columns if re.match(r'.*\bPCA[_ ]?\d+\b', c, flags=re.IGNORECASE) or re.match(r'^PC\d+$', c, flags=re.IGNORECASE)]

print(f"Found {len(mdes_cols)} MDES cols, {len(pca_cols)} PCA cols")
if not mdes_cols:
    raise RuntimeError("No MDES columns found (expected columns ending with '_response').")
if not pca_cols:
    raise RuntimeError("No PCA columns found (expected names like 'PCA_1', 'NativePCA_1', 'PC1', etc.). "
                       f"Columns available: {list(df.columns)}")

# Compute correlations (MDES × PCA)
loadings = pd.DataFrame(index=mdes_cols, columns=pca_cols, dtype=float)
for mdes_var in mdes_cols:
    for pca in pca_cols:
        r, _ = pearsonr(df[mdes_var], df[pca])
        loadings.loc[mdes_var, pca] = r

# Wordcloud helper with clear sign mapping: red=positive, blue=negative
def wordcloud_(loadings_series, title=None):
    s = loadings_series.dropna().sort_values(ascending=False)

    # Colors: map signed values in [-1,1] to RdBu_r so pos=red, neg=blue
    v = s.values
    # Normalize to [0,1] with center at 0
    # -1 -> 0 (blue), 0 -> 0.5 (white), +1 -> 1 (red)
    norm01 = (v + 1) / 2.0
    norm02 = (v + 1) / 1.5
    colours = sample_colorscale("RdBu_r", norm01)
    colour_dict = {k.replace("_response", ""): c for k, c in zip(s.index, colours)}

    # Sizes: magnitude only
    mag = np.abs(v)
    if (mag.max() - mag.min()) == 0:
        mag_scaled = np.ones_like(mag)  # avoid divide-by-zero if all equal
    else:
        mag_scaled = MinMaxScaler().fit_transform(mag.reshape(-1, 1)).flatten()

    freq_dict = {k.replace("_response", ""): w for k, w in zip(s.index, mag_scaled)}

    def color_func(word, *args, **kwargs):
        return colour_dict[word]

    wc = WordCloud(
        background_color="white",
        color_func=color_func,
        width=600,
        height=600,
        prefer_horizontal=1,
        min_font_size=10,
        max_font_size=250,
        random_state=42,
    ).generate_from_frequencies(freq_dict)

    if title:
        print(f"✅ Generated word cloud for {title}")
    return wc.to_image()

# Generate wordclouds
for pca in loadings.columns:
    wdc = wordcloud_(loadings[pca], title=pca)
    wdc.save(os.path.join(OUTPUT_DIR, f"{pca}.jpeg"))

print(f"All word clouds saved to: {OUTPUT_DIR}")
