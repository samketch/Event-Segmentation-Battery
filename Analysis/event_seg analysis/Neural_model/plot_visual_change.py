"""
=====================================================================
 Visual Change Plotter
=====================================================================
Plots the frame-to-frame visual change time series computed from 
Model.py (Deep Visual Feature Extractor).

Input:
 - *_visualchange.csv : contains columns [Time(s), VisualChange]

Output:
 - *_visualchange_plot.png : line graph saved to same folder

Author: Sam Ketcheson
=====================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
VISCHANGE_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Neural_model\Frame_Embeddings"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Neural_model\Frame_Embeddings\Change_plots"
VIDEOS = ["friends1", "friends2", "friends3"]  # base names only

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Main loop
# =========================
for base in VIDEOS:
    csv_path = os.path.join(VISCHANGE_DIR, f"{base}_visualchange.csv")
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue

    # Load data
    df = pd.read_csv(csv_path)
    if "Time(s)" not in df.columns or "VisualChange" not in df.columns:
        print(f"Columns missing in {csv_path}")
        continue

    # Basic cleaning
    df = df.dropna(subset=["VisualChange"])

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(14, 4))
    plt.plot(df["Time(s)"], df["VisualChange"], color="dodgerblue", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Visual Change (Cosine Distance)")
    plt.title(f"Visual Change Over Time – {base}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save
    outpath = os.path.join(OUTPUT_DIR, f"{base}_visualchange_plot.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved plot: {outpath}")

print("Visual change plots generated successfully.")
