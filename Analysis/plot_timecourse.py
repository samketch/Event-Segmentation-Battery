import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================
INPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\mDES_timecourses_byRunTime"
OUTPUT_DIR = os.path.join(INPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOOP OVER FILES
# ==========================================================
for file in os.listdir(INPUT_DIR):
    if not file.endswith("_PCA_timecourse.csv"):
        continue

    video_name = file.replace("_PCA_timecourse.csv", "")
    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    # Detect PCA columns (in case they vary)
    pca_cols = [c for c in df.columns if "pca" in c.lower()]
    if not pca_cols:
        print(f"⚠️ No PCA columns found in {file}")
        continue

    print(f"🎬 Plotting {video_name} ({len(df)} timepoints)")

    # ======================================================
    # PLOT
    # ======================================================
    plt.figure(figsize=(10, 6))

    for col in pca_cols:
        plt.plot(df["Run_time"], df[col], marker="o", linewidth=2, label=col)

    plt.title(f"{video_name} — Mean PCA Component Timecourses", fontsize=14)
    plt.xlabel("Run Time (s)", fontsize=12)
    plt.ylabel("Mean PCA Component Score", fontsize=12)
    plt.legend(title="Component", loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save plot
    out_path = os.path.join(OUTPUT_DIR, f"{video_name}_PCA_timecourse.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Saved plot → {out_path}")

print("\n🎬 All timecourse plots saved!")
