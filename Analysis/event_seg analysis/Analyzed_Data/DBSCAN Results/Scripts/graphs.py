import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- Parameters ----
base_path = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\DBSCAN Results"
graphs_path = os.path.join(base_path, "graphs")
videos = ["friends1", "friends2", "friends3"]
combined_file = os.path.join(base_path, "all_videos_consensus_boundaries.csv")

# ---- Create output folder ----
os.makedirs(graphs_path, exist_ok=True)

# ---- Plot each video separately ----
for video in videos:
    file = os.path.join(base_path, f"{video}_consensus_boundaries.csv")
    if not os.path.exists(file):
        print(f"Missing file: {file}")
        continue

    df = pd.read_csv(file)

    plt.figure(figsize=(12, 4))
    plt.plot(df["ConsensusTime(s)"], df["ParticipantsInAgreement"], marker='o', linestyle='-')
    plt.title(f"Consensus Boundaries - {video}")
    plt.xlabel("Time (s)")
    plt.ylabel("Participants in Agreement")
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(graphs_path, f"{video}_consensus_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")

# ---- Combined Plot ----
df_all = pd.read_csv(combined_file)
plt.figure(figsize=(14, 5))

for video in videos:
    subset = df_all[df_all['VideoName'].str.contains(video)]
    plt.plot(subset["ConsensusTime(s)"], subset["ParticipantsInAgreement"], marker='o', linestyle='-', label=video)

plt.title("Consensus Boundaries Across All Videos")
plt.xlabel("Time (s)")
plt.ylabel("Participants in Agreement")
plt.legend()
plt.grid(True)
plt.tight_layout()

combined_plot_path = os.path.join(graphs_path, "combined_consensus_plot.png")
plt.savefig(combined_plot_path)
plt.close()
print(f"Saved: {combined_plot_path}")
