import pandas as pd

p = pd.read_csv(r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Clean_Analysis\Within_Between\CSV_Within_Across\PCA_withinAcross_timepoints.csv")
b = pd.read_csv(r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results\all_videos_consensus_boundaries.csv")

for vid in p["VideoName"].unique():
    sub = p[p["VideoName"] == vid]
    tmin, tmax = sub["Run_time"].min(), sub["Run_time"].max()
    btimes = b.loc[b.iloc[:,0] == vid, b.columns[1]].dropna().values
    bad = [x for x in btimes if (x - 30 < tmin or x + 30 > tmax)]
    print(f"{vid}: {len(btimes)} boundaries, {len(bad)} too close to edges")
