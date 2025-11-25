import os
import pandas as pd

# Path to your event_seg folder
event_seg_folder = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Tasks\event_seg"
processed_folder = os.path.join(event_seg_folder, "processed_data")

# Make sure processed_data folder exists
os.makedirs(processed_folder, exist_ok=True)

# Loop through all CSV files in event_seg
for file in os.listdir(event_seg_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(event_seg_folder, file)
        df = pd.read_csv(file_path)

        if "BoundaryTime(s)" in df.columns:
            # Filter out negative times
            df_clean = df[df["BoundaryTime(s)"] >= 0].copy()

            # Round all times to 3 decimal places
            #df_clean["BoundaryTime(s)"] = df_clean["BoundaryTime(s)"].round(3)

            # Add "_processed" to the filename
            base, ext = os.path.splitext(file)
            new_filename = f"CLEAN_{base}{ext}"

            # Save to processed_data folder
            save_path = os.path.join(processed_folder, new_filename)
            df_clean.to_csv(save_path, index=False)

            print(f"Processed {file} → {new_filename}, kept {len(df_clean)} rows.")
        else:
            print(f"Skipped {file}, no BoundaryTime(s) column found.")
